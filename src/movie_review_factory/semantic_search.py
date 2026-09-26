from __future__ import annotations

import hashlib
import math
import os
import shlex
from array import array
from functools import lru_cache
from pathlib import Path
from typing import Iterable

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_DIMENSION = 384
DEFAULT_MIN_SCORE = 0.35
SHORT_QUERY_MIN_SCORE = 0.50

# Bump when the embedding model, pooling, or stored vector format changes so every persisted record is recomputed on the next refresh.
EMBED_VERSION = 1

# Rows buffered into one float32 matrix per scoring chunk: bounds the transient copy (chunk * dimension * 4 bytes) without changing any score.
_SCORE_CHUNK = 8192


class EmbeddingUnavailable(RuntimeError):
    pass


_FILTER_KEYS = {
    "person", "action", "location", "object", "kind", "project",
    "scene_type", "source",
}


def parse_advanced_query(query: str) -> dict:
    """Parse optional field:value constraints while preserving free semantic text."""
    filters: dict[str, list[str]] = {key: [] for key in _FILTER_KEYS}
    free: list[str] = []
    try:
        parts = shlex.split(query)
    except ValueError:
        parts = query.split()
    for part in parts:
        key, sep, value = part.partition(":")
        normalized = key.casefold()
        if sep and normalized in _FILTER_KEYS and value.strip():
            filters[normalized].append(value.strip())
        else:
            free.append(part)
    return {
        "text": " ".join(free).strip(),
        **{key: values for key, values in filters.items() if values},
    }


def model_name() -> str:
    return os.environ.get("MRF_EMBED_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def cache_dir() -> Path:
    configured = os.environ.get("MRF_EMBED_CACHE")
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "movie-review-factory" / "embeddings"


def offline_mode() -> bool:
    value = os.environ.get("MRF_EMBED_OFFLINE") or os.environ.get("HF_HUB_OFFLINE") or ""
    return value.strip().casefold() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=4)
def _model(name: str, cache: str, offline: bool):
    try:
        from fastembed import TextEmbedding
    except ImportError as exc:
        raise EmbeddingUnavailable("fastembed is not installed") from exc
    try:
        return TextEmbedding(
            model_name=name,
            cache_dir=cache,
            local_files_only=offline,
        )
    except Exception as exc:
        raise EmbeddingUnavailable(f"embedding model unavailable: {name}") from exc


def _as_blob(values: Iterable[float]) -> tuple[int, bytes]:
    packed = array("f", (float(value) for value in values))
    if not packed:
        raise EmbeddingUnavailable("embedding model returned an empty vector")
    return len(packed), packed.tobytes()


def _embed(texts: list[str], *, query: bool) -> list[tuple[int, bytes]]:
    normalized = [" ".join(str(text).split()).strip() for text in texts]
    if any(not text for text in normalized):
        raise ValueError("embedding text must be non-empty")
    name = model_name()
    cache = cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    model = _model(name, str(cache), offline_mode())
    try:
        generator = model.query_embed(normalized) if query else model.passage_embed(normalized)
        return [_as_blob(vector) for vector in generator]
    except Exception as exc:
        raise EmbeddingUnavailable("embedding inference failed") from exc


def embed_passages(texts: list[str]) -> list[tuple[int, bytes]]:
    return _embed(texts, query=False)


def embed_query(text: str) -> tuple[int, bytes]:
    return _embed([text], query=True)[0]


def unpack_vector(blob: bytes, dimension: int) -> array:
    values = array("f")
    values.frombytes(blob)
    if len(values) != dimension:
        raise ValueError("embedding dimension does not match stored vector")
    return values


def cosine_similarity(
    left_blob: bytes,
    left_dimension: int,
    right_blob: bytes,
    right_dimension: int,
) -> float:
    if left_dimension != right_dimension:
        return -1.0
    left = unpack_vector(left_blob, left_dimension)
    right = unpack_vector(right_blob, right_dimension)
    dot = math.fsum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(math.fsum(a * a for a in left))
    right_norm = math.sqrt(math.fsum(b * b for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return -1.0
    return dot / (left_norm * right_norm)


def _load_numpy():
    """Return the numpy module, or None when it cannot be imported.

    Called per search so a missing/broken numpy degrades to the pure-python
    scoring below instead of failing the query.
    """
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def _score_rows_pure(query_blob: bytes, dimension: int, rows: list[dict]) -> list[float]:
    """Legacy per-row cosine scoring; kept as the fallback when numpy is absent."""
    return [
        cosine_similarity(
            query_blob, dimension, row["vector"], int(row["dimension"])
        )
        for row in rows
    ]


def _score_rows_numpy(np, query_blob: bytes, dimension: int, rows: list[dict]) -> list[float]:
    """Vectorized cosine against every row, same values as ``cosine_similarity``.

    Stored blobs are loaded once into a float32 matrix (in bounded chunks, so
    the transient copy stays small), the query is normalized once and one
    einsum per chunk produces all scores. Rows whose stored dimension differs
    from the query, and rows with a non-positive norm, keep the legacy -1.0
    score (the pure path never unpacks a mismatching blob).
    """
    scores = np.full(len(rows), -1.0)
    matching = [
        index for index, row in enumerate(rows)
        if int(row["dimension"]) == dimension
    ]
    if not matching:
        # The pure path returns -1.0 before touching any blob on a dim mismatch.
        return scores.tolist()
    if len(matching) == len(rows):
        blobs = [row["vector"] for row in rows]
    else:
        blobs = [rows[index]["vector"] for index in matching]
    if len(query_blob) != dimension * 4 or set(map(len, blobs)) != {dimension * 4}:
        raise ValueError("embedding dimension does not match stored vector")
    query = np.frombuffer(query_blob, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if query_norm <= 0:
        # zero norm -> -1.0 for every row, like the pure path; a NaN norm fails this test and keeps flowing, again like the pure path.
        return scores.tolist()
    normalized_query = query / query_norm
    for start in range(0, len(matching), _SCORE_CHUNK):
        chunk = matching[start:start + _SCORE_CHUNK]
        payload = b"".join(blobs[start:start + _SCORE_CHUNK])
        matrix = np.frombuffer(payload, dtype=np.float32).reshape(len(chunk), dimension)
        # Plain einsum (never the BLAS gemv behind ``@``): its per-row
        # reduction is bit-identical for identical rows, so equal vectors keep
        # an exactly equal score and the tie-break order stays stable.
        row_norms = np.sqrt(np.einsum("ij,ij->i", matrix, matrix))
        dots = np.einsum("ij,j->i", matrix, normalized_query)
        usable = ~(row_norms <= 0)  # True for positive and NaN norms
        with np.errstate(invalid="ignore", divide="ignore"):
            computed = np.where(usable, dots.astype(np.float64) / row_norms, -1.0)
        scores[chunk] = computed
    return scores.tolist()


def _rank_rows_pure(
    scores: list[float], starts: list[float], threshold: float
) -> list[int]:
    """Indices passing the threshold, ordered like the legacy ``list.sort``."""
    kept = [index for index, score in enumerate(scores) if score >= threshold]
    kept.sort(key=lambda index: (-scores[index], starts[index], index))
    return kept


def _rank_rows_numpy(
    np, scores: list[float], starts: list[float], threshold: float
) -> list[int]:
    """Same ranking as ``_rank_rows_pure``: score desc, then original start
    time, then original row index (stable, so ties keep insertion order)."""
    if not scores:
        return []
    values = np.asarray(scores, dtype=np.float64)
    times = np.asarray(starts, dtype=np.float64)
    kept = np.flatnonzero(values >= threshold)
    if kept.size == 0:
        return []
    order = np.lexsort((kept, times[kept], -values[kept]))
    return kept[order].tolist()


def _search_result(kind: str, row: dict, score: float) -> dict:
    """Build one result dict exactly as the legacy per-row loop did."""
    if kind == "visual":
        return {
            "kind": "visual",
            "shot_id": int(row["shot_id"]),
            "start_seconds": float(row["start_seconds"]),
            "end_seconds": float(row["end_seconds"]),
            "text": row["text"],
            "score": float(score),
        }
    return {
        "kind": "transcript",
        "segment_id": int(row["transcript_segment_id"]),
        "start_seconds": float(row["start_seconds"]),
        "end_seconds": float(row["end_seconds"]),
        "text": row["text"],
        "speaker": row.get("speaker"),
        "score": float(score),
    }


def visual_text(row: dict) -> str:
    parts = [
        str(row.get("description") or ""),
        *[str(value) for value in row.get("tags") or []],
        *[str(value) for value in row.get("people") or []],
        *[str(value) for value in row.get("actions") or []],
    ]
    return " ".join(part.strip() for part in parts if part and part.strip())


def content_hash(text: str) -> str:
    """Stable sha256 of the UTF-8 encoded text exactly as it gets embedded."""
    normalized = " ".join(str(text).split()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _pending_records(
    docs: list[tuple[int, str]],
    stored: dict[int, dict],
    model: str,
) -> tuple[list[tuple[int, str, str]], int]:
    """Split documents into records that need embedding plus how many are reused."""
    pending: list[tuple[int, str, str]] = []
    skipped = 0
    for record_id, text in docs:
        digest = content_hash(text)
        row = stored.get(record_id)
        if (
            row is not None
            and str(row.get("model")) == model
            and row.get("embed_version") == EMBED_VERSION
            and row.get("content_hash") == digest
        ):
            skipped += 1
            continue
        pending.append((record_id, text, digest))
    return pending, skipped


def refresh_store_embeddings(store) -> dict:
    model = model_name()
    visuals = store.list_visual_observations()
    visual_by_shot = {int(row["shot_id"]): visual_text(row) for row in visuals}
    identity_by_shot: dict[int, list[str]] = {}
    for track in store.list_person_tracks():
        label = str(track.get("label") or "").strip()
        alias = str(track.get("alias") or "").strip()
        description = str(
            track.get("appearance_summary") or track.get("description") or ""
        ).strip()
        traits = " ".join(
            str(item.get("value") or "").strip()
            for item in track.get("traits") or []
            if str(item.get("value") or "").strip()
        )
        identity_text = " ".join(
            part for part in (label, alias, description, traits) if part
        )
        if not identity_text:
            continue
        for appearance in track.get("appearances") or []:
            shot_id = int(appearance["shot_id"])
            identity_by_shot.setdefault(shot_id, []).append(identity_text)
    shots = store.list_shots(1)
    shot_docs = [
        (
            int(shot.id),
            " ".join(
                part for part in (
                    visual_by_shot.get(int(shot.id), ""),
                    " ".join(identity_by_shot.get(int(shot.id), [])),
                    str(shot.label or ""),
                )
                if part
            ),
        )
        for shot in shots if shot.id is not None
    ]
    shot_docs = [(shot_id, text.strip()) for shot_id, text in shot_docs if text.strip()]
    segments = [segment for segment in store.list_transcript(1) if segment.id is not None]
    segment_docs = [(int(segment.id), segment.text) for segment in segments]

    shot_rows = {
        int(row["shot_id"]): row for row in store.list_shot_embeddings()
    }
    transcript_rows = {
        int(row["transcript_segment_id"]): row
        for row in store.list_transcript_embeddings()
    }
    pending_shots, skipped_shots = _pending_records(shot_docs, shot_rows, model)
    pending_segments, skipped_segments = _pending_records(
        segment_docs, transcript_rows, model
    )

    if pending_shots:
        vectors = embed_passages([text for _, text, _ in pending_shots])
    else:
        vectors = []
    store.sync_shot_embeddings(
        [shot_id for shot_id, _ in shot_docs],
        [
            (shot_id, text, model, dimension, blob, digest, EMBED_VERSION)
            for (shot_id, text, digest), (dimension, blob) in zip(pending_shots, vectors)
        ],
    )

    if pending_segments:
        vectors = embed_passages([text for _, text, _ in pending_segments])
    else:
        vectors = []
    store.sync_transcript_embeddings(
        [segment_id for segment_id, _ in segment_docs],
        [
            (segment_id, text, model, dimension, blob, digest, EMBED_VERSION)
            for (segment_id, text, digest), (dimension, blob) in zip(
                pending_segments, vectors
            )
        ],
    )

    changed = len(pending_shots) + len(pending_segments)
    return {
        "model": model,
        "embed_version": EMBED_VERSION,
        "embedded": changed,
        "changed": changed,
        "skipped": skipped_shots + skipped_segments,
        "shot_count": len(shot_docs),
        "transcript_count": len(segments),
    }


def minimum_score(query: str) -> float:
    token_count = len([part for part in query.split() if part])
    return SHORT_QUERY_MIN_SCORE if token_count <= 2 else DEFAULT_MIN_SCORE


def similar_shots(
    store,
    shot_id: int,
    *,
    limit: int = 12,
    min_score: float = 0.20,
) -> list[dict]:
    model = model_name()
    rows = store.list_shot_embeddings(model)
    source = next((row for row in rows if int(row["shot_id"]) == int(shot_id)), None)
    if source is None:
        raise KeyError(shot_id)
    results: list[dict] = []
    for row in rows:
        candidate_id = int(row["shot_id"])
        if candidate_id == int(shot_id):
            continue
        score = cosine_similarity(
            source["vector"],
            int(source["dimension"]),
            row["vector"],
            int(row["dimension"]),
        )
        if score < float(min_score):
            continue
        results.append({
            "kind": "visual",
            "shot_id": candidate_id,
            "start_seconds": float(row["start_seconds"]),
            "end_seconds": float(row["end_seconds"]),
            "text": row["text"],
            "score": score,
        })
    results.sort(key=lambda item: (-float(item["score"]), item["start_seconds"]))
    return results[:max(1, limit)]


def search_store(
    store,
    query: str,
    *,
    limit: int = 20,
    min_score: float | None = None,
) -> list[dict]:
    dimension, query_blob = embed_query(query)
    threshold = minimum_score(query) if min_score is None else float(min_score)
    model = model_name()
    shots = store.list_shot_embeddings(model)
    segments = store.list_transcript_embeddings(model)
    rows = [*shots, *segments]
    split = len(shots)
    starts = [row["start_seconds"] for row in rows]
    np = _load_numpy()
    if np is None:
        scores = _score_rows_pure(query_blob, dimension, rows)
        order = _rank_rows_pure(scores, starts, threshold)
    else:
        scores = _score_rows_numpy(np, query_blob, dimension, rows)
        order = _rank_rows_numpy(np, scores, starts, threshold)
    return [
        _search_result(
            "visual" if index < split else "transcript",
            rows[index],
            scores[index],
        )
        for index in order[: max(1, limit)]
    ]
