#!/usr/bin/env python3
"""Roadmap #12 - measured-first ANN policy benchmark.

Measures the current linear cosine scan used by ``semantic_search.search_store``
(``store.list_shot_embeddings`` + ``store.list_transcript_embeddings`` +
``semantic_search.cosine_similarity`` per row) and reports, from measurements
only:

  * p50 / p95 latency per corpus scale (real job DB + synthetic 10k / 100k),
  * memory per corpus scale (vector payload, corpus RSS, scan transient),
  * the vector count above which p95 crosses the 100 ms budget,
  * a concrete DEFER / ANN decision derived from those numbers.

Read-only with respect to ``src/`` and ``docs/``. Optional ANN backends
(FAISS / sqlite-vector / hnswlib) are measured only when already installed;
this script never installs a package and never changes dependencies.

Usage:
    py -3 scripts\\bench_ann.py [--db PATH] [--repeats N]
                                [--synthetic 100,300,1000,10000,100000]
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import random
import shutil
import sqlite3
import statistics
import sys
import tempfile
import time
import tracemalloc
from array import array
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from movie_review_factory import semantic_search  # noqa: E402
from movie_review_factory.media_store import MediaStore  # noqa: E402
from movie_review_factory.models import (  # noqa: E402
    MediaAsset,
    Shot,
    TranscriptSegment,
)

DIMENSION = semantic_search.DEFAULT_DIMENSION
SEED = 20260926
FIXTURE_MODEL = "bench-fixture"
TARGET_MS = 100.0
RESULT_LIMIT = 20
# Production filter for a long query (semantic_search.minimum_score).
QUERY_THRESHOLD = semantic_search.DEFAULT_MIN_SCORE
# Small points bracket the 100 ms crossing; 10k / 100k are the roadmap scales.
DEFAULT_SYNTHETIC = "100,300,1000,10000,100000"
# tracemalloc slows a scan ~1.6x, so it never runs inside a timed pass.


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #
def find_job_db(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"--db not found: {path}")
        return path
    jobs = REPO_ROOT / "jobs"
    if jobs.is_dir():
        candidates = sorted(jobs.glob("*/media_index.sqlite3"))
        if candidates:
            return candidates[-1]
    return None


def random_blob(rng: random.Random, dimension: int = DIMENSION) -> tuple[int, bytes]:
    """Fixed-seed random unit vector, same float32 packing as ``_as_blob``."""
    packed = array("f", (rng.random() - 0.5 for _ in range(dimension)))
    norm = math.sqrt(math.fsum(v * v for v in packed)) or 1.0
    return len(packed), array("f", (v / norm for v in packed)).tobytes()


def build_fixture_database(path: Path, shot_count: int, segment_count: int) -> None:
    """Small indexed job DB in tests/test_media_store.py style (used only if
    no ``jobs/*/media_index.sqlite3`` exists)."""
    rng = random.Random(SEED)
    with MediaStore(path) as store:
        store.migrate()
        asset = store.add_media_asset(
            MediaAsset(path=Path("fixture.mp4"), duration_seconds=7200.0)
        )
        for i in range(shot_count):
            store.add_shot(
                Shot(
                    media_asset_id=asset.id,
                    start_seconds=float(i * 4.0),
                    end_seconds=float(i * 4.0 + 3.5),
                    label=f"Shot {i}",
                )
            )
        for i in range(segment_count):
            store.add_transcript_segment(
                TranscriptSegment(
                    media_asset_id=asset.id,
                    start_seconds=float(i * 2.0),
                    end_seconds=float(i * 2.0 + 1.8),
                    text=f"fixture line {i}",
                    speaker=f"Speaker {i % 3}",
                )
            )
        store.replace_shot_embeddings(
            [
                (int(shot.id), f"shot text {i}", FIXTURE_MODEL, dim, blob)
                for i, shot in enumerate(store.list_shots(asset.id))
                for dim, blob in (random_blob(rng),)
            ]
        )
        store.replace_transcript_embeddings(
            [
                (int(segment.id), segment.text, FIXTURE_MODEL, dim, blob)
                for segment in store.list_transcript(asset.id)
                for dim, blob in (random_blob(rng),)
            ]
        )


def load_rows(db_path: Path) -> tuple[list[dict], float, str]:
    """One ``search_store`` row fetch: both embedding tables, same joins.

    A job DB older than the current schema is never written: the read is redone
    on a migrated temporary copy instead, so the job artifact stays untouched.
    """
    try:
        started = time.perf_counter()
        with MediaStore(db_path) as store:
            rows = store.list_shot_embeddings(None)
            rows += store.list_transcript_embeddings(None)
        return rows, (time.perf_counter() - started) * 1000.0, ""
    except sqlite3.OperationalError as exc:
        note = f"job DB schema is stale ({exc}); read via a migrated temp copy"
    with tempfile.TemporaryDirectory(prefix="mrf-bench-ann-db-") as tmp:
        copy = Path(tmp) / db_path.name
        shutil.copyfile(db_path, copy)
        with MediaStore(copy) as store:
            store.migrate()
            started = time.perf_counter()
            rows = store.list_shot_embeddings(None)
            rows += store.list_transcript_embeddings(None)
            elapsed = (time.perf_counter() - started) * 1000.0
    return rows, elapsed, note


def synthetic_rows(count: int, seed_offset: int = 0) -> list[dict]:
    rng = random.Random(SEED + seed_offset)
    rows: list[dict] = []
    for i in range(count):
        dim, blob = random_blob(rng)
        rows.append(
            {
                "kind": "synthetic",
                "shot_id": i,
                "start_seconds": float(i),
                "text": f"synthetic {i}",
                "dimension": dim,
                "vector": blob,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# scans under test
# --------------------------------------------------------------------------- #
def scan_current(rows: list[dict], query_blob: bytes) -> list[dict]:
    """Current production shape: score every row, build the record, filter,
    sort, truncate (see ``semantic_search.search_store``)."""
    results: list[dict] = []
    for row in rows:
        score = semantic_search.cosine_similarity(
            query_blob, DIMENSION, row["vector"], int(row["dimension"])
        )
        results.append(
            {
                "kind": row.get("kind"),
                "start_seconds": float(row["start_seconds"]),
                "text": row.get("text"),
                "score": score,
            }
        )
    results = [item for item in results if item["score"] >= QUERY_THRESHOLD]
    results.sort(key=lambda item: (-item["score"], item["start_seconds"]))
    return results[:RESULT_LIMIT]


def hoisted_score(
    query: array, query_norm: float, blob: bytes, dimension: int
) -> float:
    if dimension != DIMENSION:
        return -1.0
    candidate = array("f")
    candidate.frombytes(blob)
    dot = math.fsum(a * b for a, b in zip(query, candidate))
    candidate_norm = math.sqrt(math.fsum(v * v for v in candidate))
    if query_norm <= 0 or candidate_norm <= 0:
        return -1.0
    return dot / (query_norm * candidate_norm)


def hoisted_query(query_blob: bytes) -> tuple[array, float]:
    query = array("f")
    query.frombytes(query_blob)
    if len(query) != DIMENSION:
        raise ValueError("query dimension mismatch")
    return query, math.sqrt(math.fsum(v * v for v in query))


def scan_hoisted_query_norm(rows: list[dict], query_blob: bytes) -> list[dict]:
    """Reference only: identical math, but the query norm is computed once per
    query instead of once per candidate. Requires a src change, so this is NOT
    the current path - it only prices the cheapest pre-ANN fix."""
    query, query_norm = hoisted_query(query_blob)
    results: list[dict] = []
    for row in rows:
        score = hoisted_score(
            query, query_norm, row["vector"], int(row["dimension"])
        )
        results.append(
            {
                "kind": row.get("kind"),
                "start_seconds": float(row["start_seconds"]),
                "text": row.get("text"),
                "score": score,
            }
        )
    results = [item for item in results if item["score"] >= QUERY_THRESHOLD]
    results.sort(key=lambda item: (-item["score"], item["start_seconds"]))
    return results[:RESULT_LIMIT]


# --------------------------------------------------------------------------- #
# measurement helpers
# --------------------------------------------------------------------------- #
def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    position = (len(data) - 1) * p / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return data[int(position)]
    return data[low] + (data[high] - data[low]) * (position - low)


def repeats_for(count: int, configured: int) -> int:
    """Bound total runtime: large corpora get fewer (still >= 3) measured reps."""
    if count >= 50000:
        return max(3, min(configured, 3))
    if count >= 5000:
        return max(3, min(configured, 5))
    if count >= 500:
        return max(3, min(configured, 10))
    return max(3, configured)


def timed_scan(scan, rows: list[dict], query_blob: bytes, repeats: int) -> dict:
    """Timed pass WITHOUT tracemalloc (it inflates runtime ~1.6x)."""
    scan(rows, query_blob)  # warmup, also faults the pages in
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        scan(rows, query_blob)
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "repeats": repeats,
        "p50": percentile(samples, 50),
        "p95": percentile(samples, 95),
        "max": max(samples),
        "mean": statistics.fmean(samples),
    }


def transient_scan_mb(scan, rows: list[dict], query_blob: bytes) -> float:
    """Peak Python allocation made while one full scan runs (separate pass)."""
    tracemalloc.start()
    try:
        scan(rows, query_blob)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / (1024 * 1024)


def rss_mb() -> float | None:
    try:
        import psutil  # already installed; optional, never a new dependency
    except ImportError:
        if sys.platform != "win32":
            return None
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            return None
        return counters.WorkingSetSize / (1024 * 1024)
    return psutil.Process().memory_info().rss / (1024 * 1024)


def measure_corpus_footprint(build) -> tuple[list[dict], float | None]:
    """Build a corpus and report the RSS it adds (vector rows + blobs)."""
    before = rss_mb()
    rows = build()
    after = rss_mb()
    if before is None or after is None:
        return rows, None
    return rows, max(0.0, after - before)


def numpy_reference(rows: list[dict], query_blob: bytes) -> dict | None:
    """Exact linear search vectorized with numpy (already installed, reference)."""
    if importlib.util.find_spec("numpy") is None:
        return None
    import numpy as np

    matrix = np.frombuffer(
        b"".join(row["vector"] for row in rows), dtype=np.float32
    ).reshape(len(rows), DIMENSION)
    query = np.frombuffer(query_blob, dtype=np.float32)
    query = query / float(np.linalg.norm(query))
    samples: list[float] = []
    for _ in range(5):
        started = time.perf_counter()
        scores = matrix @ query
        samples.append((time.perf_counter() - started) * 1000.0)
    if not np.isfinite(scores).all():
        raise RuntimeError("numpy reference produced non-finite scores")
    return {"p50": percentile(samples, 50), "p95": percentile(samples, 95)}


def faiss_reference(rows: list[dict], query_blob: bytes) -> dict | None:
    """FAISS exact + HNSW, measured only if FAISS is already installed."""
    if importlib.util.find_spec("faiss") is None:
        return None
    import faiss
    import numpy as np

    matrix = np.frombuffer(
        b"".join(row["vector"] for row in rows), dtype=np.float32
    ).reshape(len(rows), DIMENSION).copy()
    query = np.frombuffer(query_blob, dtype=np.float32).reshape(1, -1).copy()
    out: dict = {"backend": "faiss"}
    for name, index in (
        ("exact", faiss.IndexFlatIP(DIMENSION)),
        ("hnsw", faiss.IndexHNSWFlat(DIMENSION, 32)),
    ):
        index.add(matrix)
        samples: list[float] = []
        for _ in range(5):
            started = time.perf_counter()
            index.search(query, RESULT_LIMIT)
            samples.append((time.perf_counter() - started) * 1000.0)
        out[name] = {"p50": percentile(samples, 50), "p95": percentile(samples, 95)}
    return out


def sqlite_vec_reference(rows: list[dict], query_blob: bytes) -> dict | None:
    """sqlite-vector k-NN, measured only if sqlite_vec is already installed."""
    if importlib.util.find_spec("sqlite_vec") is None:
        return None
    import sqlite3

    import sqlite_vec

    connection = sqlite3.connect(":memory:")
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.execute(
            f"CREATE VIRTUAL TABLE bench_vec USING vec0(embedding float[{DIMENSION}])"
        )
        connection.executemany(
            "INSERT INTO bench_vec(rowid, embedding) VALUES (?, ?)",
            ((i + 1, row["vector"]) for i, row in enumerate(rows)),
        )
        samples: list[float] = []
        for _ in range(5):
            started = time.perf_counter()
            connection.execute(
                "SELECT rowid, distance FROM bench_vec WHERE embedding MATCH ? "
                "AND k = ?",
                (query_blob, RESULT_LIMIT),
            ).fetchall()
            samples.append((time.perf_counter() - started) * 1000.0)
        return {"p50": percentile(samples, 50), "p95": percentile(samples, 95)}
    finally:
        connection.close()


def backend_status() -> list[tuple[str, str, bool]]:
    status = []
    for module, label in (
        ("faiss", "FAISS"),
        ("sqlite_vec", "sqlite-vector"),
        ("hnswlib", "hnswlib"),
        ("numpy", "numpy"),
    ):
        installed = importlib.util.find_spec(module) is not None
        status.append((label, module, installed))
    return status


# --------------------------------------------------------------------------- #
# conclusion
# --------------------------------------------------------------------------- #
def interpolate_threshold(points: list[tuple[int, float]]) -> tuple[float | None, str]:
    """Vector count where p95 = TARGET_MS, log-linear between measured points."""
    measured = sorted((n, p95) for n, p95 in points if not math.isnan(p95))
    if len(measured) < 2:
        return None, "not enough measured points"
    if all(p95 < TARGET_MS for _, p95 in measured):
        biggest = measured[-1]
        return None, f"> {biggest[0]} vectors (largest measured point still passes)"
    if all(p95 >= TARGET_MS for _, p95 in measured):
        smallest = measured[0]
        return None, f"<= {smallest[0]} vectors (smallest measured point already fails)"
    for (n_low, p_low), (n_high, p_high) in zip(measured, measured[1:]):
        if p_low < TARGET_MS <= p_high:
            ratio = math.log(TARGET_MS / p_low) / math.log(p_high / p_low)
            estimate = math.exp(math.log(n_low) + ratio * math.log(n_high / n_low))
            return estimate, (
                f"~{int(round(estimate))} vectors (interpolated between measured "
                f"{n_low} ({p_low:.1f} ms) and {n_high} ({p_high:.1f} ms))"
            )
    return None, "no crossing found"


def decide(
    real_n: int,
    real_p95: float,
    threshold: float | None,
    threshold_provenance: str,
    hoisted_threshold: float | None,
    hoisted_provenance: str,
    cost_us_per_vector: float,
    scale_numbers: dict[str, tuple[int, float]],
    installed_alts: list[str],
) -> str:
    """Data-driven verdict: DEFER / FIX-THEN-DEFER / START ANN."""
    vectors_per_job = max(1, real_n)
    budget_vectors = int(TARGET_MS * 1000 / cost_us_per_vector)
    lines = [
        f"  threshold, current path      : {threshold_provenance}",
        f"  threshold, query-norm hoisted: {hoisted_provenance}",
        f"  cost per vector (current)    : {cost_us_per_vector:.0f} us per vector"
        f" -> {TARGET_MS:.0f} ms buys ~{budget_vectors} vectors",
        f"  current corpus               : {real_n} vectors = 1 indexed job"
        f" (jobs/<name>/media_index.sqlite3)",
        "",
    ]

    if threshold is None and threshold_provenance.startswith(">"):
        lines.append(
            "  recommendation : DEFER - linear scan measured under "
            f"{TARGET_MS:.0f} ms at every scale up to "
            f"{threshold_provenance.split()[1]} vectors; no ANN work now."
        )
        lines.append(
            "  ANN trigger    : re-run this script once a single DB passes that "
            "count; start ANN when p95 > "
            f"{TARGET_MS:.0f} ms."
        )
        return "\n".join(lines)

    if threshold is not None and real_n <= threshold:
        lines.append(
            "  recommendation : DEFER - do not implement ANN now. Measured p95 "
            f"({real_p95:.1f} ms at {real_n} vectors) is under the "
            f"{TARGET_MS:.0f} ms budget."
        )
        lines.append(
            f"  ANN trigger    : start ANN when one DB passes ~{int(threshold)} "
            f"vectors (~{threshold / vectors_per_job:.1f} jobs at the measured "
            f"{real_n} vectors/job), i.e. when a re-run reports p95 > "
            f"{TARGET_MS:.0f} ms."
        )
        return "\n".join(lines)

    ten_k = scale_numbers.get("synthetic 10000")
    hundred_k = scale_numbers.get("synthetic 100000")
    lines.append(
        "  recommendation : START ANN now, behind a flag, keeping the linear scan "
        "as the correctness fallback."
    )
    lines.append(
        f"  why            : measured p95 = {real_p95:.1f} ms > {TARGET_MS:.0f} ms "
        f"at the current {real_n} vectors ({cost_us_per_vector:.0f} us/vector), so "
        "the budget is already exceeded by ONE indexed job, not by a future "
        "large library."
    )
    if hoisted_threshold is not None:
        lines.append(
            "                     cheapest linear fix (hoist the query norm out of "
            f"the per-row cosine, src change) reaches only ~{int(hoisted_threshold)} "
            f"vectors (~{hoisted_threshold / vectors_per_job:.1f} jobs), so it buys "
            f"less than one extra job of {real_n} vectors."
        )
    if ten_k is not None:
        scale_text = f"10k vectors -> {ten_k[1]:.0f} ms"
        if hundred_k is not None:
            scale_text += f", 100k -> {hundred_k[1]:.0f} ms ({hundred_k[1] / 1000:.1f} s)"
        lines.append(
            f"                     linear growth confirmed: {scale_text}; "
            "unusable for a merged multi-job index."
        )
    if installed_alts:
        lines.append(
            "  measured alt   : "
            + ", ".join(installed_alts)
            + " (table 3) - exact-search ceiling, not an ANN backend; FAISS/"
            "sqlite-vector numbers must come from a run where they are installed."
        )
    lines.append(
        "  backend        : prefer sqlite-vector (no new Python dependency when the "
        "extension ships with the runtime), else FAISS HNSW; neither is installed "
        "in this environment, so no ANN number could be measured here - add the "
        "backend behind a flag, then re-run this script."
    )
    lines.append(
        "  re-check       : acceptance = p95 < "
        f"{TARGET_MS:.0f} ms on the real job DB after the backend lands."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.2f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measured-first ANN benchmark for the linear cosine scan."
    )
    parser.add_argument("--db", help="path to a media_index.sqlite3 of an indexed job")
    parser.add_argument("--repeats", type=int, default=20, help="measured reps per scale")
    parser.add_argument(
        "--synthetic",
        default=DEFAULT_SYNTHETIC,
        help="comma separated synthetic vector counts",
    )
    parser.add_argument("--fixture-shots", type=int, default=400)
    parser.add_argument("--fixture-segments", type=int, default=1200)
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument(
        "--skip-transient",
        action="store_true",
        help="skip the tracemalloc memory pass (saves ~1 min at 100k)",
    )
    args = parser.parse_args(argv)

    db_path = find_job_db(args.db)
    source = ""
    fetch_ms = float("nan")
    load_note = ""
    rows: list[dict] = []
    rss_before_real = rss_mb()
    if db_path is not None:
        source = f"job DB {db_path}"
        try:
            rows, fetch_ms, load_note = load_rows(db_path)
        except sqlite3.Error as exc:
            # Unreadable/corrupt job DB: measure a fixture rather than abort.
            print(f"[bench_ann] cannot read {db_path}: {exc}")
            rows = []
        if not rows:
            print(
                f"[bench_ann] {db_path} has no embeddings; using a fixture DB instead"
            )
    if not rows:
        with tempfile.TemporaryDirectory(prefix="mrf-bench-ann-") as tmp:
            fixture = Path(tmp) / "media_index.sqlite3"
            build_fixture_database(fixture, args.fixture_shots, args.fixture_segments)
            rows, fetch_ms, load_note = load_rows(fixture)
            source = (
                f"fixture DB ({args.fixture_shots} shot + {args.fixture_segments} "
                f"transcript embeddings, seed={SEED})"
            )
    real_rows, real_fetch = rows, fetch_ms
    real_n = len(real_rows)
    rss_after_real = rss_mb()
    real_footprint = None
    if rss_before_real is not None and rss_after_real is not None:
        real_footprint = max(0.0, rss_after_real - rss_before_real)
    shot_rows = sum(1 for r in real_rows if "shot_id" in r)
    transcript_rows = sum(1 for r in real_rows if "transcript_segment_id" in r)

    rng = random.Random(SEED)
    _, query_blob = random_blob(rng)

    print("=" * 100)
    print("Roadmap #12 - ANN benchmark (measured-first, linear cosine scan)")
    print("=" * 100)
    print(f"vector dimension : {DIMENSION} (float32 blobs, {DIMENSION * 4} B/vector)")
    print(f"budget           : p95 < {TARGET_MS:.0f} ms of scan time per query")
    print(f"data source      : {source}")
    if load_note:
        print(f"                 : {load_note}")
    print(
        f"real corpus      : {real_n} embedding vectors "
        f"({shot_rows} shot + {transcript_rows} transcript)"
    )
    print(f"sqlite row fetch : {fmt(real_fetch)} ms per search_store fetch (both tables)")
    print("query embedding  : excluded (model load/inference identical for every backend)")
    print(f"synthetic seed   : {SEED}")
    print("timing           : tracemalloc OFF during timed passes")
    print("-" * 100)
    print("optional backends (measured only if already installed, never installed here):")
    for label, module, installed in backend_status():
        if module == "hnswlib":
            note = "not measured here (FAISS IndexHNSWFlat is the ANN reference)"
        elif not installed:
            note = "NOT installed -> skipped, deps unchanged"
        elif module == "numpy":
            note = "installed -> exact-search reference (table 3)"
        else:
            note = "installed -> measured in table 3"
        print(f"  {label:<14} {note} [{module}]")
    print("-" * 100)

    corpora: list[tuple[str, list[dict], float, float | None]] = [
        ("real job DB", real_rows, real_fetch, real_footprint)
    ]
    if not args.skip_synthetic:
        for raw in str(args.synthetic).split(","):
            raw = raw.strip()
            if not raw:
                continue
            count = int(raw)
            built, footprint = measure_corpus_footprint(
                lambda count=count: synthetic_rows(count)
            )
            corpora.append((f"synthetic {count}", built, float("nan"), footprint))

    results: list[dict] = []
    for label, corpus, fetch, footprint in corpora:
        repeats = repeats_for(len(corpus), args.repeats)
        current = timed_scan(scan_current, corpus, query_blob, repeats)
        hoisted = timed_scan(scan_hoisted_query_norm, corpus, query_blob, repeats)
        transient = (
            float("nan")
            if args.skip_transient
            else transient_scan_mb(scan_current, corpus, query_blob)
        )
        payload_mb = sum(len(row["vector"]) for row in corpus) / (1024 * 1024)
        results.append(
            {
                "label": label,
                "n": len(corpus),
                "fetch": fetch,
                "current": current,
                "hoisted": hoisted,
                "transient_mb": transient,
                "payload_mb": payload_mb,
                "footprint_mb": footprint,
                "numpy": numpy_reference(corpus, query_blob),
                "faiss": faiss_reference(corpus, query_blob),
                "sqlite_vec": sqlite_vec_reference(corpus, query_blob),
            }
        )
        print(
            f"  measured {label:<16} n={len(corpus):>7} "
            f"p50={current['p50']:9.2f} ms  p95={current['p95']:9.2f} ms  "
            f"(reps={repeats})"
        )

    # sanity: the reference scan must score the real corpus identically
    sample = real_rows[: min(200, real_n)]
    query, query_norm = hoisted_query(query_blob)
    mismatches = [
        row
        for row in sample
        if abs(
            semantic_search.cosine_similarity(
                query_blob, DIMENSION, row["vector"], int(row["dimension"])
            )
            - hoisted_score(query, query_norm, row["vector"], int(row["dimension"]))
        )
        > 1e-5
    ]
    print()
    print(
        "sanity: current vs query-norm-hoisted scores identical on "
        f"{len(sample)} real rows: {not mismatches}"
    )
    if mismatches:
        print("        WARNING: reference implementation disagrees with production path")

    print()
    print(f"TABLE 1 - LINEAR COSINE SCAN LATENCY (current production path, "
          f"budget p95 < {TARGET_MS:.0f} ms)")
    header = (
        f"{'scale':<16}{'vectors':>9}{'fetch ms':>10}{'p50 ms':>11}{'p95 ms':>11}"
        f"{'max ms':>11}{'us/vec':>10}{'e2e p95':>11}{'reps':>6}"
    )
    print(header)
    for row in results:
        current = row["current"]
        us_vec = current["p95"] * 1000.0 / row["n"]
        e2e = current["p95"] + row["fetch"] if not math.isnan(row["fetch"]) else float("nan")
        print(
            f"{row['label']:<16}{row['n']:>9}{fmt(row['fetch']):>10}"
            f"{fmt(current['p50']):>11}{fmt(current['p95']):>11}"
            f"{fmt(current['max']):>11}{us_vec:>10.1f}{fmt(e2e):>11}"
            f"{current['repeats']:>6}"
        )

    print()
    print("TABLE 2 - MEMORY (payload = stored vectors, footprint = RSS added by the")
    print("           corpus, transient = peak Python allocation during one scan)")
    print(
        f"{'scale':<16}{'vectors':>9}{'payload MiB':>13}{'corpus RSS MiB':>17}"
        f"{'scan transient MiB':>21}"
    )
    for row in results:
        print(
            f"{row['label']:<16}{row['n']:>9}{fmt(row['payload_mb']):>13}"
            f"{fmt(row['footprint_mb']):>17}{fmt(row['transient_mb']):>21}"
        )
    if args.skip_transient:
        print("  (transient pass skipped by --skip-transient)")

    print()
    print("TABLE 3 - REFERENCE PATHS (already-installed deps only; NOT the current path)")

    def pair(value: dict | None, *keys: str) -> str:
        if value is None:
            return "-"
        part = value
        for key in keys:
            part = part.get(key) if isinstance(part, dict) else None
            if part is None:
                return "-"
        return f"{part['p50']:.2f} / {part['p95']:.2f}"

    print(
        f"{'scale':<16}{'hoisted p50/p95':>19}{'numpy exact p50/p95':>22}"
        f"{'faiss exact p50/p95':>22}{'faiss hnsw p50/p95':>21}"
        f"{'sqlite-vec p50/p95':>21}"
    )
    for row in results:
        print(
            f"{row['label']:<16}"
            f"{pair(row['hoisted']):>19}"
            f"{pair(row['numpy']):>22}"
            f"{pair(row['faiss'], 'exact'):>22}"
            f"{pair(row['faiss'], 'hnsw'):>21}"
            f"{pair(row['sqlite_vec']):>21}"
        )
    print("  hoisted = same cosine with the query norm computed once per query")
    print("            (prices the cheapest src fix, not the current path)")

    current_points = [(row["n"], row["current"]["p95"]) for row in results]
    hoisted_points = [(row["n"], row["hoisted"]["p95"]) for row in results]
    threshold, provenance = interpolate_threshold(current_points)
    hoisted_threshold, hoisted_provenance = interpolate_threshold(hoisted_points)
    largest = max(results, key=lambda row: row["n"])
    cost_us = largest["current"]["p95"] * 1000.0 / largest["n"]
    real_result = results[0]
    scale_numbers = {
        row["label"]: (row["n"], row["current"]["p95"]) for row in results
    }

    print()
    print("CONCLUSION (derived only from the tables above)")
    installed_alts = [
        label
        for label, module, installed in backend_status()
        if installed and module != "hnswlib"
    ]
    print(
        decide(
            real_result["n"],
            real_result["current"]["p95"],
            threshold,
            provenance,
            hoisted_threshold,
            hoisted_provenance,
            cost_us,
            scale_numbers,
            installed_alts,
        )
    )
    print()
    print("  method: p95 of repeated full scans (tracemalloc off), threshold = the")
    print(f"  vector count where p95 = {TARGET_MS:.0f} ms found by log-linear")
    print("  interpolation between the two measured points that bracket it;")
    print("  per-vector cost taken at the largest measured scale.")
    print("  sqlite fetch is included in 'e2e p95' for the real DB only; embed model")
    print("  load/inference is excluded because it is identical for every backend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
