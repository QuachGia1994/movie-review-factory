from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal, get_args

from pydantic import BaseModel, Field, model_validator

StageStatus = Literal["pending", "running", "ready", "failed", "skipped", "cancelled"]
ContentAgentMode = Literal["scaffold", "claude", "agy"]
CONTENT_AGENT_MODES = get_args(ContentAgentMode)
# docs/watermark-removal.md "Choosing a removal method" explains the tradeoffs.
WatermarkMethod = Literal["propainter", "delogo", "blur"]
WATERMARK_METHODS = get_args(WatermarkMethod)


class CreativeBrief(BaseModel):
    review_thesis: str = Field(default="", max_length=500)
    tone: str = Field(default="", max_length=120)
    target_audience: str = Field(default="", max_length=200)
    spoiler_policy: Literal["unspecified", "none", "limited", "full"] = "unspecified"
    forbidden_claims: list[Annotated[str, Field(max_length=300)]] = Field(default_factory=list, max_length=20)


class WatermarkDetect(BaseModel):
    """Auto-generate a per-frame watermark mask folder from the video.

    Avoids hand-drawing masks: ``color`` thresholds pixels near a target colour
    per frame (handles a moving watermark), ``temporal`` marks pixels that barely
    change across frames (a fixed semi-transparent overlay), and ``external`` runs
    a user-supplied detector command (e.g. Florence-2/SAM) via
    ``MRF_MASK_DETECTOR_CMD`` so heavy ML deps are never bundled.
    """

    method: Literal["color", "temporal", "external"] = "color"
    color: list[int] = Field(default_factory=lambda: [255, 255, 255])
    tolerance: int = Field(default=30, ge=0, le=255)
    dilation: int = Field(default=4, ge=0, le=64)
    threshold: int = Field(default=12, ge=0, le=255)
    fps: float = Field(default=0, ge=0, le=120)
    external_cmd: str = ""


class WatermarkRemoval(BaseModel):
    """Optional full-frame ("đánh chìm") watermark removal via ProPainter.

    Disabled by default so existing jobs are untouched. When enabled, the
    ``watermark`` pipeline stage reconstructs a clean source video before the
    scenes/render stages read it. ``mask`` may be a single PNG applied to every
    frame (white = remove), or a folder of per-frame masks for a watermark that
    moves over time; otherwise a rectangular mask is generated from ``top_band``
    / ``bottom_band`` (fractions of frame height) or ``boxes``.
    """

    enabled: bool = False
    method: WatermarkMethod = "propainter"
    mask: Path | None = None
    detect: WatermarkDetect | None = None
    top_band: float = Field(default=0, ge=0, le=0.5)
    bottom_band: float = Field(default=0, ge=0, le=0.5)
    boxes: list[list[float]] = Field(default_factory=list, max_length=12)


class JobConfig(BaseModel):
    job_id: str
    language: str = "vi"
    target_minutes: float = Field(default=10, ge=1, le=60)
    aspect_ratio: Literal["16:9", "9:16"] = "16:9"
    source_video: Path | None = None
    movie_title: str | None = None
    creative_brief: CreativeBrief = Field(default_factory=CreativeBrief)
    content_agent: ContentAgentMode = "scaffold"
    brand_top_band: float = Field(default=0, ge=0, le=0.2)
    brand_bottom_band: float = Field(default=0, ge=0, le=0.2)
    # Optional branded intro/outro cards on the main render (0 = disabled).
    intro_seconds: float = Field(default=0, ge=0, le=15)
    outro_seconds: float = Field(default=0, ge=0, le=15)
    # Optional full-frame watermark removal. Disabled by default.
    watermark_removal: WatermarkRemoval = Field(default_factory=WatermarkRemoval)
    # Optional Content ID bypass profile: 'off', 'light', 'balanced', 'aggressive'.
    copyright_bypass: str = Field(default="off", max_length=30)
    # Optional TTS provider: 'edge' (default, free/offline), 'fptai', or 'elevenlabs'.
    # Provider API keys are read from env only (MRF_FPTAI_API_KEY /
    # MRF_ELEVENLABS_API_KEY) and are never persisted in the manifest.
    tts_provider: str = Field(default="edge", max_length=30)
    # Optional voice id/name for the fptai / elevenlabs providers (e.g. FPT.AI
    # 'banmai' or an ElevenLabs voice id). Empty falls back to MRF_TTS_VOICE then a
    # per-provider default. Ignored by the edge provider. Not a secret.
    tts_voice: str = Field(default="", max_length=120)


class Artifact(BaseModel):
    name: str
    path: Path
    status: Literal["pending", "ready", "failed"] = "pending"


class StageResult(BaseModel):
    stage: str
    status: StageStatus = "pending"
    artifacts: list[Artifact] = []
    message: str = ""
    updated_at: datetime | None = None

    def mark(
        self,
        status: StageStatus,
        message: str = "",
        artifacts: list[Artifact] | None = None,
    ) -> "StageResult":
        """Update this stage in place and stamp the time. Returns self for chaining."""
        self.status = status
        if message:
            self.message = message
        if artifacts is not None:
            self.artifacts = artifacts
        self.updated_at = datetime.now(timezone.utc)
        return self


class JobManifest(BaseModel):
    config: JobConfig
    stages: list[StageResult] = []

    def stage(self, name: str) -> StageResult | None:
        return next((s for s in self.stages if s.stage == name), None)

    def pending_stages(self) -> list[StageResult]:
        """Stages still needing work (not yet ready/skipped)."""
        return [s for s in self.stages if s.status in ("pending", "running", "failed")]

    def next_pending(self) -> StageResult | None:
        return next(iter(self.pending_stages()), None)

    @property
    def is_complete(self) -> bool:
        return bool(self.stages) and all(
            s.status in ("ready", "skipped") for s in self.stages
        )


class MediaAsset(BaseModel):
    id: int | None = Field(default=None, ge=1)
    path: Path
    duration_seconds: float = Field(gt=0)


class Shot(BaseModel):
    id: int | None = Field(default=None, ge=1)
    media_asset_id: int = Field(ge=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    label: str | None = None

    @model_validator(mode="after")
    def validate_time_range(self) -> "Shot":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class TranscriptWord(BaseModel):
    """One spoken word with its measured start/end time (from Whisper word timings)."""
    word: str = ""
    start: float = Field(default=0.0, ge=0)
    end: float = Field(default=0.0, ge=0)


class TranscriptSegment(BaseModel):
    id: int | None = Field(default=None, ge=1)
    media_asset_id: int = Field(ge=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str = Field(min_length=1)
    speaker: str | None = None
    # Per-word timings for this segment; empty for legacy rows indexed before word
    # timings were persisted. Used to time captions on the real spoken words.
    words: list[TranscriptWord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_time_range(self) -> "TranscriptSegment":
        if self.end_seconds <= self.start_seconds:
            raise ValueError("end_seconds must be greater than start_seconds")
        return self


class VisualObservation(BaseModel):
    shot_id: int = Field(ge=1)
    description: str = Field(min_length=1, max_length=240)
    tags: list[str] = Field(default_factory=list)
    people: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    source: str = Field(default="agy", min_length=1, max_length=40)


class SceneSelection(BaseModel):
    id: int | None = Field(default=None, ge=1)
    media_asset_id: int = Field(ge=1)
    shot_id: int = Field(ge=1)
    transcript_segment_id: int | None = Field(default=None, ge=1)
    position: int = Field(ge=0)
    rationale: str | None = None
