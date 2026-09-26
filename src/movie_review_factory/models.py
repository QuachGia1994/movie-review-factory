from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, Field, model_validator

StageStatus = Literal["pending", "running", "ready", "failed", "skipped", "cancelled"]
ContentAgentMode = Literal["scaffold", "claude", "agy"]
CONTENT_AGENT_MODES = get_args(ContentAgentMode)


class JobConfig(BaseModel):
    job_id: str
    language: str = "vi"
    target_minutes: float = Field(default=10, ge=1, le=60)
    aspect_ratio: Literal["16:9", "9:16"] = "16:9"
    source_video: Path | None = None
    movie_title: str | None = None
    content_agent: ContentAgentMode = "scaffold"


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


class TranscriptSegment(BaseModel):
    id: int | None = Field(default=None, ge=1)
    media_asset_id: int = Field(ge=1)
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    text: str = Field(min_length=1)
    speaker: str | None = None

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
