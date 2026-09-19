from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

StageStatus = Literal["pending", "running", "ready", "failed", "skipped"]


class JobConfig(BaseModel):
    job_id: str
    language: str = "vi"
    target_minutes: float = Field(default=10, ge=1, le=60)
    aspect_ratio: Literal["16:9", "9:16"] = "16:9"
    source_video: Path | None = None


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
