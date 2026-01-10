"""Shared strategy version metadata structures."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass
class StrategyVersionRecord:
    """Metadata describing a concrete strategy version.

    The record is intentionally generic so it can be consumed by both the
    component-oriented strategy manager (which needs rich component metadata)
    and the live trading strategy manager (which focuses on deployment state).
    """

    version_id: str
    created_at: datetime
    name: str | None = None
    description: str | None = None
    strategy_name: str | None = None
    version: str | None = None
    config: dict[str, Any] | None = None
    components: dict[str, dict[str, Any]] | None = None
    parameters: dict[str, Any] | None = None
    model_path: str | None = None
    performance_metrics: dict[str, Any] | None = None
    is_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise the record to a plain dictionary."""

        data = asdict(self)
        data["created_at"] = self.created_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyVersionRecord:
        """Reconstruct a record from ``to_dict`` output."""

        payload = dict(data)
        created_at = payload.get("created_at")
        if isinstance(created_at, str):
            payload["created_at"] = datetime.fromisoformat(created_at)
        elif created_at is None:
            payload["created_at"] = datetime.now(UTC)
        return cls(**payload)


__all__ = ["StrategyVersionRecord"]
