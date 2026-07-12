"""Point-in-time model version resolution from bundle metadata timestamps.

The backtest harness resolves models via the ``latest`` symlink at
invocation time, so a backtest spanning a model-promotion boundary silently
scores history with a model that was never live for it (GH #988). This
module answers "which version WAS latest at date X?" so the harness can pin
the right one and warn when a window spans a promotion boundary.

Each bundle's ``metadata.json`` ``created_at`` is used as the moment the
version became ``latest``. That is an approximation: promotion (the symlink
flip) happens shortly after training, and a trained-but-never-promoted
version would still be counted. The authoritative promotion record is
``docs/research/model-promotions.md`` — cross-check it when a resolution is
load-bearing. Resolution here is deliberately deterministic and in-repo so
the backtest CLI needs no external state.

Consumed by the backtest CLI only — never on the live trading path, which
must always resolve ``latest`` (see ``PredictionModelRegistry.select_bundle``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .exceptions import ModelNotAvailableError

logger = logging.getLogger(__name__)

# Version directories follow {YYYY-MM-DD}_{...} (e.g. 2026-07-04_22h_v1);
# the date prefix dates bundles whose metadata lacks created_at.
_VERSION_ID_DATE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})(?:_|$)")


@dataclass(frozen=True)
class VersionRecord:
    """One model version and when it (approximately) became ``latest``.

    Attributes:
        version_id: Version directory name, e.g. ``2026-07-04_22h_v1``.
        effective_at: UTC timestamp the version is treated as live from.
        source: Where the timestamp came from — ``"metadata"``
            (metadata.json ``created_at``) or ``"version_id"`` (date prefix
            of the directory name).
    """

    version_id: str
    effective_at: datetime
    source: str


@dataclass(frozen=True)
class PromotionSegment:
    """A maximal sub-interval of a backtest window served by one version.

    ``version_id`` is None when no version existed yet for that part of the
    window (live would have run a cross-symbol substitute or nothing).
    """

    version_id: str | None
    start: datetime
    end: datetime


def _ensure_utc(value: datetime) -> datetime:
    """Interpret naive datetimes as UTC; convert aware ones to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _timestamp_from_metadata(bundle_dir: Path) -> datetime | None:
    """Parse metadata.json's created_at, or None when absent/unparseable."""
    metadata_path = bundle_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(metadata, dict):
        return None
    created_at = metadata.get("created_at")
    if not isinstance(created_at, str):
        return None
    try:
        return _ensure_utc(datetime.fromisoformat(created_at))
    except ValueError:
        return None


def _timestamp_from_version_id(version_id: str) -> datetime | None:
    """Parse the {YYYY-MM-DD} prefix of a version directory name."""
    match = _VERSION_ID_DATE_RE.match(version_id)
    if match is None:
        return None
    year, month, day = (int(part) for part in match.groups())
    try:
        return datetime(year, month, day, tzinfo=UTC)
    except ValueError:
        return None


def list_version_records(
    registry_path: str | Path, symbol: str, model_type: str
) -> list[VersionRecord]:
    """List every dateable version under ``{registry}/{symbol}/{model_type}``.

    Sorted by (effective_at, version_id) so resolution and boundary
    detection are deterministic. Versions that cannot be dated (no
    metadata ``created_at`` and no date-prefixed directory name) are
    skipped with a warning — they cannot participate in point-in-time
    resolution.
    """
    model_type_dir = Path(registry_path) / symbol / model_type
    if not model_type_dir.is_dir():
        return []

    records: list[VersionRecord] = []
    for version_dir in model_type_dir.iterdir():
        if not version_dir.is_dir() or version_dir.name == "latest":
            continue
        effective_at = _timestamp_from_metadata(version_dir)
        source = "metadata"
        if effective_at is None:
            effective_at = _timestamp_from_version_id(version_dir.name)
            source = "version_id"
        if effective_at is None:
            logger.warning(
                "Skipping undateable model version %s/%s/%s: no metadata created_at "
                "and no YYYY-MM-DD prefix in the directory name",
                symbol,
                model_type,
                version_dir.name,
            )
            continue
        records.append(
            VersionRecord(
                version_id=version_dir.name, effective_at=effective_at, source=source
            )
        )

    records.sort(key=lambda record: (record.effective_at, record.version_id))
    return records


def resolve_version_as_of(
    registry_path: str | Path, symbol: str, model_type: str, as_of: datetime
) -> VersionRecord:
    """Resolve which version was ``latest`` for symbol/model_type at ``as_of``.

    Returns the newest version with ``effective_at <= as_of`` (inclusive:
    a version is live from the instant it appears).

    Raises:
        ModelNotAvailableError: No version existed at ``as_of`` — either
            the registry has none at all for this symbol/model_type, or all
            of them postdate ``as_of`` (live would have been running a
            cross-symbol substitute; see docs/research/model-promotions.md).
    """
    as_of_utc = _ensure_utc(as_of)
    records = list_version_records(registry_path, symbol, model_type)
    if not records:
        raise ModelNotAvailableError(
            f"No {symbol}/{model_type} model versions found under {registry_path} — "
            f"cannot resolve which version was latest at {as_of_utc.isoformat()}."
        )

    active = [record for record in records if record.effective_at <= as_of_utc]
    if not active:
        earliest = records[0]
        raise ModelNotAvailableError(
            f"No {symbol}/{model_type} model version existed at {as_of_utc.isoformat()}: "
            f"the earliest is {earliest.version_id} "
            f"(effective {earliest.effective_at.isoformat()}). Live trading before that "
            f"date ran a cross-symbol substitute or no model at all — see "
            f"docs/research/model-promotions.md."
        )
    return active[-1]


def promotion_segments(
    records: list[VersionRecord], start: datetime, end: datetime
) -> list[PromotionSegment]:
    """Split a backtest window into sub-intervals served by one version each.

    A promotion strictly inside (start, end) starts a segment; one exactly
    at ``end`` affects zero bars and does not split. The first segment's
    ``version_id`` is None when the window starts before any version
    existed. More than one segment means the window spans a promotion
    boundary and a single pinned model cannot match live's entire history.
    """
    start_utc = _ensure_utc(start)
    end_utc = _ensure_utc(end)

    def _active_version_at(moment: datetime) -> str | None:
        active_id: str | None = None
        for record in records:
            if record.effective_at <= moment:
                active_id = record.version_id
        return active_id

    # Deduplicated: two versions sharing one effective_at must not create a
    # zero-length segment (the later-sorted version wins the whole segment).
    boundaries = sorted(
        {
            record.effective_at
            for record in records
            if start_utc < record.effective_at < end_utc
        }
    )

    segments: list[PromotionSegment] = []
    segment_start = start_utc
    for boundary in boundaries:
        segments.append(
            PromotionSegment(
                version_id=_active_version_at(segment_start),
                start=segment_start,
                end=boundary,
            )
        )
        segment_start = boundary
    segments.append(
        PromotionSegment(
            version_id=_active_version_at(segment_start), start=segment_start, end=end_utc
        )
    )
    return segments
