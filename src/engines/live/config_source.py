"""Structural type for key-value app-config readers.

``ConfigSource`` is the duck type ``ConfigManager`` satisfies; settings
resolution depends on this protocol rather than the concrete config class so
tests can substitute plain mappings or specced mocks.
"""

from __future__ import annotations

from typing import Protocol


class ConfigSource(Protocol):
    """Key-value app-config reader (structural match for ``ConfigManager``)."""

    def get(self, key: str, default: str | None = None) -> str | None: ...
