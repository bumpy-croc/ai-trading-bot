"""EntryPauseGate: the single authority for "are entries blocked?" (#1096)."""

from __future__ import annotations

import pytest

from src.engines.live.execution.entry_pause import EntryPauseGate
from src.engines.live.system_halt import SystemHaltState

pytestmark = [pytest.mark.unit, pytest.mark.fast]


@pytest.fixture
def pause_flag(monkeypatch):
    """Control the FEATURE_ENTRY_PAUSE flag the gate reads."""

    def _set(enabled: bool) -> None:
        monkeypatch.setattr(
            "src.engines.live.execution.entry_pause.is_enabled",
            lambda name, default=False: enabled and name == "entry_pause",
        )

    _set(False)
    return _set


def _established(active: bool = False) -> SystemHaltState:
    return SystemHaltState(active=active, reason="board decision", established=True)


def test_no_levers_means_entries_allowed(pause_flag):
    gate = EntryPauseGate(_established())
    assert gate.entry_blocks() == ()
    assert gate.entry_block() is None
    assert gate.entry_block_reason() is None
    assert gate.paused("an entry") is False


def test_unverified_halt_state_blocks(pause_flag):
    gate = EntryPauseGate(SystemHaltState())  # never polled successfully
    blocks = gate.entry_blocks()
    assert [b.key for b in blocks] == ["system_halt"]
    assert "UNVERIFIED" in blocks[0].reason
    assert gate.paused("an entry") is True


def test_both_levers_are_reported_but_one_wins_the_log(pause_flag):
    """A shadowed lever must still be visible, or a monitor records it cleared."""
    pause_flag(True)
    gate = EntryPauseGate(_established(active=True))
    assert [b.key for b in gate.entry_blocks()] == ["system_halt", "entry_pause"]
    block = gate.entry_block()
    assert block is not None and block.key == "system_halt"
    assert gate.entry_block_reason() == block.reason


def test_entry_pause_alone_blocks(pause_flag):
    pause_flag(True)
    gate = EntryPauseGate(_established())
    assert [b.key for b in gate.entry_blocks()] == ["entry_pause"]
    assert gate.entry_block_reason() == "FEATURE_ENTRY_PAUSE active"


def test_gate_without_halt_state_reads_only_the_flag(pause_flag):
    pause_flag(True)
    gate = EntryPauseGate(None)
    assert [b.key for b in gate.entry_blocks()] == ["entry_pause"]
