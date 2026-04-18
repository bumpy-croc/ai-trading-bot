"""Signal-quality diagnostic — measure the raw predictive signal of a strategy.

The ExperimentRunner ranks variants by backtest P&L. That is the right metric
for picking a winner, but it has a critical blind spot: a degenerate signal
(e.g. constant SELL because the model was fed the wrong feature tensor) can
still produce non-zero P&L from stop-loss and trailing-stop mechanics, and
every variant on top of that signal looks bitwise-identical in the reporter.
See ``.claude/reports/hyper_growth_experiment_sweep_2026-04-17.md`` for the
incident that motivated this tool.

The diagnostic walks the strategy's ``SignalGenerator`` bar-by-bar over a
range of history and reports four distributions that, together, tell you
whether the model has any directional edge at all:

* **Decision mix** — counts of BUY / SELL / HOLD. If it's 100% any-one-side
  or 100% HOLD the signal is dead regardless of how good the P&L looks.
* **Predicted return** — ``(prediction - current_price) / current_price``
  from the generator's metadata. A healthy model has a real distribution;
  a broken pipeline returns constants like ``-1.0`` (prediction = 0.0).
* **Confidence** — the generator's per-bar confidence score. A useful
  signal varies between high- and low-conviction bars.
* **Direction-conditional hit rate** — ``P(forward return > 0 | BUY)`` and
  ``P(forward return < 0 | SELL)`` at 1h / 4h / 12h / 24h horizons. These
  are the quantities any confidence-weighted sizer is trying to exploit.

Run via ``atb experiment diagnose --strategy <name> ...`` or programmatically
through :class:`SignalDiagnostic`. This package is split across one file per
class per CODE.md; re-exports below preserve the historical flat import path
``from src.experiments.diagnostics import SignalDiagnostic, ...``.
"""

from __future__ import annotations

from src.experiments.diagnostics.hit_rate import HitRate
from src.experiments.diagnostics.report import DiagnosticReport
from src.experiments.diagnostics.runner import DEFAULT_HORIZONS, SignalDiagnostic
from src.experiments.diagnostics.stats import DistributionStats

__all__ = [
    "DEFAULT_HORIZONS",
    "DiagnosticReport",
    "DistributionStats",
    "HitRate",
    "SignalDiagnostic",
]
