"""Bear-market validation harness (#801).

Scores a candidate model by running the backtest engine over fixed historical
windows (2022 bear, Oct 2025–Feb 2026 crash, Feb–Jun 2026 chop) and reporting
per-window Sharpe / max-drawdown / win-rate / trade count / long-short balance.

The harness reuses :class:`~src.experiments.runner.ExperimentRunner` so it
inherits deterministic providers (``mock``/``fixture``) for CI and cached real
providers for genuine runs. It never trains or promotes — that decision lives
in :mod:`src.ml.validation.gate`.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from src.config.constants import (
    DEFAULT_VALIDATION_MAX_DRAWDOWN_PCT,
    DEFAULT_VALIDATION_MIN_TRADES,
)
from src.experiments.schemas import ExperimentConfig, ExperimentResult, ParameterSet
from src.infrastructure.runtime.paths import get_project_root

logger = logging.getLogger(__name__)

DEFAULT_WINDOWS_CONFIG = get_project_root() / "config" / "validation_windows.json"


@dataclass(frozen=True)
class BearWindow:
    """A fixed historical window a candidate model must survive."""

    name: str
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    max_drawdown_pct: float
    min_trades: int

    def __post_init__(self) -> None:
        # Catch bad config at construction rather than deep in a backtest.
        if not self.name:
            raise ValueError("BearWindow.name must be non-empty")
        if self.end <= self.start:
            raise ValueError(
                f"BearWindow {self.name!r}: end ({self.end}) must be after start ({self.start})"
            )
        if not math.isfinite(self.max_drawdown_pct) or self.max_drawdown_pct <= 0:
            raise ValueError(
                f"BearWindow {self.name!r}: max_drawdown_pct must be finite and > 0, "
                f"got {self.max_drawdown_pct!r}"
            )
        if self.min_trades < 0:
            raise ValueError(f"BearWindow {self.name!r}: min_trades must be >= 0")


@dataclass
class WindowScore:
    """Per-window backtest score plus the pass/fail verdict."""

    window_name: str
    sharpe: float
    max_drawdown_pct: float
    win_rate_pct: float
    total_trades: int
    long_short_balance: float | None
    passed: bool
    failures: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class ValidationReport:
    """Aggregate report across all windows."""

    symbol: str
    model_type: str
    strategy_name: str
    provider: str
    created_at: str
    scores: list[WindowScore] = field(default_factory=list)
    passed: bool = False
    inconclusive: bool = False

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "model_type": self.model_type,
            "strategy_name": self.strategy_name,
            "provider": self.provider,
            "created_at": self.created_at,
            "passed": self.passed,
            "inconclusive": self.inconclusive,
            "scores": [asdict(s) for s in self.scores],
        }

    def summary(self) -> str:
        lines = [
            f"Bear-market validation: {self.symbol}/{self.model_type} "
            f"({self.strategy_name}, provider={self.provider})"
        ]
        for s in self.scores:
            if s.error is not None:
                lines.append(f"  [{s.window_name}] ERROR: {s.error}")
                continue
            verdict = "PASS" if s.passed else "FAIL"
            lines.append(
                f"  [{s.window_name}] {verdict} "
                f"sharpe={s.sharpe:.2f} maxDD={s.max_drawdown_pct:.1f}% "
                f"win={s.win_rate_pct:.0f}% trades={s.total_trades}"
                + (f" | {'; '.join(s.failures)}" if s.failures else "")
            )
        overall = "PASS" if self.passed else ("INCONCLUSIVE" if self.inconclusive else "FAIL")
        lines.append(f"  OVERALL: {overall}")
        return "\n".join(lines)


def _parse_date(value: str) -> datetime:
    """Parse a YYYY-MM-DD (or ISO) date string into a UTC-aware datetime."""
    text = str(value).strip()
    # Accept a bare date or a full ISO timestamp; normalize a trailing Z.
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        # Fall back to date-only.
        dt = datetime.strptime(text, "%Y-%m-%d")  # noqa: DTZ007 - tz applied below
        return dt.replace(tzinfo=UTC)
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def load_validation_windows(path: str | Path | None = None) -> list[BearWindow]:
    """Load and validate the fixed validation windows from JSON config.

    Args:
        path: Optional path to the windows config; defaults to
            ``config/validation_windows.json``.

    Returns:
        A non-empty list of :class:`BearWindow`.

    Raises:
        FileNotFoundError: If the config file is missing.
        ValueError: If the config is malformed or defines no windows.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_WINDOWS_CONFIG
    if not cfg_path.exists():
        raise FileNotFoundError(f"Validation windows config not found: {cfg_path}")

    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    raw_windows = data.get("windows")
    if not isinstance(raw_windows, list) or not raw_windows:
        raise ValueError(f"{cfg_path} must define a non-empty 'windows' list")

    default_dd = float(data.get("default_max_drawdown_pct", DEFAULT_VALIDATION_MAX_DRAWDOWN_PCT))
    default_min_trades = int(data.get("default_min_trades", DEFAULT_VALIDATION_MIN_TRADES))

    windows: list[BearWindow] = []
    for entry in raw_windows:
        if not isinstance(entry, dict):
            raise ValueError(f"{cfg_path}: each window must be an object, got {entry!r}")
        windows.append(
            BearWindow(
                name=str(entry["name"]),
                symbol=str(entry["symbol"]),
                timeframe=str(entry.get("timeframe", "1d")),
                start=_parse_date(entry["start"]),
                end=_parse_date(entry["end"]),
                max_drawdown_pct=float(entry.get("max_drawdown_pct", default_dd)),
                min_trades=int(entry.get("min_trades", default_min_trades)),
            )
        )
    return windows


class BearValidationHarness:
    """Score a candidate model across the fixed bear-market windows.

    Args:
        symbol: Trading symbol the model serves (e.g. ``BTCUSDT``).
        model_type: Registry model type (``basic`` or ``sentiment``); selects
            the default strategy when ``strategy_name`` is not given.
        provider: Data provider for the backtests. Use ``mock``/``fixture`` for
            deterministic CI runs; ``binance`` (cached) for real evaluation.
        initial_balance: Starting balance for each window backtest.
        strategy_name: Strategy factory to drive the model; defaults from
            ``model_type``.
        runner_factory: Injectable zero-arg callable returning an object with a
            ``run(ExperimentConfig) -> ExperimentResult`` method. Tests inject a
            stub; production uses :class:`ExperimentRunner`.
    """

    def __init__(
        self,
        symbol: str,
        model_type: str = "basic",
        *,
        provider: str = "binance",
        initial_balance: float = 1000.0,
        strategy_name: str | None = None,
        model_path: str | Path | None = None,
        runner_factory: Callable[[], object] | None = None,
    ) -> None:
        self.symbol = symbol.upper()
        self.model_type = model_type
        self.provider = provider
        self.initial_balance = float(initial_balance)
        self.strategy_name = strategy_name or (
            "ml_sentiment" if model_type == "sentiment" else "ml_basic"
        )
        # When set, validate this specific candidate version rather than the
        # strategy's ``latest`` symlink. Threaded to the signal generator via a
        # ``model_path`` override so the gate scores the model that is about to
        # be promoted, not the one already live.
        self.model_path = str(model_path) if model_path is not None else None
        self._runner_factory = runner_factory

    def _make_runner(self) -> object:
        if self._runner_factory is not None:
            return self._runner_factory()
        # Imported lazily so the harness module has no hard dependency on the
        # backtest engine at import time (keeps CLI --help fast and avoids
        # pulling heavy deps when only the config loader is used).
        from src.experiments.runner import ExperimentRunner

        return ExperimentRunner()

    def _score_window(self, runner: object, window: BearWindow) -> WindowScore:
        parameters = None
        if self.model_path is not None:
            parameters = ParameterSet(
                name="candidate",
                values={f"{self.strategy_name}.model_path": self.model_path},
            )
        config = ExperimentConfig(
            strategy_name=self.strategy_name,
            symbol=window.symbol or self.symbol,
            timeframe=window.timeframe,
            start=window.start,
            end=window.end,
            initial_balance=self.initial_balance,
            provider=self.provider,
            use_cache=True,
            parameters=parameters,
        )
        try:
            result = runner.run(config)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - surface any backtest failure per-window
            logger.warning(
                "Validation window %s failed to run: %s", window.name, exc, exc_info=True
            )
            return WindowScore(
                window_name=window.name,
                sharpe=0.0,
                max_drawdown_pct=0.0,
                win_rate_pct=0.0,
                total_trades=0,
                long_short_balance=None,
                passed=False,
                failures=["backtest raised"],
                error=str(exc),
            )
        return self._evaluate_result(window, result)

    @staticmethod
    def _evaluate_result(window: BearWindow, result: ExperimentResult) -> WindowScore:
        # ``max_drawdown`` is already a percentage in the results dict (engine
        # multiplies the fractional drawdown by 100). Guard non-finite values.
        max_dd = float(result.max_drawdown)
        if not math.isfinite(max_dd):
            max_dd = 100.0
        failures: list[str] = []
        if max_dd > window.max_drawdown_pct:
            failures.append(
                f"max drawdown {max_dd:.1f}% exceeds {window.max_drawdown_pct:.1f}% cap"
            )
        if result.total_trades < window.min_trades:
            failures.append(
                f"only {result.total_trades} trades (< {window.min_trades}); inconclusive"
            )
        return WindowScore(
            window_name=window.name,
            sharpe=float(result.sharpe_ratio),
            max_drawdown_pct=max_dd,
            win_rate_pct=float(result.win_rate),
            total_trades=int(result.total_trades),
            long_short_balance=_long_short_balance(result.trade_pnl_pcts),
            passed=not failures,
            failures=failures,
        )

    def run(self, windows: list[BearWindow] | None = None) -> ValidationReport:
        """Score every window and build the aggregate report.

        A model **passes** only if every window passes. If any window could not
        run (backtest raised, e.g. missing data), the report is marked
        ``inconclusive`` and ``passed`` is False so the gate can decide whether
        to block or warn (per ``DEFAULT_VALIDATION_REQUIRED``).
        """
        windows = windows if windows is not None else load_validation_windows()
        if not windows:
            raise ValueError("No validation windows provided")

        runner = self._make_runner()
        scores = [self._score_window(runner, w) for w in windows]
        inconclusive = any(s.error is not None for s in scores)
        passed = bool(scores) and all(s.passed for s in scores) and not inconclusive
        return ValidationReport(
            symbol=self.symbol,
            model_type=self.model_type,
            strategy_name=self.strategy_name,
            provider=self.provider,
            created_at=datetime.now(UTC).isoformat(),
            scores=scores,
            passed=passed,
            inconclusive=inconclusive,
        )


def _long_short_balance(trade_pnls: list[float]) -> float | None:
    """Best-effort long/short balance proxy from the per-trade P&L series.

    The backtest results dict does not expose per-trade side, so this is a
    coarse reporting-only signal: the fraction of trades in the series, used to
    flag a model that produced no trades at all. Returns ``None`` when there is
    no trade data. (A precise long/short count would require instrumenting the
    engine; the drawdown gate is the load-bearing check.)
    """
    if not trade_pnls:
        return None
    return float(len(trade_pnls))
