from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal


@dataclass
class ParameterSet:
    """A concrete set of tunable parameters.
    Values are kept generic to support strategies, risk, and engine toggles.
    """

    name: str
    values: dict[str, Any]


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment/backtest run."""

    strategy_name: str
    symbol: str
    timeframe: str
    start: datetime
    end: datetime
    initial_balance: float
    risk_parameters: dict[str, Any] = field(default_factory=dict)
    feature_flags: dict[str, Any] = field(default_factory=dict)
    parameters: ParameterSet | None = None
    use_cache: bool = True
    provider: str = "binance"
    random_seed: int | None = None
    # "enforce" (default; safe for any promotion decision) halts the
    # backtest the instant drawdown crosses risk_parameters["max_drawdown"]
    # (or the ratified default if unset), matching live behaviour. "measure"
    # runs the full window without truncating so a research/characterisation
    # preregistration can observe a strategy's true drawdown profile; set it
    # explicitly and state so in the preregistration. Never leave the default
    # to silently truncate a study you meant to characterise -- see #1102.
    drawdown_cap_mode: Literal["enforce", "measure"] = "enforce"
    # Keyword arguments forwarded to the strategy factory at construction time
    # by :meth:`ExperimentRunner._load_strategy`. Use this for knobs the
    # strategy only honors in ``__init__`` (e.g. ``model_type`` on
    # hyper_growth, which swaps the underlying signal generator and cannot be
    # changed by setattr afterwards).
    factory_kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Results of a backtest/experiment with key KPIs."""

    config: ExperimentConfig
    total_trades: int
    win_rate: float
    total_return: float
    annualized_return: float
    max_drawdown: float
    sharpe_ratio: float
    final_balance: float
    session_id: int | None = None
    artifacts_path: str | None = None
    # Per-trade P&L sequence (as fractional returns in the order trades
    # closed). The reporter uses this to distinguish "different trades, same
    # aggregate" from "literally the same trades" when variants tie the
    # baseline on the headline metrics — a critical tie-breaker for
    # diagnosing dead-code overrides.
    trade_pnl_pcts: list[float] = field(default_factory=list)
    # Resolved sizing limits the run actually enforced (from the backtester's
    # results payload), so a clamped/defaulted max_position_size is visible
    # in every experiment artifact instead of silently skewing a study.
    effective_sizing: dict[str, float] = field(default_factory=dict)
    # Explicit, machine-readable truncation marker (#1102). True means every
    # metric on this result covers only the candles before the drawdown-cap
    # stop -- never compare a truncated result against an untruncated one
    # without accounting for this. Mirrors Backtester.early_stopped.
    early_stopped: bool = False
    # Which mode produced this result -- "enforce" (live-representative,
    # truncates) or "measure" (research, full window). Always populated so a
    # result can never be silently ambiguous about which behaviour it used.
    drawdown_cap_mode: Literal["enforce", "measure"] = "enforce"
    # Whether drawdown crossed the cap at any point in the run, independent
    # of mode. Under "measure", this is the answer a characterisation study
    # needs: the full profile was measured and it would have breached the
    # live cap at drawdown_cap_breach_date.
    drawdown_cap_breached: bool = False
    drawdown_cap_breach_date: datetime | None = None
    drawdown_cap_threshold: float | None = None
