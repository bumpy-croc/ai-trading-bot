"""Challenger-vs-incumbent promotion gate for retrained models.

A challenger is compared with the incumbent on three legs: holdout test RMSE,
out-of-sample profit factor, and out-of-sample return. A leg that cannot tell
the two models apart scores NO_RESULT instead of a win, so a tie on a degenerate
metric never buys a gate point. When the remaining legs cannot reach the
required number of wins the verdict is INCONCLUSIVE and the incumbent stays.

Usage from the weekly retrain: ``python -m src.ml.validation.promotion_gate
comparison.json`` where the file holds the two ``ModelEvidence`` records.

Caveat carried in every result: while HyperGrowth sizes flat (#938), the two
backtest legs mostly measure the strategy rather than the model, so RMSE is
the only leg with real discriminating power.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from src.performance.metrics import MAX_FINITE_RATIO

MIN_LOSING_TRADES_FOR_PF = 3
MIN_MATERIAL_RETURN_DIFF_PCT = 0.5
MIN_MATERIAL_RETURN_DIFF_USD = 1.0
REQUIRED_WINS = 2

BACKTEST_SENSITIVITY_CAVEAT = (
    "Backtest legs (profit factor, return) mostly measure the strategy while HyperGrowth "
    "sizes flat (#938); test RMSE is the only leg with real discriminating power."
)


class LegOutcome(str, Enum):
    WIN = "win"
    LOSS = "loss"
    NO_RESULT = "no_result"


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class ModelEvidence:
    """Measured results for one model.

    ``losing_trades`` is optional: when absent, a profit factor at the
    ``MAX_FINITE_RATIO`` sentinel is read as "no losing trades".
    """

    test_rmse: float
    profit_factor: float
    return_pct: float
    losing_trades: int | None = None


@dataclass(frozen=True)
class LegResult:
    name: str
    outcome: LegOutcome
    reason: str


@dataclass(frozen=True)
class GateResult:
    verdict: Verdict
    wins: int
    legs: list[LegResult] = field(default_factory=list)
    caveat: str = BACKTEST_SENSITIVITY_CAVEAT

    @property
    def promote(self) -> bool:
        return self.verdict is Verdict.PASS

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "promote": self.promote,
            "wins": self.wins,
            "required_wins": REQUIRED_WINS,
            "legs": [
                {"name": leg.name, "outcome": leg.outcome.value, "reason": leg.reason}
                for leg in self.legs
            ],
            "caveat": self.caveat,
        }


def _has_no_losers(evidence: ModelEvidence, min_losing_trades: int) -> bool:
    if evidence.losing_trades is not None:
        return evidence.losing_trades < min_losing_trades
    return evidence.profit_factor >= MAX_FINITE_RATIO


def _rmse_leg(challenger: ModelEvidence, incumbent: ModelEvidence) -> LegResult:
    outcome = LegOutcome.WIN if challenger.test_rmse <= incumbent.test_rmse else LegOutcome.LOSS
    return LegResult(
        "test_rmse",
        outcome,
        f"challenger {challenger.test_rmse:.6g} vs incumbent {incumbent.test_rmse:.6g} (lower wins)",
    )


def _profit_factor_leg(
    challenger: ModelEvidence, incumbent: ModelEvidence, min_losing_trades: int
) -> LegResult:
    if _has_no_losers(challenger, min_losing_trades) or _has_no_losers(
        incumbent, min_losing_trades
    ):
        return LegResult(
            "profit_factor",
            LegOutcome.NO_RESULT,
            f"fewer than {min_losing_trades} losing trades on at least one side; "
            "profit factor is a sentinel or too noisy to compare",
        )
    outcome = (
        LegOutcome.WIN if challenger.profit_factor >= incumbent.profit_factor else LegOutcome.LOSS
    )
    return LegResult(
        "profit_factor",
        outcome,
        f"challenger {challenger.profit_factor:.4g} vs incumbent {incumbent.profit_factor:.4g}",
    )


def _return_leg(
    challenger: ModelEvidence,
    incumbent: ModelEvidence,
    initial_balance: float,
    min_diff_pct: float,
    min_diff_usd: float,
) -> LegResult:
    diff_pct = challenger.return_pct - incumbent.return_pct
    diff_usd = diff_pct / 100.0 * initial_balance
    if abs(diff_pct) < min_diff_pct or abs(diff_usd) < min_diff_usd:
        return LegResult(
            "return",
            LegOutcome.NO_RESULT,
            f"difference {diff_pct:+.4f}pp (${diff_usd:+.4f}) below materiality "
            f"({min_diff_pct}pp and ${min_diff_usd})",
        )
    outcome = LegOutcome.WIN if diff_pct > 0 else LegOutcome.LOSS
    return LegResult("return", outcome, f"difference {diff_pct:+.4f}pp (${diff_usd:+.4f})")


def evaluate_promotion_gate(
    challenger: ModelEvidence,
    incumbent: ModelEvidence,
    *,
    initial_balance: float,
    min_losing_trades: int = MIN_LOSING_TRADES_FOR_PF,
    min_return_diff_pct: float = MIN_MATERIAL_RETURN_DIFF_PCT,
    min_return_diff_usd: float = MIN_MATERIAL_RETURN_DIFF_USD,
) -> GateResult:
    """Score the challenger against the incumbent.

    PASS needs ``REQUIRED_WINS`` wins. Without them, INCONCLUSIVE when the
    NO_RESULT legs could still have supplied the missing wins, otherwise FAIL.
    Only PASS promotes; INCONCLUSIVE retains the incumbent.
    """
    legs = [
        _rmse_leg(challenger, incumbent),
        _profit_factor_leg(challenger, incumbent, min_losing_trades),
        _return_leg(
            challenger, incumbent, initial_balance, min_return_diff_pct, min_return_diff_usd
        ),
    ]
    wins = sum(leg.outcome is LegOutcome.WIN for leg in legs)
    undecided = sum(leg.outcome is LegOutcome.NO_RESULT for leg in legs)
    if wins >= REQUIRED_WINS:
        verdict = Verdict.PASS
    elif wins + undecided >= REQUIRED_WINS:
        verdict = Verdict.INCONCLUSIVE
    else:
        verdict = Verdict.FAIL
    return GateResult(verdict=verdict, wins=wins, legs=legs)


def main(argv: list[str] | None = None) -> int:
    """Read ``{"challenger": {...}, "incumbent": {...}, "initial_balance": N}`` and print the result."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("usage: python -m src.ml.validation.promotion_gate comparison.json", file=sys.stderr)
        return 2
    payload = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    result = evaluate_promotion_gate(
        ModelEvidence(**payload["challenger"]),
        ModelEvidence(**payload["incumbent"]),
        initial_balance=float(payload["initial_balance"]),
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.promote else 1


__all__ = [
    "BACKTEST_SENSITIVITY_CAVEAT",
    "GateResult",
    "LegOutcome",
    "LegResult",
    "ModelEvidence",
    "Verdict",
    "evaluate_promotion_gate",
]

if __name__ == "__main__":
    sys.exit(main())
