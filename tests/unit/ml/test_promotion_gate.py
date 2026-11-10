"""Promotion gate: degenerate legs must not score free points (GH #1146)."""

import json
import math

import pytest

from src.ml.promotion_gate import (
    LegOutcome,
    ModelEvidence,
    Verdict,
    evaluate_promotion_gate,
    main,
)

SENTINEL = 999.0


def _leg(result, name):
    return next(leg for leg in result.legs if leg.name == name)


def test_sentinel_pf_tie_and_dust_return_do_not_pass():
    # 2026-09-06 week: PF 999 vs 999, +0.0052pp on an $85 book, RMSE lost.
    challenger = ModelEvidence(
        test_rmse=0.0104, profit_factor=SENTINEL, return_pct=0.0052, losing_trades=0
    )
    incumbent = ModelEvidence(
        test_rmse=0.0100, profit_factor=SENTINEL, return_pct=0.0, losing_trades=0
    )

    result = evaluate_promotion_gate(challenger, incumbent, initial_balance=85.0)

    assert _leg(result, "profit_factor").outcome is LegOutcome.NO_RESULT
    assert _leg(result, "return").outcome is LegOutcome.NO_RESULT
    assert _leg(result, "test_rmse").outcome is LegOutcome.LOSS
    assert result.verdict is Verdict.INCONCLUSIVE
    assert not result.promote


def test_real_wins_on_two_legs_pass():
    challenger = ModelEvidence(0.009, 1.8, 6.0, losing_trades=5)
    incumbent = ModelEvidence(0.010, 1.2, 1.0, losing_trades=6)

    result = evaluate_promotion_gate(challenger, incumbent, initial_balance=1000.0)

    assert result.verdict is Verdict.PASS
    assert result.promote


def test_two_real_losses_fail():
    challenger = ModelEvidence(0.012, 0.8, -4.0, losing_trades=8)
    incumbent = ModelEvidence(0.010, 1.5, 3.0, losing_trades=5)

    result = evaluate_promotion_gate(challenger, incumbent, initial_balance=1000.0)

    assert result.verdict is Verdict.FAIL


def test_few_losing_trades_blocks_pf_leg_even_with_finite_pf():
    challenger = ModelEvidence(0.009, 5.0, 6.0, losing_trades=1)
    incumbent = ModelEvidence(0.010, 1.2, 1.0, losing_trades=6)

    result = evaluate_promotion_gate(challenger, incumbent, initial_balance=1000.0)

    assert _leg(result, "profit_factor").outcome is LegOutcome.NO_RESULT
    assert result.wins == 2  # RMSE + return
    assert result.verdict is Verdict.PASS


def test_return_diff_needs_both_pct_and_dollar_materiality():
    challenger = ModelEvidence(0.009, 1.5, 1.6, losing_trades=5)
    incumbent = ModelEvidence(0.010, 1.5, 1.0, losing_trades=5)

    # 0.6pp clears the pct bar but is $0.51 on an $85 book.
    small_book = evaluate_promotion_gate(challenger, incumbent, initial_balance=85.0)
    big_book = evaluate_promotion_gate(challenger, incumbent, initial_balance=10_000.0)

    assert _leg(small_book, "return").outcome is LegOutcome.NO_RESULT
    assert _leg(big_book, "return").outcome is LegOutcome.WIN


def test_cli_exit_code_reflects_promotion(tmp_path, capsys):
    payload = {
        "challenger": {
            "test_rmse": 0.009,
            "profit_factor": 1.8,
            "return_pct": 6.0,
            "losing_trades": 5,
        },
        "incumbent": {
            "test_rmse": 0.010,
            "profit_factor": 1.2,
            "return_pct": 1.0,
            "losing_trades": 6,
        },
        "initial_balance": 1000,
    }
    path = tmp_path / "cmp.json"
    path.write_text(json.dumps(payload))

    assert main([str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "pass"


def test_identical_evidence_is_not_a_pass():
    same = ModelEvidence(0.01, 1.5, 3.0, losing_trades=5)

    result = evaluate_promotion_gate(same, same, initial_balance=1000.0)

    assert result.wins == 0
    assert not result.promote


def test_immaterial_rmse_edge_is_no_result():
    challenger = ModelEvidence(0.009999, 1.8, 6.0, losing_trades=5)
    incumbent = ModelEvidence(0.010000, 1.2, 1.0, losing_trades=6)

    result = evaluate_promotion_gate(challenger, incumbent, initial_balance=1000.0)

    assert _leg(result, "test_rmse").outcome is LegOutcome.NO_RESULT


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_evidence_rejects_non_finite(bad):
    with pytest.raises(ValueError):
        ModelEvidence(test_rmse=bad, profit_factor=1.0, return_pct=0.0, losing_trades=1)


@pytest.mark.parametrize("balance", [0.0, -5.0, math.nan])
def test_gate_rejects_bad_balance(balance):
    e = ModelEvidence(0.01, 1.5, 3.0, losing_trades=5)
    with pytest.raises(ValueError):
        evaluate_promotion_gate(e, e, initial_balance=balance)


def test_module_imports_without_validation_package():
    import subprocess
    import sys

    code = "import sys; import src.ml.promotion_gate; assert 'src.ml.validation' not in sys.modules"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_zero_profit_factor_incumbent_is_not_a_free_win():
    challenger = ModelEvidence(0.009, 1.5, 6.0, losing_trades=5)
    incumbent = ModelEvidence(0.010, 0.0, 1.0, losing_trades=5)

    result = evaluate_promotion_gate(challenger, incumbent, initial_balance=1000.0)

    assert _leg(result, "profit_factor").outcome is LegOutcome.NO_RESULT


def test_zero_rmse_incumbent_leg_has_no_result():
    challenger = ModelEvidence(0.5, 1.5, 6.0, losing_trades=5)
    incumbent = ModelEvidence(0.0, 1.2, 1.0, losing_trades=5)

    result = evaluate_promotion_gate(challenger, incumbent, initial_balance=1000.0)

    assert _leg(result, "test_rmse").outcome is LegOutcome.NO_RESULT
    assert "degenerate" in _leg(result, "test_rmse").reason


def test_losing_trades_is_required():
    with pytest.raises(TypeError):
        ModelEvidence(test_rmse=0.01, profit_factor=1.5, return_pct=1.0)
