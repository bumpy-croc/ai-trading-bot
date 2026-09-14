# GH #1145: price_normalization stamped on non-price targets

## Defect

`src/ml/model_metadata.py::uses_rolling_minmax_features()` decided whether a bundle needs
`price_normalization` metadata using only two signals:

1. `task_type == "regression"`
2. `f"{target}_normalized"` present in `feature_names`

Both signals are true for a model trained with `--force-price-only --target-type
smoothed_return`:

- `close_normalized` is fed as an **input** feature (passes check 2), because
  `--force-price-only` always builds the price-only rolling-minmax feature set regardless of
  target.
- Every TF architecture except `tft`/`tft_ternary` compiles a regression head, and
  `target_type=smoothed_return` also maps to `TaskType.REGRESSION`
  (`src/ml/training_pipeline/task_types.py::TARGET_TASK_TYPES`) — so `task_type` in the
  written metadata is `"regression"` (passes check 1).

`task_type` cannot distinguish "predicts a normalized price" from "predicts a small-scale
return" — both are `TaskType.REGRESSION`. The bundle got `price_normalization:
{method: rolling_minmax, ...}` written by `enrich_bundle_metadata()` even though its output is
a ~0.002-scale return, not a price. At inference, `PredictionEngine._apply_rolling_denormalization`
then computed `low + value * (high - low)` on that return — a number in price space, silently
wrong, no exception, no log line.

Same failure class as #1049 (missing stamp, loud: 0 trades) in the opposite direction (spurious
stamp, silent: garbage trading signals).

## Root-cause fix

`uses_rolling_minmax_features()` now also consults the recorded
`training_params.target_type` (the truth the training pipeline already writes at
`training_pipeline/pipeline.py:705`, one write site, never overridden downstream — the sibling
literal LESSONS §1.12 rule (b)/(c) call for). Logic:

```
task_type == "regression"                       (unchanged, first gate)
  AND close_normalized in feature_names          (unchanged, second gate)
  AND (target_type is None                       # legacy bundle, see below
       OR target_type.lower() == "regression")   # explicit price target
```

If `training_params.target_type` is present and is anything other than `"regression"` (e.g.
`"smoothed_return"`, `"binary_direction"`, `"triple_barrier"`, `"meta_label"`), the bundle is no
longer classified as a rolling-minmax price bundle: no `price_normalization` is synthesized, no
denormalization happens at inference, `missing_prediction_keys()` returns `[]`, and the bundle
loads/promotes cleanly (there's nothing to denormalize — the fix is not "reject the bundle",
it's "stop lying about what it predicts").

### Why an absent `target_type` falls back to the old feature-based inference, not a hard error

The issue's suggested fix asks to treat an absent `target_type` on a bundle with normalized
features as a hard error rather than an implicit yes, quoting LESSONS §1.12 rule (c) ("derive a
contract field from the recorded truth, never a proxy").

Implementing that literally breaks the **currently served** model:
`ETHUSDT/basic/2026-07-04_22h_v1` (verified in this repo — `src/ml/models/ETHUSDT/basic/2026-07-04_22h_v1/metadata.json`)
has **no** `task_type` key and **no** `training_params.target_type` key at all (it predates both
fields), yet it already carries a correct, explicitly-written `price_normalization` block. Two
call sites reach `uses_rolling_minmax_features()` unconditionally as part of every bundle load:

- `PredictionModelRegistry._load_bundle` → `validate_bundle_metadata()` (`src/prediction/models/registry.py:293`)
- `training_pipeline`/cloud sync → `enrich_bundle_metadata()`/`ensure_bundle_metadata_complete()`

A hard error inside the classifier function fires **before** either of those get to observe that
the required keys are already present, so it would raise on every registry load of this legacy
bundle — turning "the live model works" into "the live model refuses to load," entirely
independent of whether anything is actually wrong with it. That is a strictly worse regression
than the bug being fixed (LESSONS §1.12 rule (c): "a fix that removes a loud failure is a
regression unless it replaces it with a correct value on every path the loud failure used to
cover").

Every bundle produced by the current training pipeline (local and cloud — both funnel through
`training_pipeline/pipeline.py::run_training_pipeline`, the single writer since #1122/#1132)
always writes `training_params.target_type`, so the ambiguity this function exists to resolve
(a normalized price feature feeding a non-price target) is structurally impossible to hit on a
`target_type`-less bundle: no such bundle can ever be a `--force-price-only
--target-type smoothed_return` bundle, because `target_type` recording predates the flag's
production use. The `target_type is None` branch is therefore a legacy compatibility path, not
an "implicit yes" that could hide the reported bug — it preserves old behavior exactly where new
information cannot exist, and does nothing where it can.

Confirmed no regression:
```
uses_rolling_minmax_features(ETHUSDT/basic/2026-07-04_22h_v1 metadata) -> True
missing_prediction_keys(...) -> []
validate_bundle_metadata(...) -> no exception
```

## Reachability verification

- `cli/commands/train.py:68` — `--force-price-only` (`action="store_true"`).
- `cli/commands/train.py:118` — `--target-type`, `choices=[..., "smoothed_return", ...]`.
- Both flags flow into `TrainingConfig.target_type` / `force_price_only`, consumed identically
  by the local (`cli/commands/train_commands.py`) and cloud (`src/ml/cloud/entrypoint.py`)
  training paths, both of which call the same `run_training_pipeline()` writer.
- Not reachable today by any bundle in `src/ml/models/`: none of the on-disk bundles combine
  `feature_strategy=price_only_rolling_minmax`-style normalized-price features with
  `training_params.target_type == "smoothed_return"`. The deployed live bundle,
  `ETHUSDT/basic/2026-07-04_22h_v1`, predates `target_type` tracking entirely and is a genuine
  price target (verified above) — **not** an instance of this bug, just a bundle old enough
  that the new check has to explicitly not break it.
- Reachable by the next retrain that combines these flags, e.g. the TARGET-REDESIGN tournament's
  planned entrant (d) (`smoothed_return`), per the issue.

## Defense in depth

`PredictionEngine._apply_rolling_denormalization` (`src/prediction/engine.py`) now logs a
WARNING referencing GH #1145 when a rolling-minmax "price" prediction lands outside a generous
`[-0.5, 1.5]` band around the expected `[0, 1]` normalized-price range, before denormalizing it
as before. This is diagnostic only, not a hard gate: a small positive `smoothed_return` value
(e.g. `0.002`) looks exactly like a plausible normalized-price value and cannot be distinguished
by range alone, so this check cannot catch every misclassification — it exists to surface an
obviously-wrong case (e.g. a negative or >1.5 return) in production logs rather than let it pass
completely silently. The real fix is the metadata classifier change above; this is belt and
braces per the issue's ask to "consider" independent validation at the consumption site.

## Tests added

`tests/unit/test_model_metadata_contract.py`:
- `test_smoothed_return_target_is_not_flagged_despite_normalized_features` — the exact
  `--force-price-only --target-type smoothed_return` combination from the issue; asserts
  `uses_rolling_minmax_features` is `False`, no `price_normalization` is synthesized, and the
  bundle validates cleanly with none of the price-normalization keys present.
- `test_genuine_price_target_still_flagged_with_target_type_recorded` — an explicit
  `target_type="regression"` bundle is still correctly classified and enriched.
- `test_bundle_without_target_type_falls_back_to_feature_inference` — legacy bundles without
  `target_type` keep the old feature-based inference (regression-proofs the compatibility
  decision above).

`tests/unit/predictions/test_engine_prediction.py`:
- `test_out_of_range_prediction_logs_warning_but_still_denormalizes` — an out-of-band value
  still denormalizes (non-fatal) and logs a warning naming #1145.
- `test_in_range_prediction_does_not_warn` — a plausible value does not trip the new log.

## References

PR #1134 (introduced the regression this issue fixes), #1049 (the original missing-stamp bug),
#1122/#1132 (the writer/reader split this respects), LESSONS §1.12 (rule (c) in particular),
LESSONS §2.14 (silently-wrong-numbers defects are never low priority).
