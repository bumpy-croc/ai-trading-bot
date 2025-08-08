### Prediction Engine ↔ MlBasic Integration: Status and Next Steps

#### What was integrated
- Feature pipeline in `MlBasic`:
  - Replaced inline normalization with `FeaturePipeline`.
  - When engine enabled, use `PriceOnlyFeatureExtractor` to guarantee 5-feature input in fixed order (close, volume, high, low, open), window=120, matching the model.
- Full engine adoption for predictions:
  - `MlBasic` now calls `engine.predict(...)` when `USE_PREDICTION_ENGINE=1`; legacy ONNX path remains as fallback.
  - Preserved strategy semantics: same denormalization of predicted normalized price using rolling min–max over the previous 120 bars.
- Safe fallback and health checks:
  - Single warning per run on engine init/predict failure, then fallback to legacy ONNX path.
  - Per-init engine `health_check()` with a single warning when degraded.
- Logging and auditability:
  - Log engine metadata (`engine_enabled`, `engine_model_name`, `engine_batch`) in `ml_predictions` via `BaseStrategy.log_execution`.
  - Added per-row capture of `engine_direction` and `engine_confidence` (for monitoring only; not used for sizing/signals yet).
- Parity and smoke validation:
  - Parity test added: engine-off vs engine-on predictions over a 500-bar slice with relaxed relative error and direction-agreement thresholds.
  - Smoke test `test_ml_basic_backtest_2024_smoke` passes with `USE_PREDICTION_ENGINE=1` and `ENGINE_BATCH_INFERENCE=0` (batch off) to preserve baseline returns.
- Optional batching path:
  - `ENGINE_BATCH_INFERENCE` (default False). When enabled, uses the engine model’s ONNX session for batched inference over sliding windows for speed. Kept off by default to avoid small drift in returns until a performance budget test is in place.

Key files touched
- `strategies/ml_basic.py`: engine integration, health checks, fallback, logging.
- `prediction/features/price_only.py`: 5-feature price-only extractor.
- `prediction/engine.py`: added `predict_series(...)` batching API.
- `tests/test_smoke.py`: engine parity test.
- `docs/MODEL_TRAINING_AND_INTEGRATION_GUIDE.md`: training/integration guidance and checklist.

#### Current flags and defaults
- `USE_PREDICTION_ENGINE` (default False) – enables engine path.
- `PREDICTION_ENGINE_MODEL_NAME` (optional) – defaults to ONNX filename stem (e.g., `btcusdt_price`).
- `ENGINE_BATCH_INFERENCE` (default False) – enables batched ONNX inference via engine session.

#### Validations performed
- Engine ON (batch off): smoke returns preserved (≥ 73.81% for 2024 per `test_ml_basic_backtest_2024_smoke`).
- Engine parity short-slice test (direction and relative error thresholds) passes.

---

#### What remains and suggested optimizations
- Performance
  - Turn on batching with a performance budget test; verify returns stability. If needed, unify all paths to the batched path to eliminate per-step differences.
  - Consider engine-level caching across windows (share transforms and prepare batch inputs once).
- Output semantics alignment
  - Move denormalization into the engine (using metadata) so `PredictionResult.price` is on actual price scale; remove local denormalization from `MlBasic`.
  - Add a tolerance test asserting engine price scale equals local denorm within a tiny epsilon.
- Confidence/direction adoption
  - Introduce a feature flag to switch `MlBasic` sizing/signals to engine-provided `confidence`/`direction` once semantics are validated; maintain the legacy proxy as fallback.
  - Add A/B test comparing returns and drawdowns for proxy-based vs engine-based sizing.
- Feature/shape contract hardening
  - Add a unit test to validate extractor feature count and order against model metadata (`feature_count`). Fail fast on mismatch.
  - Add configuration profiles for extractors per strategy (price-only vs technical) and bind models via config.
- CI and safeguards
  - Add parity test to CI and keep smoke thresholds unchanged.
  - Add a performance test (max wall-time for the smoke run), with `ENGINE_BATCH_INFERENCE=1` to ensure speed-ups do not change returns.
  - Auto-disable engine on repeated health check failures (cooldown), with a single run-level warning.
- Model lifecycle
  - Extend metadata with: `sequence_length`, `feature_count`, normalization hints, training timeframe/instrument, model version.
  - Add a simple model registry report (listing models, metadata sanity, last loading status) and a CLI to set the active model via config.
- Monitoring
  - Enrich `ml_predictions` log with `inference_time` and `cache_hit` (from `PredictionResult`) to track inference overhead.

#### Step-by-step plan for the next PR
1) Add engine-level denormalization (respect metadata), remove local denormalization in `MlBasic` behind a flag; add tolerance tests.
2) Enable `ENGINE_BATCH_INFERENCE=1` in tests guarded by a performance budget; tweak chunk size if needed.
3) Add extractor↔model shape validation test and fail fast on mismatch.
4) Add a feature flag to use engine confidence/direction for sizing/signals; run A/B with the smoke test to compare returns.
5) Expand CI: run parity test, smoke test, and performance budget test with engine on (batch on) and ensure thresholds are met.

#### Quick commands (dev)
- Parity test: `pytest -q tests/test_smoke.py::test_ml_basic_engine_parity_short_slice -n 4`
- Smoke test (engine on, batch off): `USE_PREDICTION_ENGINE=1 ENGINE_BATCH_INFERENCE=0 pytest -q tests/test_smoke.py::test_ml_basic_backtest_2024_smoke -n 4`
- Smoke test (engine on, batch on): `USE_PREDICTION_ENGINE=1 ENGINE_BATCH_INFERENCE=1 pytest -q tests/test_smoke.py::test_ml_basic_backtest_2024_smoke -n 4`