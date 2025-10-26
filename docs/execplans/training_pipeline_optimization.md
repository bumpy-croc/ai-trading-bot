# Training Pipeline Optimization (ONNX Retained)

## Purpose / Big Picture

Researchers use the `atb train` CLI command to produce updated price and sentiment models before validating them in backtests and live paper trading. After the October 2025 refactors that merged data downloads, feature engineering, training, diagnostics, and ONNX export into a single script, each run now takes many minutes even when the underlying data has not changed. This plan explains how to reorganize the training flow so that a person new to this repository can fetch data once, reuse cached inputs, adjust hyperparameters quickly, and still generate the ONNX bundles consumed by the live trading and backtesting engines. When the work is complete, the command `atb train model BTCUSDT --timeframe 1d --start-date 2019-01-01 --end-date 2024-12-01` will finish at least forty percent faster on the same hardware while reporting nearly identical RMSE values (difference less than or equal to one percent) compared to the current baseline. A novice will be able to observe the faster execution by running the same command before and after the change, checking the wall-clock durations printed in the console logs, and verifying that both the `.keras` and `.onnx` artifacts appear under `src/ml/` as they do today.

## Scope and Non-Goals

The scope is limited to the training pipeline implemented in `cli/commands/train_commands.py` and any helper modules created to support it. The ONNX conversion step must remain available because every strategy currently references `.onnx` files and the live inference layer uses `onnxruntime`. Engines under `src/live`, `src/backtesting`, and `src/prediction` are out of scope for code modifications. The plan does not remove TensorFlow; it restructures how the trainer feeds it. Any optional shortcuts introduced here must default to the current behavior so that unattended jobs keep producing identical artifacts.

## Context and Definitions

The training command performs five broad tasks today. First, it downloads OHLCV candles into `data/` by calling the CLI data downloader each time, regardless of whether the same file already exists. Second, it loads market-wide sentiment from the Alternative.me Fear & Greed API through `src/data_providers/feargreed_provider.py`, which fetches the entire historical dataset on every import. Third, it merges those sources, scales features with `MinMaxScaler`, and creates sliding windows in pure Python loops (`create_sequences`). Fourth, it instantiates a convolutional plus recurrent TensorFlow model and trains it for fixed defaults (300 epochs, 120-step windows, batch size 32). Finally, it runs multiple diagnostics (robustness tests, matplotlib plots, evaluation) and always converts the SavedModel into ONNX via `tf2onnx`. Because every step is inside one function, no part can be skipped or reused, making experimentation slow.

Caching means storing previously downloaded data locally so that subsequent runs can read it from disk without reissuing the same HTTP requests. A `tf.data.Dataset` is a TensorFlow abstraction that streams batches efficiently; using it avoids Python loops and automatically pipelines preprocessing with training. Mixed precision uses float16 math on GPUs to accelerate training while keeping float32 master weights; it must be enabled selectively because CPUs often lack the required support. These concepts are defined here so a new contributor understands each reference without consulting other documents.

## Progress

- [x] (2025-10-26 23:35Z) Captured initial scope, goals, and constraints in this ExecPlan.
- [ ] Record baseline timings for the current trainer by running `atb train model BTCUSDT --timeframe 1d --start-date 2019-01-01 --end-date 2024-12-01` and logging durations for ingestion, feature prep, dataset creation, training, evaluation, diagnostics, and ONNX export.
- [ ] Implement deterministic caching for price data and sentiment data plus a `--force-refresh` flag; add unit tests for the new logic.
- [ ] Refactor the monolithic trainer into modules (ingestion, feature engineering, dataset building, training runner, artifact writer) with docstrings and tests.
- [ ] Replace the Python sequence builder with a vectorized or `tf.data` based approach, ensuring shapes and order remain stable.
- [ ] Add CLI flags for epochs, batch size, sequence length, and toggles (`--skip-plots`, `--skip-robustness`, `--skip-onnx`) while keeping ONNX export on by default; mutate metadata to record which steps ran.
- [ ] Detect GPUs and optionally enable mixed precision and XLA, with documentation on how to opt out when debugging.
- [ ] Re-run the baseline command, capture new timings, confirm RMSE parity (≤1% delta), and update this plan’s Outcomes section before closing.

## Surprises & Discoveries

No surprises have been observed yet. Whenever you encounter an unexpected behavior—such as API rate limits, TensorFlow incompatibilities, or caching pitfalls—document it here with the date, the symptom, and how you handled it. This running log ensures future contributors inherit the full story.

## Decision Log

Decision: Maintain ONNX conversion support but allow researchers to skip it explicitly when they do not need an immediately deployable artifact.
Rationale: Live trading, backtesting, and prediction layers reference `.onnx` files and are tightly coupled to `onnxruntime`. Optional conversion shortens inner development loops without forcing other teams to change tooling.
Date/Author: 2025-10-26 / Codex agent.

Decision: Modularize the trainer into clearly named stages (ingestion, features, dataset, trainer, artifacts) hosted either under a new `src/ml/training_pipeline/` package or equivalent namespaced modules referenced from the CLI.
Rationale: Small modules are easier to profile, unit test, and reuse. A novice can open the dedicated file responsible for each task rather than navigating a single 700-line function.
Date/Author: 2025-10-26 / Codex agent.

## Plan of Work

Begin by capturing the baseline so that improvements are measurable. Instrument the existing code with simple timing utilities (for example, `time.perf_counter()` stored in local variables) around each major block, run the command listed in the Progress section, and record the durations in this document. This establishes the comparison point for the forty percent improvement target and reveals which stages cost the most.

Next, tackle ingestion caching. Before calling `cli.commands.data._download`, compute the expected filename pattern `{symbol}_{timeframe}_{start}_{end}` inside the `data/` directory. If the file exists (CSV or Feather), log that it will be reused and skip the download. Expose a new CLI flag `--force-refresh` (default false) that deletes or bypasses cached artifacts when the user needs fresh candles. For sentiment, reuse a single `FearGreedProvider` instance per run and persist its DataFrame into `cache/sentiment/feargreed.parquet` along with the last timestamp. On subsequent runs, read from that file when it covers the requested date range and remains within `freshness_days`. Write unit tests that simulate existing files and assert that downloads occur only when necessary or when `--force-refresh` is true.

With ingestion addressed, refactor the pipeline into discrete helpers. Create a `TrainingConfig` dataclass that captures symbol, timeframe, date range, epochs, batch size, sequence length, flags for diagnostics and exports, and whether mixed precision is allowed. Move ingestion logic to a module such as `src/ml/training_pipeline/ingestion.py`, feature engineering to `features.py`, dataset preparation to `datasets.py`, trainer orchestration to `runner.py`, and artifact writing (including ONNX conversion and metadata emission) to `artifacts.py`. Each module must expose functions that accept explicit inputs and return well-defined outputs so they can be unit tested in isolation. Update `cli/commands/train.py` to parse the new flags and instantiate `TrainingConfig`, then call a top-level `run_training_pipeline(config: TrainingConfig) -> TrainingResult` function that coordinates the modules.

For dataset creation, eliminate the Python for-loop that slices arrays one sequence at a time. Use `numpy.lib.stride_tricks.sliding_window_view` to create the `(num_samples, sequence_length, num_features)` tensor in one pass, or build a `tf.data.Dataset` that windows the normalized feature matrix lazily. Wrap the resulting dataset in `.cache().shuffle(buffer_size) .batch(batch_size) .prefetch(tf.data.AUTOTUNE)` so TensorFlow can overlap preprocessing with training. Provide tests that feed small synthetic DataFrames and assert that the sequences match the previous implementation’s output order and values.

Introduce configurability by threading the new CLI flags through the trainer. Allow users to override epochs, batch size, and sequence length; ensure defaults match the current hard-coded values to preserve historical behavior. Add boolean flags like `--skip-plots`, `--skip-robustness`, and `--skip-onnx`. When a flag is true, skip the corresponding stage entirely and emit a log line noting the skip. Record these decisions inside the metadata JSON (for example, `"diagnostics": {"plots": false, ...}`). Do not change the existing default of running every diagnostic and exporting ONNX so that automated jobs continue to behave identically unless a maintainer opts out.

For performance enhancements, detect whether a GPU is available by checking `tf.config.list_physical_devices("GPU")`. If a GPU exists and the user has not disabled the optimization via `--no-mixed-precision`, call `tf.keras.mixed_precision.set_global_policy("mixed_float16")` and `tf.config.optimizer.set_jit(True)` before model creation. Document in this plan and in `docs/prediction.md` how to disable the feature if training becomes unstable. Consider increasing the default batch size when mixed precision is active, but respect the CLI override.

Finally, validate the improvements. Re-run the original command after the refactor, capture the new timings, and compute the percentage improvement for each stage. Compare RMSE values before and after; if the difference exceeds one percent, investigate whether numerical drift occurred in preprocessing or training and adjust accordingly. Update this plan’s Outcomes section with the results and list any follow-up tasks. Run `python tests/run_tests.py unit` and any integration suites that touch the trainer (for example, scripts that assume certain metadata fields) to ensure nothing regresses. Document how to reproduce the verification so a future maintainer can repeat it.

## Outcomes & Retrospective

(To be completed when the plan finishes. Include the measured speedup, final RMSE comparison, tests executed, and any follow-up tickets.)

## Revision History

2025-10-26: Reformatted and expanded the plan to comply with `.agents/PLANS.md`, clarified scope, added definitions, and detailed execution steps.
