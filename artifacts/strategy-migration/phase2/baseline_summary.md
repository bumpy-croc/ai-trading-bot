# Strategy Migration Baseline (Legacy Contract)

> **Status (2025-10)**: Retained for historical benchmarking; component strategies now run
> without the legacy contract in production.

| Scenario | Strategy | Timeframe | Dataset/Steps | Trades | Final Balance | Return % | Wall Time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| backtest | ml_basic | 1h | 721 rows | 300 | $10,147.26 | 1.47% | 3.00 |
| live_paper | ml_basic | 1h | 50 steps | 50 | $10,031.46 | 0.31% | 67.34 |
| backtest | ml_adaptive | 1h | 721 rows | 300 | $10,147.26 | 1.47% | 1.44 |
| live_paper | ml_adaptive | 1h | 50 steps | 50 | $10,028.96 | 0.29% | 52.18 |
