# LightGBM OOS Benchmark

Best model: LightGBM
Feature set: target_only
Target: diff
Preset: balanced
RMSE: 1163.621340
MAE: 1093.011362
MAPE: 6.726457%

Top 5 candidates:
| model    | feature_set   | target     | preset   |    rmse |     mae |    mape |
|:---------|:--------------|:-----------|:---------|--------:|--------:|--------:|
| LightGBM | target_only   | diff       | balanced | 1163.62 | 1093.01 | 6.72646 |
| LightGBM | basic         | diff       | balanced | 1163.62 | 1093.01 | 6.72646 |
| LightGBM | target_only   | log_return | balanced | 1271    | 1180.92 | 7.29121 |
| LightGBM | basic         | log_return | balanced | 1271    | 1180.92 | 7.29121 |
| Naive    | -             | level      | -        | 1284.21 | 1132.9  | 6.90072 |