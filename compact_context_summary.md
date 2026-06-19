# Compact Context Summary

## Current Best Result

- Best model: `pure_lag_transformer_return`
- Seed: `5`
- Context: `72`
- OOS RMSE on `data_test_actual.csv`: `276.63`
- MAE: `227.46`
- MAPE: `1.41%`

## Benchmark Reference

- ElasticNet template RMSE: `386.36`
- Previous target: `290`
- Current best already beats both.

## What Was Tried

- `Lag-Llama`-style AR backbone hybrid
- local linear trend + lag residual transformer
- pure lag transformer on return space
- context sweeps
- seed sweeps

## Key Finding

- The pure lag transformer is the best-performing family so far.
- Performance is highly sensitive to random seed.
- Hybrid residual with ElasticNet did not improve over the pure lag transformer.

## Next Step

- Main task now is to reduce seed variance.

## Recommended Next Experiments

- Multi-seed ensemble of the best pure lag transformer settings.
- Fixed-seed stability sweep around `seed=5`.
- Check whether averaging several low-RMSE seeds lowers OOS RMSE further.
- Keep `return` target formulation and `context=72` as the anchor.

## Useful Artifacts

- `lag_transformer_sweep.py`
- `lag_transformer_sweep_results.csv`
- `lag_transformer_best_predictions.csv`
- `lag_transformer_best_report.md`
- `lag_llama_backbone_forecast.py`
- `lag_llama_oos_report.md`
