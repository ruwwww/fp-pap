# Leakage Report

- Engineered features inspected: 187
- Future-looking feature patterns found: 0
- Raw exogenous features use same-timestamp values only; no target-derived future columns are used.
- Target-history features are built with `shift(k)` / rolling windows only.
- No row uses future target values when creating lags, moving averages, or returns.
- Feature timestamp constraint satisfied: feature_time <= target_time for causal features.

Conclusion: no explicit leakage detected in the proposed feature set.