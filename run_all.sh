#!/bin/bash
set -e

echo "=== Running all models in parallel ==="

python model_a/run_experiment.py &
python model_b/run_experiment.py &
python model_c/run_experiment.py &
python model_d/run_experiment.py &

echo "Waiting for all models to finish..."
wait
echo "=== All models complete ==="

echo ""
echo "=== Results Summary ==="
for m in model_a model_b model_c model_d; do
    if [ -f "$m/results/metrics.json" ]; then
        echo "--- $m ---"
        python -c "import json; d=json.load(open('$m/results/metrics.json')); print(f\"  RMSE={d['RMSE']:.2f}  MAPE={d['MAPE']:.2f}%  MAE={d['MAE']:.2f}  R2={d['R2']:.4f}\")"
    fi
done
