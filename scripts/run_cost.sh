#!/usr/bin/env bash
# Reproduce the efficiency analysis (params / FLOPs / wall-clock) at 50% sparsity.
# Usage: bash scripts/run_cost.sh [MODEL] [DTYPE]
set -e

MODEL=${1:-roberta-large}
DTYPE=${2:-float32}
OUT=./output/cost

for METHOD in mp_head mag_head wanda_head sparsegpt_head grad_head hybrid_head; do
  echo "=== ${MODEL} | ${METHOD} | s=0.50 ==="
  python src/cost_experiment.py \
    --model "${MODEL}" \
    --method "${METHOD}" \
    --sparsity 0.50 \
    --head_z 2.0 \
    --seq_len 512 --batch 1 \
    --dtype "${DTYPE}" \
    --seed 0 \
    --output_dir "${OUT}"
done

python src/aggregate_results.py --mode cost --input_dir "${OUT}" \
  --output_csv "${OUT}/all_cost.csv" --output_json "${OUT}/all_cost.json"
