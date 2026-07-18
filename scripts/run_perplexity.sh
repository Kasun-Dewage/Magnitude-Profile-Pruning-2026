#!/usr/bin/env bash
# Reproduce Table I (WikiText-2 perplexity) for a given model.
# Usage: bash scripts/run_perplexity.sh [MODEL] [DTYPE]
#   e.g. bash scripts/run_perplexity.sh roberta-large float32
#        bash scripts/run_perplexity.sh facebook/opt-6.7b float16
set -e

MODEL=${1:-roberta-large}
DTYPE=${2:-float32}
OUT=./output/prune

for SPARSITY in 0.125 0.25 0.50; do
  for METHOD in mp_head_g mag_head wanda_head sparsegpt_head grad_head hybrid_head; do
    echo "=== ${MODEL} | ${METHOD} | s=${SPARSITY} ==="
    python src/prune_experiment.py \
      --model "${MODEL}" \
      --method "${METHOD}" \
      --sparsity "${SPARSITY}" \
      --head_z 2.0 \
      --dtype "${DTYPE}" \
      --eval_dataset wikitext2 \
      --calib_dataset wikitext2 \
      --seed 0 \
      --output_dir "${OUT}"
  done
done

python src/aggregate_results.py --mode prune --input_dir "${OUT}" \
  --output_csv "${OUT}/all_results.csv" --output_json "${OUT}/all_results.json"
