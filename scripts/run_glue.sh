#!/usr/bin/env bash
# Reproduce Table III (GLUE, fine-tune-then-prune) on RoBERTa-large.
# Usage: bash scripts/run_glue.sh
set -e

MODEL=roberta-large
OUT=./output/glue

declare -A EPOCHS=( [sst2]=3 [mrpc]=5 [rte]=10 )

for TASK in sst2 mrpc rte; do
  for SPARSITY in 0.25 0.50; do
    for METHOD in mp_head_g wanda_head sparsegpt_head hybrid_head; do
      echo "=== ${TASK} | ${METHOD} | s=${SPARSITY} ==="
      python src/glue_experiment.py \
        --model "${MODEL}" \
        --task "${TASK}" \
        --method "${METHOD}" \
        --sparsity "${SPARSITY}" \
        --prune_when after \
        --epochs "${EPOCHS[$TASK]}" \
        --lr 1e-5 --bs 16 --seed 0 \
        --output_dir "${OUT}"
    done
  done
done

python src/aggregate_results.py --mode glue --input_dir "${OUT}" \
  --output_csv "${OUT}/all_glue.csv" --output_json "${OUT}/all_glue.json"
