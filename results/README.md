# Results

Aggregated CSVs from all paper experiments (seed 0).

| File | Produced by | Description |
|---|---|---|
| `all_results.csv` | `src/prune_experiment.py` → `src/aggregate_results.py --mode prune` | WikiText-2 perplexity / pseudo-perplexity for every model × method × sparsity × z configuration (Tables I–II, IV) |
| `all_glue.csv` | `src/glue_experiment.py` → `--mode glue` | GLUE fine-tune-then-prune results for RoBERTa-large on SST-2 / MRPC / RTE (Table III) |
| `all_cost.csv` | `src/cost_experiment.py` → `--mode cost` | Parameter counts, attention FLOPs, and wall-clock timing before/after pruning (Sec. V-D) |

Key columns:
- `method` — scoring criterion (`mp_head_g` = ours; `mp_head`, `mag_head`, `wanda_head`, `sparsegpt_head`, `grad_head`, `hybrid_head` = variants/baselines)
- `sparsity_target` — target head sparsity (0.125 / 0.25 / 0.5)
- `head_z` — MP outlier threshold z
- `ppl` / `pseudo_ppl` — causal LM perplexity (decoders) / MLM pseudo-perplexity (encoders)
- `metrics` (GLUE) — JSON with accuracy/F1
