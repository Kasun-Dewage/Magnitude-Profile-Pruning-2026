<div align="center">

# Magnitude Profile Pruning (MP / MP-G)

### Calibration-Free Structured Attention Head Removal for Transformer Compression

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.x-yellow.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/Paper-PDF-b31b1b.svg)](paper/Magnitude_Profile_Pruning.pdf)

*Kasun Dewage, Marianna Pensky, Heranga K. Rathnasekara, Suranadi De Silva*

**University of Central Florida · Old Dominion University**

</div>

---

## 🔍 Overview

**Magnitude Profile (MP) scoring** is a *training-free, calibration-free* criterion for structured attention head pruning in Transformer language models. Instead of relying on calibration data, gradients, or Hessian estimation, MP identifies dispensable heads through **statistical outlier detection on weight row norms**:

- Heads whose projection weights sit within the population bulk → **pruned**
- Heads with **outlier** weight norms (which carry disproportionate representational capacity) → **preserved**

**MP-G** extends MP to modern **Grouped Query Attention (GQA)** architectures (LLaMA-3, Mistral) by distributing shared key-value group scores across their associated query heads.

```
exc(rᵢ) = max(0, rᵢ − (μ + z·σ))                          # per-row excess, z = 2.0 by default

φ(h) = α_q · [exc_Q(h) + exc_O(h)] + α_kv · exc_KV(⌊h/g⌋)/g   # MP-G head score
```

Heads are ranked globally by ascending score and the lowest ⌊s·L·H⌋ heads are zeroed. **No calibration data. No forward or backward passes. No Hessian. O(p) cost — runs in seconds.**

## ✨ Highlights

| | MP / MP-G | Wanda-Head | SparseGPT-Head | Gradient-Head |
|---|:---:|:---:|:---:|:---:|
| Calibration data | ❌ none | ✅ 64 samples | ✅ 64 samples | ✅ 32 samples |
| Forward passes | ❌ | ✅ | ✅ | ✅ |
| Backward passes | ❌ | ❌ | ❌ | ✅ |
| Hessian estimation | ❌ | ❌ | ✅ | ❌ |
| Cost | **seconds** | minutes | minutes | minutes |

- 🥇 **Best perplexity on OPT-6.7B at all sparsity levels** (18.46 @ 12.5%, 27.87 @ 25%, 152.0 @ 50%) and on RoBERTa-large at 12.5% and 25% (7.27, 10.28)
- 🥈 **Top-two performance in most primary model–sparsity settings** across 5 models (RoBERTa-base/large, OPT-6.7B, LLaMA-3-8B, Mistral-7B)
- ⚡ At 50% head sparsity: up to **16% total parameter reduction** with **50% attention FLOP savings**

## 📊 Main Results

WikiText-2 perplexity (↓) at head sparsity *s*, with z = 2.0. RoBERTa models report pseudo-perplexity (FP32); decoders report causal LM perplexity (FP16). **Best in bold.**

| Method | RoBERTa-base<br>12.5 / 25 / 50 | RoBERTa-large<br>12.5 / 25 / 50 | OPT-6.7B<br>12.5 / 25 / 50 | LLaMA-3-8B<br>12.5 / 25 / 50 | Mistral-7B<br>12.5 / 25 / 50 |
|---|---|---|---|---|---|
| Gradient-Head | **8.22 / 11.8 / 64.7** | 7.79 / 81.8 / 13485 | 28.5 / – / – | **11.1 / 17.7 / 47.9** | 93.6 / **13.3 / 54.1** |
| Wanda-Head | 8.89 / 16.5 / 103.8 | 8.01 / 13.7 / 86.0 | 42.5 / 152.3 / 406.3 | 1608 / – / 20009 | 107.0 / 472.6 / 552.8 |
| SparseGPT-Head | 11.2 / 113.4 / 4730 | 7.90 / 10.3 / **42.2** | 257.9 / 479.0 / 2034 | 142.3 / 344.6 / 1870 | 75.3 / 430.8 / 871.8 |
| **MP-G (ours)** | 10.0 / 17.8 / 71.3 | **7.27 / 10.3** / 45.8 | **18.5 / 27.9 / 152.0** | 20.4 / 41.4 / 7227 | 22.5 / 121.0 / 1072 |
| Hybrid | 9.09 / 14.0 / 96.1 | 7.82 / 11.1 / 85.6 | 25.6 / 136.3 / 499.6 | 40.3 / 467.9 / 13428 | **14.3** / 1290 / 4339 |

Full raw results live in [`results/`](results/):

| File | Contents |
|---|---|
| [`all_results.csv`](results/all_results.csv) | Perplexity experiments (all models, methods, sparsities, z values) |
| [`all_glue.csv`](results/all_glue.csv) | GLUE fine-tune-then-prune results (SST-2, MRPC, RTE on RoBERTa-large) |
| [`all_cost.csv`](results/all_cost.csv) | Efficiency analysis (params, FLOPs, wall-clock) |

## 🗂️ Repository Structure

```
Magnitude-Profile-Pruning-2026/
├── paper/
│   └── Magnitude_Profile_Pruning.pdf   # The paper
├── src/
│   ├── prune_experiment.py             # Core: MP/MP-G scoring, all baselines, WikiText-2 PPL eval
│   ├── glue_experiment.py              # GLUE fine-tune → prune → evaluate pipeline
│   ├── cost_experiment.py              # Params / FLOPs / wall-clock efficiency measurement
│   └── aggregate_results.py            # Collect per-run JSONs into all_*.csv / all_*.json
├── results/
│   ├── all_results.csv                 # Perplexity results
│   ├── all_glue.csv                    # GLUE results
│   └── all_cost.csv                    # Efficiency results
├── scripts/
│   ├── run_perplexity.sh               # Example: reproduce Table I entries
│   ├── run_glue.sh                     # Example: reproduce Table III entries
│   └── run_cost.sh                     # Example: reproduce Table IV / efficiency numbers
├── requirements.txt
├── CITATION.cff
├── LICENSE
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/Kasun-Dewage/Magnitude-Profile-Pruning-2026.git
cd Magnitude-Profile-Pruning-2026

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

> **Note:** Gated models (e.g. `meta-llama/Meta-Llama-3-8B`) require a Hugging Face token: `huggingface-cli login`.

## 🚀 Quick Start

### 1 · Prune with MP-G and evaluate perplexity (zero calibration!)

```bash
python src/prune_experiment.py \
    --model roberta-large \
    --method mp_head_g \
    --sparsity 0.25 \
    --head_z 2.0 \
    --eval_dataset wikitext2 \
    --output_dir ./output/prune
```

Decoder example (FP16, as in the paper):

```bash
python src/prune_experiment.py \
    --model facebook/opt-6.7b \
    --method mp_head_g \
    --sparsity 0.50 \
    --head_z 2.0 \
    --dtype float16 \
    --eval_dataset wikitext2 \
    --output_dir ./output/prune
```

### 2 · Run a calibration-dependent baseline for comparison

```bash
python src/prune_experiment.py \
    --model facebook/opt-6.7b \
    --method wanda_head \
    --sparsity 0.25 \
    --head_calib_samples 64 \
    --calib_dataset wikitext2 \
    --dtype float16
```

Available `--method` choices for head pruning: `mp_head`, `mp_head_g`, `mag_head`, `wanda_head`, `sparsegpt_head`, `grad_head`, `hybrid_head` (plus unstructured baselines `magnitude`, `wanda`, `sparsegpt`, `random`, and `dense`).

### 3 · GLUE (fine-tune-then-prune)

```bash
python src/glue_experiment.py \
    --model roberta-large \
    --task mrpc \
    --method mp_head_g \
    --sparsity 0.25 \
    --prune_when after \
    --epochs 5 --lr 1e-5 --bs 16
```

Tasks: `sst2` (3 epochs), `mrpc` (5 epochs), `rte` (10 epochs) — all with `lr=1e-5`, `bs=16` as in the paper.

### 4 · Efficiency measurement

```bash
python src/cost_experiment.py \
    --model roberta-large \
    --method mp_head \
    --sparsity 0.50 \
    --seq_len 512 --batch 1
```

### 5 · Aggregate all runs into CSV

```bash
python src/aggregate_results.py --mode prune --input_dir ./output/prune \
    --output_csv ./output/prune/all_results.csv --output_json ./output/prune/all_results.json

python src/aggregate_results.py --mode glue --input_dir ./output/glue \
    --output_csv ./output/glue/all_glue.csv --output_json ./output/glue/all_glue.json

python src/aggregate_results.py --mode cost --input_dir ./output/cost \
    --output_csv ./output/cost/all_cost.csv --output_json ./output/cost/all_cost.json
```

Batch reproduction scripts are provided in [`scripts/`](scripts/).

## 🔑 Key Hyperparameters

| Argument | Default | Description |
|---|---|---|
| `--sparsity` | 0.5 | Target head sparsity *s* ∈ (0, 1) |
| `--head_z` | 2.0 | Outlier threshold *z* in μ + z·σ (1.5 = conservative, 2.5 = aggressive) |
| `--mpg_query_weight` | 1.0 | α_q — weight of query-side excess (Q + O) |
| `--mpg_kv_weight` | 1.0 | α_kv — weight of shared KV-group excess |
| `--head_calib_samples` | 64 | Calibration samples (baselines only) |
| `--head_grad_samples` | 32 | Gradient samples (Gradient-Head only) |
| `--seed` | 0 | Random seed (paper uses seed 0) |

**Tip from the paper (Table II):** the optimal *z* is model-dependent — MHA models like RoBERTa-base often prefer z = 1.5, while GQA models (LLaMA-3) and larger encoders benefit from z = 2.0–2.5. At 50% sparsity on LLaMA-3-8B, z = 2.5 improves perplexity from 7227 → 282.8.

## 🧠 Models Evaluated

| Model | Params | Heads × Layers | Attention |
|---|---|---|---|
| `roberta-base` | 125M | 12 × 12 | MHA |
| `roberta-large` | 355M | 16 × 24 | MHA |
| `facebook/opt-6.7b` | 6.7B | 32 × 32 | MHA |
| `meta-llama/Meta-Llama-3-8B` | 8B | 32 × 32 | GQA (8 KV heads) |
| `mistralai/Mistral-7B-v0.1` | 7B | 32 × 32 | GQA (8 KV heads) |

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@article{dewage2026magnitude,
  title   = {Magnitude Profile Pruning: Calibration-Free Structured Attention Head
             Removal for Transformer Compression},
  author  = {Dewage, Kasun and Pensky, Marianna and Rathnasekara, Heranga K. and
             De Silva, Suranadi},
  year    = {2026}
}
```

## 📄 License

This project is released under the [MIT License](LICENSE).

## 🙏 Acknowledgements

Our baselines adapt criteria from [Wanda](https://github.com/locuslab/wanda) (Sun et al., 2024), [SparseGPT](https://github.com/IST-DASLab/sparsegpt) (Frantar & Alistarh, 2023), and gradient-based head importance (Michel et al., 2019). Built on [PyTorch](https://pytorch.org/) and [🤗 Transformers](https://github.com/huggingface/transformers).
