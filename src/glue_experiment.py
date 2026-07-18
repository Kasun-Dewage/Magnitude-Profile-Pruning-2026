import os
import gc
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import evaluate

from prune_experiment import (
    get_linears, group_attn_layers, set_seed,
    random_mask, magnitude_mask, wanda_mask, wanda_mp_mask,
    sparsegpt_prune_tensor, apply_mask, apply_tensor,
    head_scores_magnitude, head_scores_mp, prune_heads_globally,
    head_scores_wanda, head_scores_sparsegpt, head_scores_grad,
    head_scores_mp_g, head_scores_hybrid,
    collect_attn_act_norms, collect_attn_hessians, collect_attn_grads,
    per_layer_sparsity, compute_sparsity, collect_calibration_data,
    load_calib_texts,
)


TASK_INFO = {
    "sst2": {"keys": ("sentence", None), "metric": "accuracy", "n_labels": 2},
    "mrpc": {"keys": ("sentence1", "sentence2"), "metric": "f1", "n_labels": 2},
    "rte":  {"keys": ("sentence1", "sentence2"), "metric": "accuracy", "n_labels": 2},
}


def preprocess_glue(examples, tokenizer, keys, max_len=128):
    k1, k2 = keys
    if k2 is None:
        return tokenizer(examples[k1], truncation=True, max_length=max_len)
    return tokenizer(examples[k1], examples[k2], truncation=True, max_length=max_len)


def fine_tune(model, tokenizer, task, output_dir, n_epochs=3, lr=2e-5, bs=16,
              max_len=128, seed=0):
    info = TASK_INFO[task]
    raw = load_dataset("glue", task)

    def _prep(ex):
        return preprocess_glue(ex, tokenizer, info["keys"], max_len)

    enc = raw.map(_prep, batched=True)
    enc = enc.rename_column("label", "labels")
    cols = ["input_ids", "attention_mask", "labels"]
    enc.set_format("torch", columns=cols)

    train_split = enc["train"]
    val_split = enc["validation"]

    metric = evaluate.load("glue", task)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    ta_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=n_epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs * 2,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=100,
        save_strategy="no",
        seed=seed,
        report_to="none",
        fp16=False,
        dataloader_num_workers=2,
    )
    try:
        args = TrainingArguments(eval_strategy="no", **ta_kwargs)
    except TypeError:
        args = TrainingArguments(evaluation_strategy="no", **ta_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_split,
        eval_dataset=val_split,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    res = trainer.evaluate()
    return res


def apply_pruning_to_seq_cls(model, tokenizer, method, sparsity,
                              prune_target, calib_samples, max_len,
                              alpha, z, head_z, seed,
                              layerwise_sparsity, wanda_norm,
                              calib_dataset, device,
                              head_calib_samples=64, head_grad_samples=32,
                              hybrid_lam_mag=1.0, hybrid_lam_mp=1.0,
                              hybrid_lam_wanda=1.0,
                              mpg_query_weight=1.0, mpg_kv_weight=1.0,
                              is_decoder=False, is_seq2seq=False):
    is_head_method = method in ("mag_head", "mp_head",
                                  "wanda_head", "sparsegpt_head",
                                  "grad_head", "mp_head_g", "hybrid_head")
    only_attn = (prune_target == "attn") or is_head_method

    if is_head_method:
        attn_entries = get_linears(model, only_attn=True)
        layers = group_attn_layers(attn_entries)
        layers = {k: v for k, v in layers.items() if all(r in v for r in "QKVO")}
        entries_for_report = attn_entries
    else:
        entries = get_linears(model, only_attn=only_attn)
        entries_for_report = entries

    layer_ids = sorted(set(l for _, _, _, l in entries_for_report))
    sched = per_layer_sparsity(sparsity, len(layer_ids), mode=layerwise_sparsity)
    layer_target = {l: sched[i] for i, l in enumerate(layer_ids)}

    cfg = model.config
    n_heads = getattr(cfg, "num_attention_heads", None)
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads) or n_heads
    hidden = getattr(cfg, "hidden_size", None)
    head_dim = hidden // n_heads if hidden and n_heads else None

    pruned_info = {}

    if method == "dense":
        pass
    elif method == "random":
        role_offset = {"Q": 1, "K": 2, "V": 3, "O": 4, "I": 5, "D": 6, "G": 7, "U": 8}
        for name, m, role, lidx in entries:
            sp = layer_target[lidx]
            mk = random_mask(m.weight, sp,
                             seed=seed + 1000 * lidx + role_offset.get(role, 0))
            apply_mask(m, mk)
    elif method == "magnitude":
        for name, m, role, lidx in entries:
            sp = layer_target[lidx]
            mk = magnitude_mask(m.weight, sp)
            apply_mask(m, mk)
    elif method in ("wanda", "wanda_mp"):
        texts = load_calib_texts(calib_dataset, calib_samples)
        norms, _ = collect_calibration_data(
            model, tokenizer, texts, entries, device,
            max_samples=calib_samples, max_len=max_len,
            need_hessian=False, wanda_norm=wanda_norm,
        )
        for name, m, role, lidx in entries:
            an = norms.get(id(m))
            if an is None:
                continue
            sp = layer_target[lidx]
            if method == "wanda":
                mk = wanda_mask(m.weight, an, sp)
            else:
                mk = wanda_mp_mask(m.weight, an, sp, alpha=alpha, z=z)
            apply_mask(m, mk)
        del norms
    elif method == "sparsegpt":
        texts = load_calib_texts(calib_dataset, calib_samples)
        _, hs = collect_calibration_data(
            model, tokenizer, texts, entries, device,
            max_samples=calib_samples, max_len=max_len,
            need_hessian=True, wanda_norm="l2",
        )
        for name, m, role, lidx in entries:
            H = hs.get(id(m))
            if H is None:
                continue
            sp = layer_target[lidx]
            new_W, _ = sparsegpt_prune_tensor(m.weight, H, sp)
            apply_tensor(m, new_W)
            hs[id(m)] = None
            gc.collect()
        del hs
    elif method == "mag_head":
        scorer = lambda lm, n, hd, n_kv_heads=n_kv_heads: \
            head_scores_magnitude(lm, n, hd, n_kv_heads=n_kv_heads)
        pruned = prune_heads_globally(layers, n_heads, head_dim, sparsity, scorer,
                                       n_kv_heads=n_kv_heads)
        pruned_info["pruned_heads"] = pruned
    elif method == "mp_head":
        scorer = lambda lm, n, hd, n_kv_heads=n_kv_heads: \
            head_scores_mp(lm, n, hd, z=head_z, n_kv_heads=n_kv_heads)
        pruned = prune_heads_globally(layers, n_heads, head_dim, sparsity, scorer,
                                       n_kv_heads=n_kv_heads)
        pruned_info["pruned_heads"] = pruned
    elif method == "wanda_head":
        texts = load_calib_texts(calib_dataset, head_calib_samples)
        layer_norms = collect_attn_act_norms(
            model, tokenizer, texts, layers, device,
            max_samples=head_calib_samples, max_len=max_len,
            wanda_norm=wanda_norm, is_seq2seq=is_seq2seq,
        )

        def scorer(lm, n, hd, n_kv_heads=n_kv_heads, _ln=layer_norms):
            lid = None
            for lidx, mods in layers.items():
                if mods is lm:
                    lid = lidx
                    break
            an = _ln.get(lid) if lid is not None else None
            return head_scores_wanda(lm, n, hd, act_norms=an, n_kv_heads=n_kv_heads)
        pruned = prune_heads_globally(layers, n_heads, head_dim, sparsity, scorer,
                                       n_kv_heads=n_kv_heads)
        pruned_info["pruned_heads"] = pruned
    elif method == "sparsegpt_head":
        texts = load_calib_texts(calib_dataset, head_calib_samples)
        layer_hess = collect_attn_hessians(
            model, tokenizer, texts, layers, device,
            max_samples=head_calib_samples, max_len=max_len,
            is_seq2seq=is_seq2seq,
        )

        def scorer(lm, n, hd, n_kv_heads=n_kv_heads, _lh=layer_hess):
            lid = None
            for lidx, mods in layers.items():
                if mods is lm:
                    lid = lidx
                    break
            hh = _lh.get(lid) if lid is not None else None
            return head_scores_sparsegpt(lm, n, hd, hessians=hh, n_kv_heads=n_kv_heads)
        pruned = prune_heads_globally(layers, n_heads, head_dim, sparsity, scorer,
                                       n_kv_heads=n_kv_heads)
        pruned_info["pruned_heads"] = pruned
    elif method == "grad_head":
        texts = load_calib_texts(calib_dataset, head_grad_samples)
        layer_grads = collect_attn_grads(
            model, tokenizer, texts, layers, device,
            max_samples=head_grad_samples, max_len=max_len,
            is_seq2seq=is_seq2seq, is_decoder=is_decoder,
        )

        def scorer(lm, n, hd, n_kv_heads=n_kv_heads, _lg=layer_grads):
            lid = None
            for lidx, mods in layers.items():
                if mods is lm:
                    lid = lidx
                    break
            gg = _lg.get(lid) if lid is not None else None
            return head_scores_grad(lm, n, hd, grads=gg, n_kv_heads=n_kv_heads)
        pruned = prune_heads_globally(layers, n_heads, head_dim, sparsity, scorer,
                                       n_kv_heads=n_kv_heads)
        pruned_info["pruned_heads"] = pruned
    elif method == "mp_head_g":
        scorer = lambda lm, n, hd, n_kv_heads=n_kv_heads: \
            head_scores_mp_g(lm, n, hd, z=head_z, n_kv_heads=n_kv_heads,
                             query_weight=mpg_query_weight, kv_weight=mpg_kv_weight)
        pruned = prune_heads_globally(layers, n_heads, head_dim, sparsity, scorer,
                                       n_kv_heads=n_kv_heads)
        pruned_info["pruned_heads"] = pruned
    elif method == "hybrid_head":
        texts = load_calib_texts(calib_dataset, head_calib_samples)
        layer_norms = collect_attn_act_norms(
            model, tokenizer, texts, layers, device,
            max_samples=head_calib_samples, max_len=max_len,
            wanda_norm=wanda_norm, is_seq2seq=is_seq2seq,
        )

        def scorer(lm, n, hd, n_kv_heads=n_kv_heads, _ln=layer_norms):
            lid = None
            for lidx, mods in layers.items():
                if mods is lm:
                    lid = lidx
                    break
            an = _ln.get(lid) if lid is not None else None
            return head_scores_hybrid(
                lm, n, hd, act_norms=an, z=head_z, n_kv_heads=n_kv_heads,
                lam_mag=hybrid_lam_mag, lam_mp=hybrid_lam_mp,
                lam_wanda=hybrid_lam_wanda,
            )
        pruned = prune_heads_globally(layers, n_heads, head_dim, sparsity, scorer,
                                       n_kv_heads=n_kv_heads)
        pruned_info["pruned_heads"] = pruned

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pruned_info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="roberta-large")
    p.add_argument("--task", default="sst2", choices=["sst2", "mrpc", "rte"])
    p.add_argument("--method", default="dense",
                   choices=["dense", "random", "magnitude", "wanda",
                            "sparsegpt", "wanda_mp", "mag_head", "mp_head",
                            "wanda_head", "sparsegpt_head", "grad_head",
                            "mp_head_g", "hybrid_head"])
    p.add_argument("--sparsity", type=float, default=0.0)
    p.add_argument("--prune_target", default="all", choices=["all", "attn"])
    p.add_argument("--prune_when", default="before",
                   choices=["before", "after"],
                   help="before = prune then fine-tune; after = fine-tune then prune")
    p.add_argument("--calib_samples", type=int, default=128)
    p.add_argument("--max_calib_len", type=int, default=256)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--z", type=float, default=2.0)
    p.add_argument("--head_z", type=float, default=2.0)
    p.add_argument("--head_calib_samples", type=int, default=64)
    p.add_argument("--head_grad_samples", type=int, default=32)
    p.add_argument("--hybrid_lam_mag", type=float, default=1.0)
    p.add_argument("--hybrid_lam_mp", type=float, default=1.0)
    p.add_argument("--hybrid_lam_wanda", type=float, default=1.0)
    p.add_argument("--mpg_query_weight", type=float, default=1.0)
    p.add_argument("--mpg_kv_weight", type=float, default=1.0)
    p.add_argument("--layerwise_sparsity", default="uniform",
                   choices=["uniform", "linear_inc", "linear_dec"])
    p.add_argument("--wanda_norm", default="l2", choices=["l2", "l1", "linf"])
    p.add_argument("--calib_dataset", default="wikitext2",
                   choices=["wikitext2", "c4_small"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default="./output/glue")
    p.add_argument("--tmp_dir", default="./output/glue_tmp")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.tmp_dir, exist_ok=True)

    info = TASK_INFO[args.task]
    n_labels = info["n_labels"]

    print("Model: {} | Task: {} | Method: {} | Sparsity: {} | When: {}".format(
        args.model, args.task, args.method, args.sparsity, args.prune_when))

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=n_labels,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pruned_info = {}
    if args.method != "dense" and args.prune_when == "before":
        pruned_info = apply_pruning_to_seq_cls(
            model, tokenizer, args.method, args.sparsity,
            args.prune_target, args.calib_samples, args.max_calib_len,
            args.alpha, args.z, args.head_z, args.seed,
            args.layerwise_sparsity, args.wanda_norm,
            args.calib_dataset, device,
            head_calib_samples=args.head_calib_samples,
            head_grad_samples=args.head_grad_samples,
            hybrid_lam_mag=args.hybrid_lam_mag,
            hybrid_lam_mp=args.hybrid_lam_mp,
            hybrid_lam_wanda=args.hybrid_lam_wanda,
            mpg_query_weight=args.mpg_query_weight,
            mpg_kv_weight=args.mpg_kv_weight,
            is_decoder=False, is_seq2seq=False,
        )

    print("Fine-tuning on {}".format(args.task))
    fine_tune(model, tokenizer, args.task,
              os.path.join(args.tmp_dir,
                           "{}_{}_{}_{}_s{:.3f}_seed{}".format(
                               args.model.replace("/", "_"), args.task,
                               args.method, args.prune_when,
                               args.sparsity, args.seed)),
              n_epochs=args.epochs, lr=args.lr, bs=args.bs,
              max_len=args.max_len, seed=args.seed)

    if args.method != "dense" and args.prune_when == "after":
        pruned_info = apply_pruning_to_seq_cls(
            model, tokenizer, args.method, args.sparsity,
            args.prune_target, args.calib_samples, args.max_calib_len,
            args.alpha, args.z, args.head_z, args.seed,
            args.layerwise_sparsity, args.wanda_norm,
            args.calib_dataset, device,
            head_calib_samples=args.head_calib_samples,
            head_grad_samples=args.head_grad_samples,
            hybrid_lam_mag=args.hybrid_lam_mag,
            hybrid_lam_mp=args.hybrid_lam_mp,
            hybrid_lam_wanda=args.hybrid_lam_wanda,
            mpg_query_weight=args.mpg_query_weight,
            mpg_kv_weight=args.mpg_kv_weight,
            is_decoder=False, is_seq2seq=False,
        )

    raw = load_dataset("glue", args.task)
    enc = raw["validation"].map(
        lambda ex: preprocess_glue(ex, tokenizer, info["keys"], args.max_len),
        batched=True,
    )
    enc = enc.rename_column("label", "labels")
    enc.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    metric = evaluate.load("glue", args.task)
    model.eval()
    collator = DataCollatorWithPadding(tokenizer)
    from torch.utils.data import DataLoader
    loader = DataLoader(enc, batch_size=args.bs * 2, collate_fn=collator)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**{k: v for k, v in batch.items() if k != "labels"})
            preds = torch.argmax(out.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
    metrics = metric.compute(predictions=all_preds, references=all_labels)

    attn_only_entries = get_linears(model, only_attn=True)
    all_entries = get_linears(model, only_attn=False)
    sp_attn = compute_sparsity(attn_only_entries)
    sp_all = compute_sparsity(all_entries)

    result = {
        "model": args.model,
        "task": args.task,
        "method": args.method,
        "sparsity_target": args.sparsity,
        "prune_target": args.prune_target,
        "prune_when": args.prune_when,
        "sparsity_attn": sp_attn,
        "sparsity_all_linears": sp_all,
        "head_z": args.head_z,
        "layerwise_sparsity": args.layerwise_sparsity,
        "wanda_norm": args.wanda_norm,
        "calib_dataset": args.calib_dataset,
        "metric_name": info["metric"],
        "metrics": metrics,
        "epochs": args.epochs,
        "lr": args.lr,
        "bs": args.bs,
        "seed": args.seed,
    }
    result.update(pruned_info)

    print("GLUE RESULT")
    print(json.dumps({k: v for k, v in result.items() if k != "pruned_heads"}, indent=2))

    safe_model = args.model.replace("/", "_")
    tag = "{}_{}_{}_s{:.3f}_{}_seed{}".format(
        safe_model, args.task, args.method, args.sparsity,
        args.prune_when, args.seed,
    )
    out_path = os.path.join(args.output_dir, "{}.json".format(tag))
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to {}".format(out_path))


if __name__ == "__main__":
    main()
