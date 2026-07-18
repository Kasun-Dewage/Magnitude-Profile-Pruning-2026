import os
import gc
import json
import time
import argparse
import torch
import torch.nn as nn
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from prune_experiment import (
    set_seed, is_decoder_model, is_seq2seq_model,
    get_linears, group_attn_layers,
    head_scores_magnitude, head_scores_mp, prune_heads_globally,
    head_scores_wanda, head_scores_sparsegpt, head_scores_grad,
    head_scores_mp_g, head_scores_hybrid,
    collect_attn_act_norms, collect_attn_hessians, collect_attn_grads,
    load_calib_texts,
    compute_sparsity,
)


def count_nonzero_params(model):
    total = 0
    nonzero = 0
    for n, p in model.named_parameters():
        total += p.numel()
        nonzero += (p != 0).sum().item()
    return nonzero, total


def count_attn_active_heads(model):
    entries = get_linears(model, only_attn=True)
    layers = group_attn_layers(entries)
    cfg = model.config
    n_heads = getattr(cfg, "num_attention_heads", None)
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads) or n_heads
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
    head_dim = hidden // n_heads if hidden and n_heads else None
    total_q_heads = 0
    active_q_heads = 0
    for lidx, mods in layers.items():
        if "Q" not in mods or "O" not in mods:
            continue
        Wq = mods["Q"][1].weight.data
        for h in range(n_heads):
            total_q_heads += 1
            slc = Wq[h * head_dim:(h + 1) * head_dim, :]
            if (slc != 0).any().item():
                active_q_heads += 1
    return {
        "n_heads_per_layer": n_heads,
        "n_kv_heads_per_layer": n_kv_heads,
        "head_dim": head_dim,
        "total_q_heads": total_q_heads,
        "active_q_heads": active_q_heads,
        "head_active_frac": active_q_heads / max(1, total_q_heads),
    }


def estimate_attn_flops(model, seq_len=128):
    cfg = model.config
    n_heads = getattr(cfg, "num_attention_heads", None)
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads) or n_heads
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
    head_dim = hidden // n_heads if hidden and n_heads else None
    n_layers = getattr(cfg, "num_hidden_layers",
                       getattr(cfg, "n_layer",
                               getattr(cfg, "num_layers", None)))
    if not (n_heads and head_dim and n_layers):
        return None

    entries = get_linears(model, only_attn=True)
    layers = group_attn_layers(entries)

    total_flops = 0.0
    active_flops = 0.0
    for lidx, mods in layers.items():
        if "Q" not in mods or "O" not in mods:
            continue
        Wq = mods["Q"][1].weight.data
        for h in range(n_heads):
            slc = Wq[h * head_dim:(h + 1) * head_dim, :]
            is_active = (slc != 0).any().item()
            qkv_flops = 2 * 4 * seq_len * head_dim * hidden
            attn_flops = 2 * 2 * seq_len * seq_len * head_dim
            head_total = qkv_flops + attn_flops
            total_flops += head_total
            if is_active:
                active_flops += head_total

    return {
        "seq_len": seq_len,
        "total_attn_flops": total_flops,
        "active_attn_flops": active_flops,
        "flops_reduction_frac": 1.0 - active_flops / max(1.0, total_flops),
    }


def time_forward(model, tokenizer, device, seq_len=128, batch=4,
                 warmup=5, iters=20, is_seq2seq=False):
    model.eval()
    vocab = tokenizer.vocab_size
    ids = torch.randint(0, max(2, vocab - 1), (batch, seq_len),
                        dtype=torch.long, device=device)
    am = torch.ones_like(ids)
    if is_seq2seq:
        dec_ids = torch.full((batch, 1), tokenizer.pad_token_id or 0,
                             dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            if is_seq2seq:
                model(input_ids=ids, attention_mask=am, decoder_input_ids=dec_ids)
            else:
                model(input_ids=ids, attention_mask=am)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            if is_seq2seq:
                model(input_ids=ids, attention_mask=am, decoder_input_ids=dec_ids)
            else:
                model(input_ids=ids, attention_mask=am)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
    return (t1 - t0) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="roberta-base")
    p.add_argument("--method", default="mp_head",
                   choices=["dense", "mag_head", "mp_head",
                            "wanda_head", "sparsegpt_head", "grad_head",
                            "mp_head_g", "hybrid_head"])
    p.add_argument("--sparsity", type=float, default=0.25)
    p.add_argument("--head_z", type=float, default=2.0)
    p.add_argument("--head_calib_samples", type=int, default=64)
    p.add_argument("--head_grad_samples", type=int, default=32)
    p.add_argument("--max_len", type=int, default=512)
    p.add_argument("--calib_dataset", default="wikitext2",
                   choices=["wikitext2", "c4_small"])
    p.add_argument("--wanda_norm", default="l2", choices=["l2", "l1", "linf"])
    p.add_argument("--hybrid_lam_mag", type=float, default=1.0)
    p.add_argument("--hybrid_lam_mp", type=float, default=1.0)
    p.add_argument("--hybrid_lam_wanda", type=float, default=1.0)
    p.add_argument("--mpg_query_weight", type=float, default=1.0)
    p.add_argument("--mpg_kv_weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "float16", "bfloat16"])
    p.add_argument("--output_dir", default="./output/cost")
    p.add_argument("--device_map", default="auto")
    p.add_argument("--trust_remote_code", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    decoder = is_decoder_model(args.model)
    seq2seq = is_seq2seq_model(args.model)

    tok_kwargs = {"use_fast": True}
    if args.trust_remote_code:
        tok_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tok_kwargs)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else tokenizer.unk_token

    model_kwargs = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
    if args.trust_remote_code:
        model_kwargs["trust_remote_code"] = True

    if seq2seq:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, **model_kwargs).to(device)
    elif decoder:
        model_kwargs["device_map"] = args.device_map
        model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForMaskedLM.from_pretrained(args.model, **model_kwargs).to(device)
    model.eval()

    nz_before, total_before = count_nonzero_params(model)
    flops_before = estimate_attn_flops(model, seq_len=args.seq_len)
    head_before = count_attn_active_heads(model)
    t_before = time_forward(model, tokenizer, device,
                            seq_len=args.seq_len, batch=args.batch,
                            warmup=args.warmup, iters=args.iters,
                            is_seq2seq=seq2seq)

    if args.method != "dense":
        cfg = model.config
        n_heads = getattr(cfg, "num_attention_heads", None)
        n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads) or n_heads
        hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
        head_dim = hidden // n_heads if hidden and n_heads else None

        attn_entries = get_linears(model, only_attn=True)
        layers = group_attn_layers(attn_entries)
        layers = {k: v for k, v in layers.items() if all(r in v for r in "QKVO")}

        if args.method == "mag_head":
            scorer = lambda lm, n, hd, n_kv_heads=n_kv_heads: \
                head_scores_magnitude(lm, n, hd, n_kv_heads=n_kv_heads)
        elif args.method == "mp_head":
            scorer = lambda lm, n, hd, n_kv_heads=n_kv_heads: \
                head_scores_mp(lm, n, hd, z=args.head_z, n_kv_heads=n_kv_heads)
        elif args.method == "wanda_head":
            texts = load_calib_texts(args.calib_dataset, args.head_calib_samples)
            layer_norms = collect_attn_act_norms(
                model, tokenizer, texts, layers, device,
                max_samples=args.head_calib_samples, max_len=args.max_len,
                wanda_norm=args.wanda_norm, is_seq2seq=seq2seq,
            )

            def scorer(lm, n, hd, n_kv_heads=n_kv_heads, _ln=layer_norms):
                lid = None
                for lidx, mods in layers.items():
                    if mods is lm:
                        lid = lidx
                        break
                an = _ln.get(lid) if lid is not None else None
                return head_scores_wanda(lm, n, hd, act_norms=an, n_kv_heads=n_kv_heads)
        elif args.method == "sparsegpt_head":
            texts = load_calib_texts(args.calib_dataset, args.head_calib_samples)
            layer_hess = collect_attn_hessians(
                model, tokenizer, texts, layers, device,
                max_samples=args.head_calib_samples, max_len=args.max_len,
                is_seq2seq=seq2seq,
            )

            def scorer(lm, n, hd, n_kv_heads=n_kv_heads, _lh=layer_hess):
                lid = None
                for lidx, mods in layers.items():
                    if mods is lm:
                        lid = lidx
                        break
                hh = _lh.get(lid) if lid is not None else None
                return head_scores_sparsegpt(lm, n, hd, hessians=hh, n_kv_heads=n_kv_heads)
        elif args.method == "grad_head":
            texts = load_calib_texts(args.calib_dataset, args.head_grad_samples)
            layer_grads = collect_attn_grads(
                model, tokenizer, texts, layers, device,
                max_samples=args.head_grad_samples, max_len=args.max_len,
                is_seq2seq=seq2seq, is_decoder=decoder,
            )

            def scorer(lm, n, hd, n_kv_heads=n_kv_heads, _lg=layer_grads):
                lid = None
                for lidx, mods in layers.items():
                    if mods is lm:
                        lid = lidx
                        break
                gg = _lg.get(lid) if lid is not None else None
                return head_scores_grad(lm, n, hd, grads=gg, n_kv_heads=n_kv_heads)
        elif args.method == "mp_head_g":
            scorer = lambda lm, n, hd, n_kv_heads=n_kv_heads: \
                head_scores_mp_g(lm, n, hd, z=args.head_z, n_kv_heads=n_kv_heads,
                                 query_weight=args.mpg_query_weight,
                                 kv_weight=args.mpg_kv_weight)
        elif args.method == "hybrid_head":
            texts = load_calib_texts(args.calib_dataset, args.head_calib_samples)
            layer_norms = collect_attn_act_norms(
                model, tokenizer, texts, layers, device,
                max_samples=args.head_calib_samples, max_len=args.max_len,
                wanda_norm=args.wanda_norm, is_seq2seq=seq2seq,
            )

            def scorer(lm, n, hd, n_kv_heads=n_kv_heads, _ln=layer_norms):
                lid = None
                for lidx, mods in layers.items():
                    if mods is lm:
                        lid = lidx
                        break
                an = _ln.get(lid) if lid is not None else None
                return head_scores_hybrid(
                    lm, n, hd, act_norms=an, z=args.head_z, n_kv_heads=n_kv_heads,
                    lam_mag=args.hybrid_lam_mag, lam_mp=args.hybrid_lam_mp,
                    lam_wanda=args.hybrid_lam_wanda,
                )
        else:
            raise ValueError("Unsupported method for cost: {}".format(args.method))
        prune_heads_globally(layers, n_heads, head_dim, args.sparsity, scorer,
                              n_kv_heads=n_kv_heads)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    nz_after, total_after = count_nonzero_params(model)
    flops_after = estimate_attn_flops(model, seq_len=args.seq_len)
    head_after = count_attn_active_heads(model)
    t_after = time_forward(model, tokenizer, device,
                           seq_len=args.seq_len, batch=args.batch,
                           warmup=args.warmup, iters=args.iters,
                           is_seq2seq=seq2seq)

    result = {
        "model": args.model,
        "method": args.method,
        "sparsity_target": args.sparsity,
        "head_z": args.head_z,
        "seed": args.seed,
        "seq_len": args.seq_len,
        "batch": args.batch,
        "iters": args.iters,
        "dtype": args.dtype,
        "params_total": total_before,
        "params_nonzero_before": nz_before,
        "params_nonzero_after": nz_after,
        "params_reduction_frac": 1.0 - nz_after / max(1, nz_before),
        "active_heads_before": head_before,
        "active_heads_after": head_after,
        "attn_flops_before": flops_before,
        "attn_flops_after": flops_after,
        "wallclock_s_before": t_before,
        "wallclock_s_after": t_after,
        "speedup_x": t_before / max(1e-9, t_after),
    }

    print("COST RESULT")
    print(json.dumps(result, indent=2, default=str))

    safe_model = args.model.replace("/", "_")
    tag = "{}_{}_s{:.3f}_hz{:.1f}_seed{}_cost".format(
        safe_model, args.method, args.sparsity, args.head_z, args.seed,
    )
    out_path = os.path.join(args.output_dir, "{}.json".format(tag))
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("Saved to {}".format(out_path))


if __name__ == "__main__":
    main()
