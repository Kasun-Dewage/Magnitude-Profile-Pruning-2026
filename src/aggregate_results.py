import os
import json
import csv
import glob
import argparse

AGG_BASENAMES = {
    "all_results.json", "all_results.csv",
    "all_glue.json", "all_glue.csv",
    "all_cost.json", "all_cost.csv",
}


def load_dir(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    files = [f for f in files if os.path.basename(f) not in AGG_BASENAMES]
    rows = []
    bad = 0
    for f in files:
        try:
            with open(f) as h:
                d = json.load(h)
        except Exception as e:
            print("Skipping {}: {}".format(f, e))
            bad += 1
            continue
        d["_source_file"] = os.path.basename(f)
        rows.append(d)
    return rows, bad


def write_outputs(rows, out_csv, out_json, preferred_keys, drop_keys=()):
    json_rows = [{k: v for k, v in r.items() if k not in drop_keys} for r in rows]
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as h:
        json.dump(json_rows, h, indent=2, default=str)

    csv_rows = []
    for r in rows:
        flat = {}
        for k, v in r.items():
            if k in drop_keys:
                continue
            if isinstance(v, (list, dict)):
                flat[k] = json.dumps(v, default=str)[:1000]
            else:
                flat[k] = v
        csv_rows.append(flat)

    keys = []
    seen = set()
    for k in preferred_keys:
        if any(k in r for r in csv_rows):
            keys.append(k)
            seen.add(k)
    for r in csv_rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=keys)
        w.writeheader()
        for r in csv_rows:
            w.writerow(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="./output/prune")
    p.add_argument("--output_csv", default="./output/prune/all_results.csv")
    p.add_argument("--output_json", default="./output/prune/all_results.json")
    p.add_argument("--mode", default="prune", choices=["prune", "glue", "cost"])
    p.add_argument("--include_pruned_heads", action="store_true")
    args = p.parse_args()

    rows, bad = load_dir(args.input_dir)

    if args.mode == "prune":
        def sort_key(r):
            return (
                str(r.get("model", "")), str(r.get("method", "")),
                float(r.get("sparsity_target", 0.0) or 0.0),
                float(r.get("head_z", 0.0) or 0.0),
                str(r.get("layerwise_sparsity", "")),
                str(r.get("wanda_norm", "")),
                str(r.get("eval_dataset", "")),
                int(r.get("seed", 0) or 0),
            )
        rows.sort(key=sort_key)
        preferred = [
            "model", "is_decoder", "is_seq2seq", "method",
            "sparsity_target", "prune_target",
            "sparsity_attn", "sparsity_all_linears",
            "head_z", "alpha", "z",
            "layerwise_sparsity", "wanda_norm",
            "eval_dataset", "calib_dataset", "eval_kind",
            "loss", "ppl", "pseudo_ppl",
            "n_attention_heads", "n_kv_heads", "hidden_size", "head_dim",
            "calib_samples", "eval_samples", "eval_token_count",
            "seed", "dtype", "_source_file",
        ]
        drop = () if args.include_pruned_heads else ("pruned_heads",)
    elif args.mode == "glue":
        def sort_key(r):
            return (
                str(r.get("model", "")), str(r.get("task", "")),
                str(r.get("method", "")),
                float(r.get("sparsity_target", 0.0) or 0.0),
                str(r.get("prune_when", "")),
                int(r.get("seed", 0) or 0),
            )
        rows.sort(key=sort_key)
        preferred = [
            "model", "task", "method", "sparsity_target",
            "prune_target", "prune_when",
            "sparsity_attn", "sparsity_all_linears",
            "head_z", "layerwise_sparsity", "wanda_norm",
            "metric_name", "metrics",
            "epochs", "lr", "bs", "seed", "_source_file",
        ]
        drop = () if args.include_pruned_heads else ("pruned_heads",)
    else:
        def sort_key(r):
            return (
                str(r.get("model", "")), str(r.get("method", "")),
                float(r.get("sparsity_target", 0.0) or 0.0),
                float(r.get("head_z", 0.0) or 0.0),
                int(r.get("seed", 0) or 0),
            )
        rows.sort(key=sort_key)
        preferred = [
            "model", "method", "sparsity_target", "head_z",
            "seq_len", "batch", "iters", "dtype",
            "params_total", "params_nonzero_before", "params_nonzero_after",
            "params_reduction_frac",
            "wallclock_s_before", "wallclock_s_after", "speedup_x",
            "active_heads_before", "active_heads_after",
            "attn_flops_before", "attn_flops_after",
            "seed", "_source_file",
        ]
        drop = ()

    write_outputs(rows, args.output_csv, args.output_json, preferred, drop_keys=drop)
    print("Aggregated {} result file(s) ({} unreadable).".format(len(rows), bad))
    print("  JSON: {}".format(args.output_json))
    print("  CSV:  {}".format(args.output_csv))


if __name__ == "__main__":
    main()
