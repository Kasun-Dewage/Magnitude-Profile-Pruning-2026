import os
import gc
import math
import json
import argparse
import random
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from transformers import (
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)
from datasets import load_dataset


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def is_decoder_model(name):
    n = name.lower()
    return ("llama" in n) or ("mistral" in n) or ("gpt" in n) or ("opt" in n) \
        or ("falcon" in n) or ("qwen" in n)


def is_seq2seq_model(name):
    n = name.lower()
    return ("t5" in n) or ("bart" in n) or ("flan" in n)


_ENCODER_LAYER_RE = re.compile(r"encoder\.layer\.(\d+)\.")
_DECODER_LAYER_RE = re.compile(r"(?:model|transformer|gpt_neox|h)\.(?:layers|h)\.(\d+)\.")
_DECODER_LAYER_RE2 = re.compile(r"\.layers?\.(\d+)\.")
_T5_BLOCK_RE = re.compile(r"(?:encoder|decoder)\.block\.(\d+)\.")


def get_linears(model, only_attn=False):
    entries = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        m_enc = _ENCODER_LAYER_RE.search(name)
        m_dec = _DECODER_LAYER_RE.search(name)
        m_dec2 = _DECODER_LAYER_RE2.search(name)
        m_t5 = _T5_BLOCK_RE.search(name)

        role = None
        lidx = None

        if m_enc is not None:
            lidx = int(m_enc.group(1))
            if name.endswith(".query") or name.endswith(".q_proj"):
                role = "Q"
            elif name.endswith(".key") or name.endswith(".k_proj"):
                role = "K"
            elif name.endswith(".value") or name.endswith(".v_proj"):
                role = "V"
            elif "attention.output.dense" in name:
                role = "O"
            elif "intermediate.dense" in name:
                role = "I"
            elif name.endswith("output.dense") and "attention" not in name:
                role = "D"
        elif m_t5 is not None:
            lidx = int(m_t5.group(1))
            if name.endswith(".q"):
                role = "Q"
            elif name.endswith(".k"):
                role = "K"
            elif name.endswith(".v"):
                role = "V"
            elif name.endswith(".o"):
                role = "O"
            elif name.endswith(".wi") or name.endswith(".wi_0") or name.endswith(".wi_1"):
                role = "I"
            elif name.endswith(".wo"):
                role = "D"
        elif m_dec is not None or m_dec2 is not None:
            mm = m_dec if m_dec is not None else m_dec2
            lidx = int(mm.group(1))
            if name.endswith(".q_proj") or name.endswith(".query_key_value"):
                role = "Q"
            elif name.endswith(".k_proj"):
                role = "K"
            elif name.endswith(".v_proj"):
                role = "V"
            elif name.endswith(".o_proj") or name.endswith(".dense") and "attention" in name:
                role = "O"
            elif name.endswith(".gate_proj"):
                role = "G"
            elif name.endswith(".up_proj") or name.endswith(".dense_h_to_4h"):
                role = "U"
            elif name.endswith(".down_proj") or name.endswith(".dense_4h_to_h"):
                role = "D"
            elif name.endswith(".c_attn"):
                role = "Q"
            elif name.endswith(".c_proj"):
                role = "O"
            elif name.endswith(".c_fc"):
                role = "U"
            elif name.endswith(".fc1"):
                role = "U"
            elif name.endswith(".fc2"):
                role = "D"
            elif name.endswith(".out_proj"):
                role = "O"

        if role is None or lidx is None:
            continue
        if only_attn and role not in ("Q", "K", "V", "O"):
            continue
        entries.append((name, module, role, lidx))
    return entries


def group_attn_layers(entries):
    d = defaultdict(dict)
    for name, mod, role, lidx in entries:
        if role in ("Q", "K", "V", "O"):
            d[lidx][role] = (name, mod)
    return dict(d)


def _filter_texts(texts, min_len):
    return [t for t in texts if t and t.strip() and len(t.strip()) >= min_len]


def load_calib_texts(name, n):
    if name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        texts = ds["text"]
    elif name == "c4_small":
        try:
            ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
            texts = []
            for ex in ds:
                texts.append(ex["text"])
                if len(texts) >= n * 4:
                    break
        except Exception as e:
            print("c4 streaming failed ({}); falling back to wikitext2".format(e))
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            texts = ds["text"]
    else:
        raise ValueError("Unknown calib_dataset: {}".format(name))
    texts = _filter_texts(texts, min_len=100)
    return texts[: n * 2]


def load_eval_texts(name):
    if name == "wikitext2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        return _filter_texts(ds["text"], min_len=50)
    elif name == "ptb":
        try:
            ds = load_dataset("ptb_text_only", split="validation")
            texts = ds["sentence"]
        except Exception as e:
            print("ptb load failed ({}); falling back to wikitext2".format(e))
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            texts = ds["text"]
        return _filter_texts(texts, min_len=50)
    elif name == "c4_small":
        try:
            ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
            texts = []
            for i, ex in enumerate(ds):
                texts.append(ex["text"])
                if i >= 2000:
                    break
        except Exception as e:
            print("c4 val streaming failed ({}); falling back to wikitext2".format(e))
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
            texts = ds["text"]
        return _filter_texts(texts, min_len=50)
    else:
        raise ValueError("Unknown eval_dataset: {}".format(name))


def _agg_norm(x, mode):
    if mode == "l2":
        return (x * x).sum(dim=0)
    elif mode == "l1":
        return x.abs().sum(dim=0)
    elif mode == "linf":
        return x.abs().amax(dim=0)
    else:
        raise ValueError("Unknown wanda_norm: {}".format(mode))


def collect_calibration_data(model, tokenizer, texts, entries, device,
                              max_samples=128, max_len=512,
                              need_hessian=True, wanda_norm="l2",
                              is_seq2seq=False):
    accum = {id(m): None for _, m, _, _ in entries}
    cnt = {id(m): 0 for _, m, _, _ in entries}
    Hmap = {}
    if need_hessian:
        for _, m, _, _ in entries:
            Hmap[id(m)] = torch.zeros(m.in_features, m.in_features, dtype=torch.float32)
    hooks = []

    def make_hook(mid, nH):
        def hk(mod, inp, out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu()
            agg = _agg_norm(x, wanda_norm)
            if accum[mid] is None:
                accum[mid] = agg
            else:
                if wanda_norm == "linf":
                    accum[mid] = torch.maximum(accum[mid], agg)
                else:
                    accum[mid] += agg
            cnt[mid] += x.shape[0]
            if nH:
                Hmap[mid] += x.t() @ x
        return hk

    for _, m, _, _ in entries:
        hooks.append(m.register_forward_hook(make_hook(id(m), need_hessian)))

    model.eval()
    seen = 0
    with torch.no_grad():
        for text in texts:
            if seen >= max_samples:
                break
            if not text or not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            if enc.input_ids.shape[1] < 8:
                continue
            try:
                if is_seq2seq:
                    dec_ids = torch.tensor([[tokenizer.pad_token_id or 0]], device=device)
                    model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                          decoder_input_ids=dec_ids)
                else:
                    model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")})
            except Exception as e:
                print("Calib forward failed: {}".format(e))
                continue
            seen += 1

    for h in hooks:
        h.remove()

    norms = {}
    hs = {}
    for _, m, _, _ in entries:
        mid = id(m)
        if cnt[mid] > 0 and accum[mid] is not None:
            if wanda_norm == "l2":
                norms[mid] = (accum[mid] / cnt[mid]).sqrt()
            elif wanda_norm == "l1":
                norms[mid] = accum[mid] / cnt[mid]
            elif wanda_norm == "linf":
                norms[mid] = accum[mid]
            if need_hessian:
                hs[mid] = Hmap[mid] / cnt[mid]
    return norms, hs


def eval_mlm(model, tokenizer, texts, device, n_samples=500, max_len=512,
             mask_prob=0.15, seed=123):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    total_loss, total_n = 0.0, 0
    used = 0
    with torch.no_grad():
        for text in texts:
            if used >= n_samples:
                break
            if not text or not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
            ids = enc.input_ids
            if ids.shape[1] < 10:
                continue
            am = enc.attention_mask.to(device)
            labels = ids.clone()
            probs = torch.full(labels.shape, mask_prob)
            mk = torch.bernoulli(probs, generator=g).bool()
            specials = [tokenizer.get_special_tokens_mask(x, already_has_special_tokens=True)
                        for x in ids.tolist()]
            sp = torch.tensor(specials, dtype=torch.bool)
            mk = mk & ~sp
            if mk.sum() == 0:
                continue
            mi = ids.clone()
            mi[mk] = tokenizer.mask_token_id
            labels[~mk] = -100
            mi = mi.to(device)
            labels = labels.to(device)
            out = model(input_ids=mi, attention_mask=am, labels=labels)
            n = mk.sum().item()
            total_loss += out.loss.item() * n
            total_n += n
            used += 1
    avg = total_loss / max(1, total_n)
    return avg, math.exp(avg), total_n


def eval_clm(model, tokenizer, texts, device, n_samples=200, max_len=2048):
    model.eval()
    sep = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
    if sep is None:
        sep = 0
    all_ids = []
    for text in texts:
        if not text or not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        if not ids:
            continue
        all_ids.extend(ids)
        all_ids.append(sep)
        if len(all_ids) > n_samples * max_len + max_len:
            break

    total_loss, total_n = 0.0, 0
    chunks_done = 0
    with torch.no_grad():
        for start in range(0, len(all_ids), max_len):
            if chunks_done >= n_samples:
                break
            chunk = all_ids[start: start + max_len]
            if len(chunk) < 32:
                continue
            ids = torch.tensor([chunk], dtype=torch.long, device=device)
            labels = ids.clone()
            out = model(input_ids=ids, labels=labels)
            n = ids.shape[1] - 1
            total_loss += out.loss.item() * n
            total_n += n
            chunks_done += 1
    avg = total_loss / max(1, total_n)
    ppl = math.exp(min(avg, 30))
    return avg, ppl, total_n


def eval_seq2seq(model, tokenizer, texts, device, n_samples=200, max_len=512):
    model.eval()
    total_loss, total_n = 0.0, 0
    used = 0
    with torch.no_grad():
        for text in texts:
            if used >= n_samples:
                break
            if not text or not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).to(device)
            if enc.input_ids.shape[1] < 16:
                continue
            half = enc.input_ids.shape[1] // 2
            inp = enc.input_ids[:, :half]
            am = enc.attention_mask[:, :half]
            labels = enc.input_ids[:, half:].clone()
            try:
                out = model(input_ids=inp, attention_mask=am, labels=labels)
            except Exception as e:
                print("Seq2seq eval failed: {}".format(e))
                continue
            n = labels.numel()
            total_loss += out.loss.item() * n
            total_n += n
            used += 1
    avg = total_loss / max(1, total_n)
    ppl = math.exp(min(avg, 30))
    return avg, ppl, total_n


def _row_topk_mask(score, sparsity):
    rows, cols = score.shape
    k = int(sparsity * cols)
    if k <= 0:
        return torch.ones_like(score, dtype=torch.bool)
    sorted_s, _ = torch.sort(score, dim=1)
    thresh = sorted_s[:, k - 1: k]
    return score > thresh


def random_mask(W, sparsity, seed=0):
    if sparsity <= 0:
        return torch.ones_like(W, dtype=torch.bool, device=W.device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = torch.rand(W.shape, generator=g)
    return _row_topk_mask(r, sparsity).to(W.device)


def magnitude_mask(W, sparsity):
    if sparsity <= 0:
        return torch.ones_like(W, dtype=torch.bool, device=W.device)
    absW = W.detach().float().abs().cpu()
    return _row_topk_mask(absW, sparsity).to(W.device)


def wanda_mask(W, act_norm, sparsity):
    if sparsity <= 0:
        return torch.ones_like(W, dtype=torch.bool, device=W.device)
    absW = W.detach().float().abs().cpu()
    score = absW * act_norm.cpu().view(1, -1)
    return _row_topk_mask(score, sparsity).to(W.device)


def wanda_mp_mask(W, act_norm, sparsity, alpha=2.0, z=2.0):
    if sparsity <= 0:
        return torch.ones_like(W, dtype=torch.bool, device=W.device)
    Wf = W.detach().float().cpu()
    cn = Wf.norm(dim=0)
    rn = Wf.norm(dim=1)
    mu_c, sd_c = cn.mean(), cn.std() + 1e-12
    mu_r, sd_r = rn.mean(), rn.std() + 1e-12
    c_boost = 1.0 + alpha * torch.clamp((cn - mu_c) / sd_c - z, min=0.0)
    r_boost = 1.0 + alpha * torch.clamp((rn - mu_r) / sd_r - z, min=0.0)
    score = (Wf.abs() * act_norm.cpu().view(1, -1)
             * c_boost.view(1, -1) * r_boost.view(-1, 1))
    return _row_topk_mask(score, sparsity).to(W.device)


def sparsegpt_prune_tensor(W, H, sparsity, blocksize=128, damp_frac=0.01):
    dev = torch.device("cpu")
    W = W.detach().float().clone().to(dev)
    H = H.detach().float().clone().to(dev)
    rows, cols = W.shape
    dead = torch.diag(H) == 0
    H[dead, dead] = 1.0
    W[:, dead] = 0.0
    damp = damp_frac * torch.mean(torch.diag(H))
    idx = torch.arange(cols, device=dev)
    H[idx, idx] += damp
    try:
        L = torch.linalg.cholesky(H)
    except Exception:
        H[idx, idx] += damp * 10.0
        L = torch.linalg.cholesky(H)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    mask = torch.ones_like(W, dtype=torch.bool)
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        cnt = i2 - i1
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        Err1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        diag1 = torch.diag(Hinv1).view(1, -1)
        tmp = (W1 * W1) / (diag1 * diag1 + 1e-12)
        k = int(sparsity * tmp.numel())
        if k > 0:
            thresh = torch.kthvalue(tmp.flatten(), k).values
            mask1 = tmp > thresh
        else:
            mask1 = torch.ones_like(tmp, dtype=torch.bool)
        for j in range(cnt):
            w = W1[:, j]
            d = Hinv1[j, j]
            q = w.clone()
            q[~mask1[:, j]] = 0.0
            Q1[:, j] = q
            err = (w - q) / d
            if j + 1 < cnt:
                W1[:, j + 1:] -= err.unsqueeze(1) * Hinv1[j, j + 1:].unsqueeze(0)
            Err1[:, j] = err
        W[:, i1:i2] = Q1
        mask[:, i1:i2] = mask1
        if i2 < cols:
            W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]
    return W, mask


def apply_tensor(module, new_W):
    with torch.no_grad():
        module.weight.data.copy_(new_W.to(module.weight.device, dtype=module.weight.dtype))


def apply_mask(module, mask):
    with torch.no_grad():
        m = mask.to(module.weight.device, dtype=module.weight.dtype)
        module.weight.data.mul_(m)


def head_scores_magnitude(layer_mods, n_heads, head_dim, n_kv_heads=None):
    if n_kv_heads is None:
        n_kv_heads = n_heads
    Q = layer_mods["Q"][1].weight.detach().float().cpu()
    K = layer_mods["K"][1].weight.detach().float().cpu()
    V = layer_mods["V"][1].weight.detach().float().cpu()
    O = layer_mods["O"][1].weight.detach().float().cpu()
    group = max(1, n_heads // max(1, n_kv_heads))
    scores = []
    for h in range(n_heads):
        s = 0.0
        s += Q[h * head_dim:(h + 1) * head_dim].pow(2).sum().item()
        kv_h = h // group
        s += K[kv_h * head_dim:(kv_h + 1) * head_dim].pow(2).sum().item()
        s += V[kv_h * head_dim:(kv_h + 1) * head_dim].pow(2).sum().item()
        s += O[:, h * head_dim:(h + 1) * head_dim].pow(2).sum().item()
        scores.append(math.sqrt(s))
    return scores


def head_scores_mp(layer_mods, n_heads, head_dim, z=2.0, n_kv_heads=None):
    if n_kv_heads is None:
        n_kv_heads = n_heads
    group = max(1, n_heads // max(1, n_kv_heads))
    scores = [0.0] * n_heads

    def _exc_rows(W):
        rn = W.norm(dim=1)
        mu, sd = rn.mean().item(), rn.std().item() + 1e-12
        return torch.clamp(rn - (mu + z * sd), min=0.0)

    Wq = layer_mods["Q"][1].weight.detach().float().cpu()
    exc_q = _exc_rows(Wq)
    Wk = layer_mods["K"][1].weight.detach().float().cpu()
    exc_k = _exc_rows(Wk)
    Wv = layer_mods["V"][1].weight.detach().float().cpu()
    exc_v = _exc_rows(Wv)

    Wo = layer_mods["O"][1].weight.detach().float().cpu()
    cn = Wo.norm(dim=0)
    mu_o, sd_o = cn.mean().item(), cn.std().item() + 1e-12
    exc_o = torch.clamp(cn - (mu_o + z * sd_o), min=0.0)

    for h in range(n_heads):
        kv_h = h // group
        scores[h] += exc_q[h * head_dim:(h + 1) * head_dim].sum().item()
        scores[h] += exc_k[kv_h * head_dim:(kv_h + 1) * head_dim].sum().item()
        scores[h] += exc_v[kv_h * head_dim:(kv_h + 1) * head_dim].sum().item()
        scores[h] += exc_o[h * head_dim:(h + 1) * head_dim].sum().item()
    return scores


def head_scores_wanda(layer_mods, n_heads, head_dim, act_norms=None, n_kv_heads=None):
    if n_kv_heads is None:
        n_kv_heads = n_heads
    group = max(1, n_heads // max(1, n_kv_heads))
    Wq = layer_mods["Q"][1].weight.detach().float().cpu()
    Wk = layer_mods["K"][1].weight.detach().float().cpu()
    Wv = layer_mods["V"][1].weight.detach().float().cpu()
    Wo = layer_mods["O"][1].weight.detach().float().cpu()

    if act_norms is None:
        a_in = torch.ones(Wq.shape[1])
        a_o = torch.ones(Wo.shape[1])
    else:
        a_in = act_norms.get("attn_in", torch.ones(Wq.shape[1])).cpu()
        a_o = act_norms.get("o_in", torch.ones(Wo.shape[1])).cpu()

    sq = (Wq.abs() * a_in.view(1, -1)).sum(dim=1)
    sk = (Wk.abs() * a_in.view(1, -1)).sum(dim=1)
    sv = (Wv.abs() * a_in.view(1, -1)).sum(dim=1)
    so = (Wo.abs() * a_o.view(1, -1)).sum(dim=0)

    scores = []
    for h in range(n_heads):
        kv_h = h // group
        s = 0.0
        s += sq[h * head_dim:(h + 1) * head_dim].sum().item()
        s += sk[kv_h * head_dim:(kv_h + 1) * head_dim].sum().item()
        s += sv[kv_h * head_dim:(kv_h + 1) * head_dim].sum().item()
        s += so[h * head_dim:(h + 1) * head_dim].sum().item()
        scores.append(s)
    return scores


def head_scores_sparsegpt(layer_mods, n_heads, head_dim, hessians=None, n_kv_heads=None):
    if n_kv_heads is None:
        n_kv_heads = n_heads
    group = max(1, n_heads // max(1, n_kv_heads))
    Wq = layer_mods["Q"][1].weight.detach().float().cpu()
    Wk = layer_mods["K"][1].weight.detach().float().cpu()
    Wv = layer_mods["V"][1].weight.detach().float().cpu()
    Wo = layer_mods["O"][1].weight.detach().float().cpu()

    if hessians is None:
        d_in = torch.ones(Wq.shape[1])
        d_o = torch.ones(Wo.shape[1])
    else:
        Hin = hessians.get("attn_in")
        Ho = hessians.get("o_in")
        d_in = torch.diag(Hin).clamp(min=1e-8) if Hin is not None else torch.ones(Wq.shape[1])
        d_o = torch.diag(Ho).clamp(min=1e-8) if Ho is not None else torch.ones(Wo.shape[1])

    sq = ((Wq * Wq) * d_in.view(1, -1)).sum(dim=1)
    sk = ((Wk * Wk) * d_in.view(1, -1)).sum(dim=1)
    sv = ((Wv * Wv) * d_in.view(1, -1)).sum(dim=1)
    so = ((Wo * Wo) * d_o.view(1, -1)).sum(dim=0)

    scores = []
    for h in range(n_heads):
        kv_h = h // group
        s = 0.0
        s += sq[h * head_dim:(h + 1) * head_dim].sum().item()
        s += sk[kv_h * head_dim:(kv_h + 1) * head_dim].sum().item()
        s += sv[kv_h * head_dim:(kv_h + 1) * head_dim].sum().item()
        s += so[h * head_dim:(h + 1) * head_dim].sum().item()
        scores.append(s)
    return scores


def head_scores_grad(layer_mods, n_heads, head_dim, grads=None, n_kv_heads=None):
    if n_kv_heads is None:
        n_kv_heads = n_heads
    group = max(1, n_heads // max(1, n_kv_heads))
    Wq = layer_mods["Q"][1].weight.detach().float().cpu()
    Wk = layer_mods["K"][1].weight.detach().float().cpu()
    Wv = layer_mods["V"][1].weight.detach().float().cpu()
    Wo = layer_mods["O"][1].weight.detach().float().cpu()

    if grads is None:
        Gq = torch.zeros_like(Wq)
        Gk = torch.zeros_like(Wk)
        Gv = torch.zeros_like(Wv)
        Go = torch.zeros_like(Wo)
    else:
        Gq = grads.get("Q", torch.zeros_like(Wq))
        Gk = grads.get("K", torch.zeros_like(Wk))
        Gv = grads.get("V", torch.zeros_like(Wv))
        Go = grads.get("O", torch.zeros_like(Wo))

    tq = (Gq * Wq).abs().sum(dim=1)
    tk = (Gk * Wk).abs().sum(dim=1)
    tv = (Gv * Wv).abs().sum(dim=1)
    to_ = (Go * Wo).abs().sum(dim=0)

    scores = []
    for h in range(n_heads):
        kv_h = h // group
        s = 0.0
        s += tq[h * head_dim:(h + 1) * head_dim].sum().item()
        s += tk[kv_h * head_dim:(kv_h + 1) * head_dim].sum().item()
        s += tv[kv_h * head_dim:(kv_h + 1) * head_dim].sum().item()
        s += to_[h * head_dim:(h + 1) * head_dim].sum().item()
        scores.append(s)
    return scores


def head_scores_mp_g(layer_mods, n_heads, head_dim, z=2.0, n_kv_heads=None,
                     query_weight=1.0, kv_weight=1.0):
    if n_kv_heads is None:
        n_kv_heads = n_heads
    group = max(1, n_heads // max(1, n_kv_heads))

    def _exc_rows(W):
        rn = W.norm(dim=1)
        mu, sd = rn.mean().item(), rn.std().item() + 1e-12
        return torch.clamp(rn - (mu + z * sd), min=0.0)

    Wq = layer_mods["Q"][1].weight.detach().float().cpu()
    exc_q = _exc_rows(Wq)
    Wk = layer_mods["K"][1].weight.detach().float().cpu()
    exc_k = _exc_rows(Wk)
    Wv = layer_mods["V"][1].weight.detach().float().cpu()
    exc_v = _exc_rows(Wv)
    Wo = layer_mods["O"][1].weight.detach().float().cpu()
    cn = Wo.norm(dim=0)
    mu_o, sd_o = cn.mean().item(), cn.std().item() + 1e-12
    exc_o = torch.clamp(cn - (mu_o + z * sd_o), min=0.0)

    kv_group_score = [0.0] * n_kv_heads
    for kv_h in range(n_kv_heads):
        kv_group_score[kv_h] += exc_k[kv_h * head_dim:(kv_h + 1) * head_dim].sum().item()
        kv_group_score[kv_h] += exc_v[kv_h * head_dim:(kv_h + 1) * head_dim].sum().item()

    scores = [0.0] * n_heads
    for h in range(n_heads):
        kv_h = h // group
        q_part = exc_q[h * head_dim:(h + 1) * head_dim].sum().item()
        o_part = exc_o[h * head_dim:(h + 1) * head_dim].sum().item()
        kv_part = kv_group_score[kv_h] / max(1, group)
        scores[h] = query_weight * (q_part + o_part) + kv_weight * kv_part
    return scores


def head_scores_hybrid(layer_mods, n_heads, head_dim, act_norms=None, z=2.0,
                       n_kv_heads=None, lam_mag=1.0, lam_mp=1.0, lam_wanda=1.0):
    s_mag = head_scores_magnitude(layer_mods, n_heads, head_dim, n_kv_heads=n_kv_heads)
    s_mp = head_scores_mp(layer_mods, n_heads, head_dim, z=z, n_kv_heads=n_kv_heads)
    s_w = head_scores_wanda(layer_mods, n_heads, head_dim, act_norms=act_norms,
                            n_kv_heads=n_kv_heads)

    def _norm(s):
        t = torch.tensor(s, dtype=torch.float32)
        mn, mx = t.min().item(), t.max().item()
        rng = mx - mn + 1e-12
        return [(v - mn) / rng for v in s]

    s_mag_n = _norm(s_mag)
    s_mp_n = _norm(s_mp)
    s_w_n = _norm(s_w)
    return [lam_mag * a + lam_mp * b + lam_wanda * c
            for a, b, c in zip(s_mag_n, s_mp_n, s_w_n)]


def collect_attn_act_norms(model, tokenizer, texts, layers_dict, device,
                           max_samples=64, max_len=512, wanda_norm="l2",
                           is_seq2seq=False):
    accum = {}
    cnt = {}
    for lidx, mods in layers_dict.items():
        for role in ("Q", "K", "V", "O"):
            if role in mods:
                mid = id(mods[role][1])
                accum[mid] = None
                cnt[mid] = 0
    hooks = []

    def make_hook(mid):
        def hk(mod, inp, out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu()
            agg = _agg_norm(x, wanda_norm)
            if accum[mid] is None:
                accum[mid] = agg
            else:
                if wanda_norm == "linf":
                    accum[mid] = torch.maximum(accum[mid], agg)
                else:
                    accum[mid] += agg
            cnt[mid] += x.shape[0]
        return hk

    for lidx, mods in layers_dict.items():
        for role in ("Q", "K", "V", "O"):
            if role in mods:
                m = mods[role][1]
                hooks.append(m.register_forward_hook(make_hook(id(m))))

    model.eval()
    seen = 0
    with torch.no_grad():
        for text in texts:
            if seen >= max_samples:
                break
            if not text or not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_len).to(device)
            if enc.input_ids.shape[1] < 8:
                continue
            try:
                if is_seq2seq:
                    dec_ids = torch.tensor([[tokenizer.pad_token_id or 0]], device=device)
                    model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                          decoder_input_ids=dec_ids)
                else:
                    model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")})
            except Exception as e:
                print("Calib forward failed: {}".format(e))
                continue
            seen += 1

    for h in hooks:
        h.remove()

    layer_norms = {}
    for lidx, mods in layers_dict.items():
        d = {}
        if "Q" in mods:
            mid = id(mods["Q"][1])
            if cnt[mid] > 0 and accum[mid] is not None:
                if wanda_norm == "l2":
                    d["attn_in"] = (accum[mid] / cnt[mid]).sqrt()
                elif wanda_norm == "l1":
                    d["attn_in"] = accum[mid] / cnt[mid]
                else:
                    d["attn_in"] = accum[mid]
        if "O" in mods:
            mid = id(mods["O"][1])
            if cnt[mid] > 0 and accum[mid] is not None:
                if wanda_norm == "l2":
                    d["o_in"] = (accum[mid] / cnt[mid]).sqrt()
                elif wanda_norm == "l1":
                    d["o_in"] = accum[mid] / cnt[mid]
                else:
                    d["o_in"] = accum[mid]
        layer_norms[lidx] = d
    return layer_norms


def collect_attn_hessians(model, tokenizer, texts, layers_dict, device,
                          max_samples=64, max_len=512, is_seq2seq=False):
    Hmap = {}
    cnt = {}
    for lidx, mods in layers_dict.items():
        for role in ("Q", "O"):
            if role in mods:
                m = mods[role][1]
                mid = id(m)
                Hmap[mid] = torch.zeros(m.in_features, m.in_features, dtype=torch.float32)
                cnt[mid] = 0
    hooks = []

    def make_hook(mid):
        def hk(mod, inp, out):
            x = inp[0].detach().float().reshape(-1, inp[0].shape[-1]).cpu()
            Hmap[mid] += x.t() @ x
            cnt[mid] += x.shape[0]
        return hk

    for lidx, mods in layers_dict.items():
        for role in ("Q", "O"):
            if role in mods:
                m = mods[role][1]
                hooks.append(m.register_forward_hook(make_hook(id(m))))

    model.eval()
    seen = 0
    with torch.no_grad():
        for text in texts:
            if seen >= max_samples:
                break
            if not text or not text.strip():
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True,
                            max_length=max_len).to(device)
            if enc.input_ids.shape[1] < 8:
                continue
            try:
                if is_seq2seq:
                    dec_ids = torch.tensor([[tokenizer.pad_token_id or 0]], device=device)
                    model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                          decoder_input_ids=dec_ids)
                else:
                    model(**{k: v for k, v in enc.items() if k in ("input_ids", "attention_mask")})
            except Exception as e:
                print("Hessian forward failed: {}".format(e))
                continue
            seen += 1

    for h in hooks:
        h.remove()

    layer_hessians = {}
    for lidx, mods in layers_dict.items():
        d = {}
        if "Q" in mods:
            mid = id(mods["Q"][1])
            if cnt[mid] > 0:
                d["attn_in"] = Hmap[mid] / cnt[mid]
        if "O" in mods:
            mid = id(mods["O"][1])
            if cnt[mid] > 0:
                d["o_in"] = Hmap[mid] / cnt[mid]
        layer_hessians[lidx] = d
    return layer_hessians


def collect_attn_grads(model, tokenizer, texts, layers_dict, device,
                       max_samples=32, max_len=512, is_seq2seq=False, is_decoder=False):
    target_params = {}
    for lidx, mods in layers_dict.items():
        for role in ("Q", "K", "V", "O"):
            if role in mods:
                m = mods[role][1]
                target_params[(lidx, role)] = m.weight

    for p in model.parameters():
        p.requires_grad_(False)
    for w in target_params.values():
        w.requires_grad_(True)

    grad_accum = {key: torch.zeros_like(w.detach(), device="cpu", dtype=torch.float32)
                  for key, w in target_params.items()}
    cnt = 0

    model.eval()
    seen = 0
    for text in texts:
        if seen >= max_samples:
            break
        if not text or not text.strip():
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=max_len).to(device)
        if enc.input_ids.shape[1] < 16:
            continue
        try:
            model.zero_grad(set_to_none=True)
            if is_seq2seq:
                half = enc.input_ids.shape[1] // 2
                inp = enc.input_ids[:, :half]
                am = enc.attention_mask[:, :half]
                labels = enc.input_ids[:, half:].clone()
                out = model(input_ids=inp, attention_mask=am, labels=labels)
            elif is_decoder:
                labels = enc.input_ids.clone()
                out = model(input_ids=enc.input_ids, labels=labels)
            else:
                ids = enc.input_ids.clone()
                labels = ids.clone()
                probs = torch.full(labels.shape, 0.15)
                mk = torch.bernoulli(probs).bool().to(device)
                if mk.sum() == 0:
                    continue
                ids[mk] = tokenizer.mask_token_id if tokenizer.mask_token_id is not None else 0
                labels[~mk] = -100
                out = model(input_ids=ids, attention_mask=enc.attention_mask, labels=labels)
            loss = out.loss
            if loss is None or not torch.isfinite(loss):
                continue
            loss.backward()
            with torch.no_grad():
                for key, w in target_params.items():
                    if w.grad is not None:
                        grad_accum[key] += w.grad.detach().float().cpu()
            cnt += 1
            seen += 1
        except Exception as e:
            print("Grad collection failed: {}".format(e))
            continue

    model.zero_grad(set_to_none=True)
    for w in target_params.values():
        w.requires_grad_(False)

    layer_grads = defaultdict(dict)
    if cnt > 0:
        for (lidx, role), g in grad_accum.items():
            layer_grads[lidx][role] = g / cnt
    return dict(layer_grads)


def prune_heads_globally(layers_dict, n_heads, head_dim, sparsity, scorer, n_kv_heads=None):
    if n_kv_heads is None:
        n_kv_heads = n_heads
    group = max(1, n_heads // max(1, n_kv_heads))

    all_scores = []
    for lidx, mods in layers_dict.items():
        s = scorer(mods, n_heads, head_dim, n_kv_heads=n_kv_heads)
        for h, v in enumerate(s):
            all_scores.append((v, lidx, h))
    total = len(all_scores)
    k = int(sparsity * total)
    if k <= 0:
        return []
    all_scores.sort(key=lambda x: x[0])
    pruned = all_scores[:k]
    pruned_set = set((lidx, h) for _, lidx, h in pruned)

    for _, lidx, h in pruned:
        mods = layers_dict[lidx]
        Wq = mods["Q"][1].weight.data
        Wq[h * head_dim:(h + 1) * head_dim, :] = 0.0
        Wo = mods["O"][1].weight.data
        Wo[:, h * head_dim:(h + 1) * head_dim] = 0.0

    for lidx, mods in layers_dict.items():
        for kv_h in range(n_kv_heads):
            qs = [kv_h * group + g for g in range(group)]
            if all((lidx, q) in pruned_set for q in qs):
                Wk = mods["K"][1].weight.data
                Wv = mods["V"][1].weight.data
                Wk[kv_h * head_dim:(kv_h + 1) * head_dim, :] = 0.0
                Wv[kv_h * head_dim:(kv_h + 1) * head_dim, :] = 0.0

    return [(int(lidx), int(h)) for _, lidx, h in pruned]


def per_layer_sparsity(target, n_layers, mode="uniform"):
    if mode == "uniform" or n_layers <= 1:
        return [target] * n_layers
    if mode == "linear_inc":
        return [max(0.0, min(0.95, target * (0.5 + i / (n_layers - 1))))
                for i in range(n_layers)]
    if mode == "linear_dec":
        return [max(0.0, min(0.95, target * (1.5 - i / (n_layers - 1))))
                for i in range(n_layers)]
    raise ValueError("Unknown layerwise_sparsity mode: {}".format(mode))


def compute_sparsity(entries):
    total, zero = 0, 0
    for _, m, _, _ in entries:
        w = m.weight.data
        total += w.numel()
        zero += (w == 0).sum().item()
    return zero / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="roberta-base")
    parser.add_argument("--method", default="magnitude",
                        choices=["dense", "random", "magnitude", "wanda",
                                 "sparsegpt", "wanda_mp", "mag_head", "mp_head",
                                 "wanda_head", "sparsegpt_head", "grad_head",
                                 "mp_head_g", "hybrid_head"])
    parser.add_argument("--head_calib_samples", type=int, default=64)
    parser.add_argument("--head_grad_samples", type=int, default=32)
    parser.add_argument("--hybrid_lam_mag", type=float, default=1.0)
    parser.add_argument("--hybrid_lam_mp", type=float, default=1.0)
    parser.add_argument("--hybrid_lam_wanda", type=float, default=1.0)
    parser.add_argument("--mpg_query_weight", type=float, default=1.0)
    parser.add_argument("--mpg_kv_weight", type=float, default=1.0)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--prune_target", default="all", choices=["all", "attn"])
    parser.add_argument("--calib_samples", type=int, default=128)
    parser.add_argument("--eval_samples", type=int, default=500)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=2.0)
    parser.add_argument("--z", type=float, default=2.0)
    parser.add_argument("--head_z", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default="./output/prune")
    parser.add_argument("--dtype", default="float32",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--layerwise_sparsity", default="uniform",
                        choices=["uniform", "linear_inc", "linear_dec"])
    parser.add_argument("--wanda_norm", default="l2",
                        choices=["l2", "l1", "linf"])
    parser.add_argument("--eval_dataset", default="wikitext2",
                        choices=["wikitext2", "ptb", "c4_small"])
    parser.add_argument("--calib_dataset", default="wikitext2",
                        choices=["wikitext2", "c4_small"])
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    args = parser.parse_args()

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
    print("Model: {} | Method: {} | Sparsity: {:.3f} | head_z: {:.2f} | "
          "Target: {} | Decoder: {} | Seq2Seq: {}".format(
              args.model, args.method, args.sparsity, args.head_z,
              args.prune_target, decoder, seq2seq))
    print("Layerwise: {} | Wanda-norm: {} | Eval: {} | Calib: {}".format(
        args.layerwise_sparsity, args.wanda_norm,
        args.eval_dataset, args.calib_dataset))

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

    cfg = model.config
    n_heads = getattr(cfg, "num_attention_heads", None)
    n_kv_heads = getattr(cfg, "num_key_value_heads", n_heads) or n_heads
    hidden = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model", None)
    head_dim = getattr(cfg, "head_dim", None) or (hidden // n_heads if hidden and n_heads else None)

    is_head_method = args.method in ("mag_head", "mp_head",
                                       "wanda_head", "sparsegpt_head",
                                       "grad_head", "mp_head_g", "hybrid_head")
    only_attn = (args.prune_target == "attn") or is_head_method

    if is_head_method:
        attn_entries = get_linears(model, only_attn=True)
        layers = group_attn_layers(attn_entries)
        layers = {k: v for k, v in layers.items() if all(r in v for r in "QKVO")}
        entries_for_sparsity_report = attn_entries
    else:
        entries = get_linears(model, only_attn=only_attn)
        entries_for_sparsity_report = entries

    print("Discovered {} prunable linears (only_attn={})".format(
        len(entries_for_sparsity_report), only_attn))

    layer_ids = sorted(set(l for _, _, _, l in entries_for_sparsity_report))
    sched = per_layer_sparsity(args.sparsity, len(layer_ids), mode=args.layerwise_sparsity)
    layer_target = {l: sched[i] for i, l in enumerate(layer_ids)}

    pruned_info = {}

    if args.method == "dense":
        pass
    elif args.method == "random":
        role_offset = {"Q": 1, "K": 2, "V": 3, "O": 4, "I": 5, "D": 6, "G": 7, "U": 8}
        for name, m, role, lidx in entries:
            sp = layer_target[lidx]
            mk = random_mask(m.weight, sp,
                             seed=args.seed + 1000 * lidx + role_offset.get(role, 0))
            apply_mask(m, mk)
    elif args.method == "magnitude":
        for name, m, role, lidx in entries:
            sp = layer_target[lidx]
            mk = magnitude_mask(m.weight, sp)
            apply_mask(m, mk)
    elif args.method == "wanda":
        print("Collecting Wanda calibration activations ({})".format(args.wanda_norm))
        texts = load_calib_texts(args.calib_dataset, args.calib_samples)
        norms, _ = collect_calibration_data(
            model, tokenizer, texts, entries, device,
            max_samples=args.calib_samples, max_len=args.max_len,
            need_hessian=False, wanda_norm=args.wanda_norm, is_seq2seq=seq2seq,
        )
        for name, m, role, lidx in entries:
            an = norms.get(id(m))
            if an is None:
                continue
            sp = layer_target[lidx]
            mk = wanda_mask(m.weight, an, sp)
            apply_mask(m, mk)
        del norms
    elif args.method == "wanda_mp":
        print("Collecting Wanda-MP calibration activations ({})".format(args.wanda_norm))
        texts = load_calib_texts(args.calib_dataset, args.calib_samples)
        norms, _ = collect_calibration_data(
            model, tokenizer, texts, entries, device,
            max_samples=args.calib_samples, max_len=args.max_len,
            need_hessian=False, wanda_norm=args.wanda_norm, is_seq2seq=seq2seq,
        )
        for name, m, role, lidx in entries:
            an = norms.get(id(m))
            if an is None:
                continue
            sp = layer_target[lidx]
            mk = wanda_mp_mask(m.weight, an, sp, alpha=args.alpha, z=args.z)
            apply_mask(m, mk)
        del norms
    elif args.method == "sparsegpt":
        print("Collecting SparseGPT calibration Hessians")
        texts = load_calib_texts(args.calib_dataset, args.calib_samples)
        _, hs = collect_calibration_data(
            model, tokenizer, texts, entries, device,
            max_samples=args.calib_samples, max_len=args.max_len,
            need_hessian=True, wanda_norm="l2", is_seq2seq=seq2seq,
        )
        for name, m, role, lidx in entries:
            H = hs.get(id(m))
            if H is None:
                continue
            sp = layer_target[lidx]
            new_W, _ = sparsegpt_prune_tensor(m.weight, H, sp)
            apply_tensor(m, new_W)
            del H
            hs[id(m)] = None
            gc.collect()
        del hs
    elif args.method == "mag_head":
        scorer = lambda lm, n, hd, n_kv_heads=n_kv_heads: \
            head_scores_magnitude(lm, n, hd, n_kv_heads=n_kv_heads)
        pruned = prune_heads_globally(layers, n_heads, head_dim, args.sparsity, scorer,
                                       n_kv_heads=n_kv_heads)
        pruned_info["pruned_heads"] = pruned
    elif args.method == "mp_head":
        scorer = lambda lm, n, hd, n_kv_heads=n_kv_heads: \
            head_scores_mp(lm, n, hd, z=args.head_z, n_kv_heads=n_kv_heads)
        pruned = prune_heads_globally(layers, n_heads, head_dim, args.sparsity, scorer,
                                       n_kv_heads=n_kv_heads)
        pruned_info["pruned_heads"] = pruned
    elif args.method == "wanda_head":
        print("Collecting attention activation norms for Wanda-Head")
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
        pruned = prune_heads_globally(layers, n_heads, head_dim, args.sparsity, scorer,
                                       n_kv_heads=n_kv_heads)
        pruned_info["pruned_heads"] = pruned
    elif args.method == "sparsegpt_head":
        print("Collecting attention Hessian diagonals for SparseGPT-Head")
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
        pruned = prune_heads_globally(layers, n_heads, head_dim, args.sparsity, scorer,
                                       n_kv_heads=n_kv_heads)
        pruned_info["pruned_heads"] = pruned
    elif args.method == "grad_head":
        print("Collecting first-order gradients for Grad-Head")
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
        pruned = prune_heads_globally(layers, n_heads, head_dim, args.sparsity, scorer,
                                       n_kv_heads=n_kv_heads)
        pruned_info["pruned_heads"] = pruned
    elif args.method == "mp_head_g":
        scorer = lambda lm, n, hd, n_kv_heads=n_kv_heads: \
            head_scores_mp_g(lm, n, hd, z=args.head_z, n_kv_heads=n_kv_heads,
                             query_weight=args.mpg_query_weight,
                             kv_weight=args.mpg_kv_weight)
        pruned = prune_heads_globally(layers, n_heads, head_dim, args.sparsity, scorer,
                                       n_kv_heads=n_kv_heads)
        pruned_info["pruned_heads"] = pruned
    elif args.method == "hybrid_head":
        print("Collecting calibration norms for Hybrid-Head")
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
        pruned = prune_heads_globally(layers, n_heads, head_dim, args.sparsity, scorer,
                                       n_kv_heads=n_kv_heads)
        pruned_info["pruned_heads"] = pruned

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    attn_only_entries = get_linears(model, only_attn=True)
    all_entries = get_linears(model, only_attn=False)
    sp_attn = compute_sparsity(attn_only_entries)
    sp_all = compute_sparsity(all_entries)
    print("Sparsity (attn): {:.4f} | (all linears): {:.4f}".format(sp_attn, sp_all))

    print("Evaluating on {}".format(args.eval_dataset))
    eval_texts = load_eval_texts(args.eval_dataset)
    if seq2seq:
        avg_loss, ppl, tokens_used = eval_seq2seq(
            model, tokenizer, eval_texts, device,
            n_samples=args.eval_samples, max_len=args.max_len,
        )
        eval_kind = "seq2seq_ppl"
    elif decoder:
        clm_max = args.max_len if args.max_len > 512 else 1024
        avg_loss, ppl, tokens_used = eval_clm(
            model, tokenizer, eval_texts, device,
            n_samples=args.eval_samples, max_len=clm_max,
        )
        eval_kind = "clm_ppl"
    else:
        avg_loss, ppl, tokens_used = eval_mlm(
            model, tokenizer, eval_texts, device,
            n_samples=args.eval_samples, max_len=args.max_len, seed=123,
        )
        eval_kind = "mlm_pseudo_ppl"

    result = {
        "model": args.model,
        "is_decoder": decoder,
        "is_seq2seq": seq2seq,
        "method": args.method,
        "sparsity_target": args.sparsity,
        "prune_target": args.prune_target,
        "sparsity_attn": sp_attn,
        "sparsity_all_linears": sp_all,
        "n_attention_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "hidden_size": hidden,
        "head_dim": head_dim,
        "alpha": args.alpha,
        "z": args.z,
        "head_z": args.head_z,
        "layerwise_sparsity": args.layerwise_sparsity,
        "wanda_norm": args.wanda_norm,
        "eval_dataset": args.eval_dataset,
        "calib_dataset": args.calib_dataset,
        "calib_samples": args.calib_samples,
        "eval_samples": args.eval_samples,
        "eval_token_count": tokens_used,
        "eval_kind": eval_kind,
        "loss": avg_loss,
        "ppl": ppl,
        "pseudo_ppl": ppl if (not decoder and not seq2seq) else None,
        "seed": args.seed,
        "dtype": args.dtype,
    }
    result.update(pruned_info)

    print("RESULT")
    print(json.dumps({k: v for k, v in result.items() if k != "pruned_heads"}, indent=2))

    safe_model = args.model.replace("/", "_")
    if is_head_method:
        tag = "{}_{}_{}_s{:.3f}_hz{:.1f}_{}_seed{}".format(
            safe_model, args.prune_target, args.method, args.sparsity,
            args.head_z, args.eval_dataset, args.seed,
        )
    else:
        tag = "{}_{}_{}_s{:.3f}_a{:.1f}_z{:.1f}_{}_{}_{}_seed{}".format(
            safe_model, args.prune_target, args.method, args.sparsity,
            args.alpha, args.z, args.layerwise_sparsity,
            args.wanda_norm, args.eval_dataset, args.seed,
        )
    out_path = os.path.join(args.output_dir, "{}.json".format(tag))
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to {}".format(out_path))


if __name__ == "__main__":
    main()
