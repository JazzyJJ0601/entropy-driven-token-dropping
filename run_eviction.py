#!/usr/bin/env python3
"""Real results: KV-cache token eviction on Qwen3-8B.

For each WikiText-2 test passage: run a CTX-token context into the KV cache, evict
tokens from the cache (every layer, original RoPE positions kept), then measure the
model's perplexity on the next CONT tokens. Less memory, same model.

Every method keeps 4 attention-sink tokens and the most recent 25% of the budget;
the rest of the budget is chosen by:
  recent    nothing extra: the recent window simply grows (StreamingLLM)
  random    random older tokens
  h2o       older tokens that received the most attention (Heavy-Hitter Oracle)
  edtd      older tokens that received the most attention from *focused* queries:
            each query's attention is weighted by 1 - its normalised entropy, so
            diffuse "looking everywhere" attention counts little and sharp
            retrieval-style attention counts a lot
Results go to eviction.json. Passages N..N+DEV-1 are a dev split, reported separately.

  python run_eviction.py --start 50 --count 200 --out holdout.json
re-runs everything on a fresh block of passages (no overlap with test/dev) and also
saves each passage's loss so method differences can be checked passage by passage.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

MODEL = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"
CTX, CONT, N, DEV = 512, 64, 40, 10
KEEPS = [0.5, 0.25]
SINKS = 4
OUT = Path(__file__).resolve().parent / "eviction.json"


def select(scores, k, rng, method):
    n = CTX
    recent = max(1, int(k * 0.25)) if method != "recent" else k - SINKS
    keep = set(range(SINKS)) | set(range(n - recent, n))
    pool = [i for i in range(n) if i not in keep]
    budget = k - len(keep)
    if budget > 0:
        if method == "random":
            keep |= set(rng.choice(pool, budget, replace=False).tolist())
        else:
            pool.sort(key=lambda i: -scores[i])
            keep |= set(pool[:budget])
    return torch.tensor(sorted(keep))


@torch.no_grad()
def main():
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, local_files_only=True, device_map="cuda",
        attn_implementation="eager").eval()
    from datasets import load_dataset
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_path = Path(args.out)
    if args.count:
        plan = [(p, "holdout") for p in range(args.start, args.start + args.count)]
    else:
        plan = [(p, "test" if p < N else "dev") for p in range(N + DEV)]
    rng = np.random.default_rng(0)
    totals = {}
    per_passage = []

    for step, (p, split) in enumerate(plan):
        chunk = ids[p * (CTX + CONT):(p + 1) * (CTX + CONT)].cuda()
        ctx, cont = chunk[:CTX], chunk[CTX:]
        cache = DynamicCache()
        out = model(ctx[None], past_key_values=cache, use_cache=True, output_attentions=True)
        last_logits = out.logits[0, -1].float()

        h2o = torch.zeros(CTX, device="cuda")
        edtd = torch.zeros(CTX, device="cuda")
        q_len = torch.arange(1, CTX + 1, device="cuda").float()
        for att in out.attentions:                       # (1, heads, q, k)
            a = att[0].float()
            ent = -(a.clamp_min(1e-12).log() * a).sum(-1)  # (heads, q)
            focus = (1 - ent / q_len.log().clamp_min(1e-6)).clamp(0, 1)
            focus[:, 0] = 0
            h2o += a.sum((0, 1))
            edtd += (a * focus[..., None]).sum((0, 1))
        del out
        scores = {"h2o": h2o.cpu().numpy(), "edtd": edtd.cpu().numpy(), "recent": None, "random": None}

        full_kv = [(l.keys.clone(), l.values.clone()) for l in cache.layers]

        def cont_nll(keep_idx):
            c = DynamicCache()
            for li, (k, v) in enumerate(full_kv):
                c.update(k[:, :, keep_idx], v[:, :, keep_idx], li)
            kept = len(keep_idx)
            o = model(cont[None], past_key_values=c, use_cache=True,
                      position_ids=torch.arange(CTX, CTX + CONT, device="cuda")[None],
                      cache_position=torch.arange(kept, kept + CONT, device="cuda"))
            logits = torch.cat([last_logits[None], o.logits[0, :-1].float()])
            return torch.nn.functional.cross_entropy(logits, cont, reduction="sum").item()

        row = {"passage": p}
        per_passage.append(row)

        def add(name, nll):
            t = totals.setdefault(f"{split}/{name}", [0.0, 0])
            t[0] += nll
            t[1] += CONT
            row[name] = round(nll / CONT, 5)

        add("full", cont_nll(torch.arange(CTX, device="cuda")))
        for keep in KEEPS:
            k = int(CTX * keep)
            for m in ["recent", "random", "h2o", "edtd"]:
                add(f"{m}@{keep}", cont_nll(select(scores[m], k, rng, m).cuda()))
        print(f"passage {step + 1}/{len(plan)}", flush=True)
        counts = {}
        for _, s in plan[:step + 1]:
            counts[s] = counts.get(s, 0) + 1
        res = {"setup": {"model": "Qwen3-8B", "context": CTX, "continuation": CONT,
                         "passages": counts, "sinks": SINKS, "recent_share": 0.25}}
        res.update({k: round(math.exp(v[0] / v[1]), 3) for k, v in totals.items()})
        if args.count:
            res["per_passage_mean_nll"] = per_passage
        out_path.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
