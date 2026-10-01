# Entropy-Driven Token Dropping (EDTD)

When a language model's KV cache has to shrink, which old tokens should stay? The standard answer
(H2O, "Heavy-Hitter Oracle") keeps the tokens that have received the most attention. EDTD changes one
thing: **each query's attention is weighted by how focused it is.** A query that spreads its attention
over everything (high entropy) says little about which tokens matter; a query that locks onto a few
tokens (low entropy) is doing retrieval, and those are the tokens worth keeping.

```
focus(q)   = 1 - H(attention of q) / log(number of tokens q can see)
score(tok) = sum over layers, heads, queries of  attention(q -> tok) * focus(q)
```

H2O is the same sum without `focus(q)`.

## Result (Qwen3-8B, real KV-cache eviction)

The 512-token context is run into the cache. Then tokens are evicted from every layer's cache
(original RoPE positions kept). Finally, perplexity is measured on the next 64 tokens. Every
method keeps 4 attention-sink tokens and a recent window, and spends the rest of the budget its own way.

**Main result: 200 held-out WikiText-2 passages** (chosen before the run, no overlap with the
development passages; `holdout.json`)

| Cache kept | Full cache | Recent only (StreamingLLM) | Random | H2O | **EDTD** |
|---|---|---|---|---|---|
| 50% | 9.35 | 10.30 | 10.37 | 10.20 | **10.01** |
| 25% | 9.35 | 11.51 | 11.90 | 11.70 | **11.33** |

Paired, passage by passage (mean loss difference in nats/token, 95% bootstrap interval):

| EDTD vs | 50% kept | 25% kept |
|---|---|---|
| H2O | better on 124/200, +0.019 [0.012, 0.027] | better on 129/200, +0.033 [0.022, 0.045] |
| Random | better on 121/200, +0.035 [0.019, 0.054] | better on 124/200, +0.049 [0.027, 0.071] |
| Recent only | better on 100/200, +0.029 [0.006, 0.052] | better on 101/200, +0.016 [−0.018, 0.051] |

**What this shows, plainly:**

- **EDTD beats H2O at both budgets.** The interval is well clear of zero. At 25% kept, it closes about
  16% of H2O's gap to the full cache (11.70 → 11.33, with full at 9.35).
- **H2O is not a strong baseline here.** It barely beats random tokens, and at 25% it loses to simply
  keeping the most recent tokens. Raw attention sums favour early tokens, because every later query can
  see them. Focus weighting removes much of that bias.
- **Against "just keep the recent tokens", EDTD wins on average at 50%, but not on most passages.** It
  wins 100 of 200: a few passages gain a lot, and the rest are a wash. At 25% the difference is not
  significant. A recent window is a hard baseline on running text like WikiText. Long-range
  retrieval tasks are where a scored cache should matter more; that is not tested here.

### Development runs, and why the 200-passage run exists

The first run (`eviction.json`) used 40 test passages and 10 development passages.

- **On the 40 test passages**, EDTD beat H2O: 8.62 vs 8.66 at 50% kept, and 9.33 vs 9.61 at 25%.
- **On the 10 development passages, keeping only recent tokens won outright.** Every scored method did
  worse there, including H2O.

Ten passages is too few to tell a real effect from noise, so the whole comparison was re-run on 200 fresh
passages. The development result did not hold up: on the large set, "recent only" is no better than EDTD.

## The earlier version of this repo was wrong

The first version masked tokens inside a single forward pass and scored loss only on the tokens left.
That measures neither cache memory nor quality fairly, and it had no baseline. It reported perplexity
going from 12.2 to 26.6 at 10% "dropped". That code and its numbers have been removed. Everything above
comes from real cache eviction against standard baselines.

## Run it

```bash
pip install torch transformers datasets numpy
python run_eviction.py                                         # 40 test + 10 dev passages -> eviction.json
python run_eviction.py --start 50 --count 200 --out holdout.json   # held-out run, with per-passage losses
pytest -q tests
```

Set `MODEL` in `run_eviction.py` to your local Qwen3-8B path. It uses about 18 GB of GPU memory in bf16
with eager attention; the 200-passage run took about 3 minutes on an RTX 3090 Ti.

## Limits

- Attention scores come from the prefill pass (one-shot eviction after the prompt), not continuous
  eviction during long generation.
- Scoring needs the full attention matrices (eager attention), so it is a measurement of *which tokens to
  keep*, not a fast kernel.
- Tested on one model, one dataset, and a 512-token context.

MIT License
