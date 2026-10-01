import numpy as np

import run_eviction as ev


def test_every_method_keeps_exactly_the_budget_with_sinks_and_recent():
    rng = np.random.default_rng(0)
    scores = np.arange(ev.CTX, dtype=float)[::-1]          # oldest tokens score highest
    for method in ["recent", "random", "h2o", "edtd"]:
        for keep in ev.KEEPS:
            k = int(ev.CTX * keep)
            idx = ev.select(None if method in ("recent", "random") else scores, k, rng, method).tolist()
            assert len(idx) == k and len(set(idx)) == k
            assert idx == sorted(idx)
            assert set(range(ev.SINKS)) <= set(idx)        # attention sinks always kept
            assert ev.CTX - 1 in idx                        # newest token always kept


def test_scored_methods_take_the_highest_scoring_older_tokens():
    rng = np.random.default_rng(0)
    scores = np.zeros(ev.CTX)
    scores[[10, 20, 30]] = 5.0
    idx = set(ev.select(scores, 4 + 3 + 3 * 4, rng, "h2o").tolist())   # tiny budget
    assert {10, 20, 30} <= idx
