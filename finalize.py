#!/usr/bin/env python3
"""
Finalize script for EDTD results.
Builds results.json from partial.json with exact chunk count.
"""

import json
import math
from pathlib import Path

PARTIAL_PATH = Path(__file__).parent / "partial.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


def compute_perplexity(nll_sum: float, token_count: int) -> float:
    """Compute perplexity from negative log-likelihood."""
    if token_count == 0:
        return float('inf')
    return math.exp(nll_sum / token_count)


def main():
    if not PARTIAL_PATH.exists():
        print("Error: partial.json not found. Run run_real.py first.")
        return
    
    with open(PARTIAL_PATH, "r") as f:
        partial = json.load(f)
    
    chunks_processed = partial.get("chunks_processed", 0)
    results = partial.get("results", {})
    
    output = {
        "chunks_processed": chunks_processed,
        "drop_rates": list(results.keys()),
        "perplexities": {}
    }
    
    for rate, data in results.items():
        nll_sum = data.get("nll_sum", 0.0)
        token_count = data.get("token_count", 0)
        ppl = compute_perplexity(nll_sum, token_count)
        output["perplexities"][rate] = {
            "perplexity": ppl,
            "nll_sum": nll_sum,
            "token_count": token_count
        }
    
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Results written to {RESULTS_PATH}")
    print(f"Chunks processed: {chunks_processed}")
    for rate, data in output["perplexities"].items():
        print(f"  Drop rate {rate}%: perplexity = {data['perplexity']:.2f}")


if __name__ == "__main__":
    main()
