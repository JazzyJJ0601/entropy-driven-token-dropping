#!/usr/bin/env python3
"""
Real run script for entropy-driven token dropping on Qwen3-8B.
Processes wikitext-2 chunks of 512 tokens, checkpoints to partial.json after each chunk,
and computes perplexity at drop rates 0, 10, 20, 30 percent.
"""


import os
import json
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from core import EntropyTokenDropper

MODEL_NAME = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"  # Qwen3-8B, same local copy as kv-svd-compress
SEQ_LEN = 512
STRIDE = 256
DROP_RATES = [0, 10, 20, 30]
PARTIAL_PATH = Path(__file__).parent / "partial.json"
RESULTS_PATH = Path(__file__).parent / "results.json"
NUM_CHUNKS_TO_PROCESS = 50  # Process this many chunks (adjustable)


def load_checkpoint() -> Optional[Dict[str, Any]]:
    """Load existing checkpoint if it exists."""
    if PARTIAL_PATH.exists():
        with open(PARTIAL_PATH, "r") as f:
            return json.load(f)
    return None


def save_checkpoint(data: Dict[str, Any]) -> None:
    """Save checkpoint to partial.json."""
    with open(PARTIAL_PATH, "w") as f:
        json.dump(data, f, indent=2)


def load_wikitext_chunks(n_chunks: int) -> list:
    """Load n_chunks of 512-token chunks from wikitext-2 test split."""
    from transformers import AutoTokenizer
    from datasets import load_dataset
    
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Concatenate and tokenize
    text = "\n".join([row["text"] for row in ds if row["text"]])
    tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    chunks = []
    for i in range(0, len(tokens) - SEQ_LEN, STRIDE):
        chunk_ids = tokens[i:i + SEQ_LEN].unsqueeze(0)  # Ensure shape is (1, SEQ_LEN)
        chunks.append(chunk_ids)
        if len(chunks) >= n_chunks:
            break
    return chunks


def compute_perplexity(nll_sum: float, token_count: int) -> float:
    """Compute perplexity from negative log-likelihood."""
    if token_count == 0:
        return float('inf')
    return torch.exp(torch.tensor(nll_sum / token_count)).item()


def process_chunk(
    model: nn.Module,
    tokenizer,
    chunk_ids: torch.Tensor,
    drop_rate: float,
    dropper: EntropyTokenDropper
) -> tuple[float, int]:
    """
    Process a single chunk with specified drop rate.
    Returns (nll_sum, token_count) where token_count is the number of kept tokens.
    """
    batch_size, seq_len = chunk_ids.shape

    if batch_size == 0 or seq_len == 0:
        return 0.0, 0

    # Reset dropper for this chunk
    dropper.reset()
    dropper.register_hooks(model)

    try:
        # FIRST PASS: compute logits and attention weights to estimate entropy
        with torch.no_grad():
            # Get logits from the model
            outputs = model(
                input_ids=chunk_ids,
            )
            logits = outputs.logits  # Shape: (batch, seq_len, vocab_size)
            
            # Compute per-token entropy from logits (entropy of next-token distribution)
            # Softmax over vocab, then entropy
            probs = torch.softmax(logits, dim=-1)  # (batch, seq_len, vocab)
            # Clamp to avoid log(0)
            probs = torch.clamp(probs, min=1e-10, max=1.0)
            # Per-token entropy: -sum(p * log(p)) over vocab
            token_entropy = -torch.sum(probs * torch.log(probs), dim=-1)  # (batch, seq_len)
            token_entropy = token_entropy.squeeze(0)  # (seq_len,)

        # Select tokens to drop: lowest-information = lowest entropy tokens
        num_tokens_to_drop = int(seq_len * drop_rate / 100.0)
        
        if num_tokens_to_drop > 0:
            # Get indices of lowest-entropy tokens (these are the ones to drop)
            # We want to drop the ones with LOWEST entropy (most predictable, least informative)
            _, drop_indices = torch.topk(token_entropy, num_tokens_to_drop, largest=False)
            
            # Create mask: True = keep, False = drop
            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=chunk_ids.device)
            keep_mask[drop_indices] = False
        else:
            keep_mask = torch.ones(seq_len, dtype=torch.bool, device=chunk_ids.device)

        # SECOND PASS: compute loss only on kept tokens
        with torch.no_grad():
            # Create input_ids: drop masked tokens by setting to pad_token_id
            masked_input_ids = chunk_ids.clone()
            pad_token_id = tokenizer.pad_token_id
            masked_input_ids[:, ~keep_mask] = pad_token_id

            # Create labels: -100 for dropped tokens (excludes them from loss)
            # Standard convention: -100 positions are ignored in loss computation
            labels = chunk_ids.clone()
            labels[:, ~keep_mask] = -100

            # Forward pass with labels
            outputs = model(
                input_ids=masked_input_ids,
                labels=labels,
            )
            loss = outputs.loss

        # loss is already averaged over non-ignored positions
        # Compute total NLL by multiplying by number of kept tokens
        token_count = keep_mask.sum().item()
        if token_count == 0:
            nll_sum = 0.0
        else:
            nll_sum = loss.item() * token_count

        return nll_sum, int(token_count)

    finally:
        dropper.remove_hooks()


def main():
    print("Loading model and tokenizer...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16, 
        local_files_only=True,
        device_map="auto"
    )
    model.eval()
    
    # Load chunks
    print(f"Loading wikitext-2 dataset ({NUM_CHUNKS_TO_PROCESS} chunks)...")
    chunks = load_wikitext_chunks(NUM_CHUNKS_TO_PROCESS)
    print(f"Loaded {len(chunks)} chunks of {SEQ_LEN} tokens each")
    
    # Initialize checkpoint
    checkpoint = load_checkpoint()
    if checkpoint is None:
        checkpoint = {
            "chunks_processed": 0,
            "results": {str(rate): {"nll_sum": 0.0, "token_count": 0} for rate in DROP_RATES}
        }
        save_checkpoint(checkpoint)
    
    chunks_processed = checkpoint["chunks_processed"]
    results = checkpoint["results"]
    
    print(f"Resuming from chunk {chunks_processed}")
    
    print(f"Processing {min(len(chunks), NUM_CHUNKS_TO_PROCESS)} chunks total...")
    
    dropper = EntropyTokenDropper(threshold_factor=1.0)
    
    total_chunks = min(len(chunks), NUM_CHUNKS_TO_PROCESS)
    
    for i in range(chunks_processed, total_chunks):
        chunk_ids = chunks[i].to(model.device)
        
        for rate in DROP_RATES:
            nll_sum, token_count = process_chunk(
                model, tokenizer, chunk_ids, float(rate), dropper
            )
            
            rate_key = str(rate)
            results[rate_key]["nll_sum"] += nll_sum
            results[rate_key]["token_count"] += token_count
        
        chunks_processed += 1
        checkpoint["chunks_processed"] = chunks_processed
        checkpoint["results"] = results
        save_checkpoint(checkpoint)
        
        print(f"Chunk {chunks_processed}/{total_chunks} completed. Checkpoint saved.")
    
    print("All chunks processed. Finalizing...")
    print("completed")


if __name__ == "__main__":
    main()
