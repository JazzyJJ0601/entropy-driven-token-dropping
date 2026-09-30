# Entropy-Driven Token Dropping (EDTD)

A prototype implementation of entropy-based token pruning for transformer models. This technique dynamically identifies and marks low-importance tokens based on their attention entropy, enabling memory-efficient inference for long sequences.

## Overview

Transformer models process all tokens in a sequence uniformly, which becomes prohibitively expensive for long contexts. EDTD addresses this by:

1. Computing Shannon entropy of attention weights per token
2. Using dynamic thresholding based on sequence median entropy
3. Generating compact boolean masks for cache management

## Installation

```bash
pip install torch
```

## Usage

```python
from repos.edtd.core import EntropyTokenDropper
import torch

# Initialize dropper with custom threshold factor
dropper = EntropyTokenDropper(threshold_factor=1.0)

# Register hooks on a transformer model (e.g., GPT-2)
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2")
dropper.register_hooks(model)

# Run inference
inputs = torch.tensor([[1, 2, 3, 4, 5]])
with torch.no_grad():
    outputs = model(inputs)

# Get combined mask for token retention
mask = dropper.get_combined_mask(seq_len=5)
print(f"Keeping {mask.sum()} of {len(mask)} tokens")

# Clean up
dropper.remove_hooks()
```

## API

### EntropyTokenDropper

| Method | Description |
|--------|-------------|
| `compute_shannon_entropy(probs, dim=-1)` | Compute Shannon entropy of probability distribution |
| `compute_dynamic_threshold(entropies)` | Get dynamic threshold from median entropy |
| `create_token_mask(entropies, num_tokens=None)` | Create boolean mask for token dropping |
| `register_hooks(model)` | Attach forward hooks to attention layers |
| `get_combined_mask(seq_len)` | Combine masks across all layers |
| `remove_hooks()` | Clean up registered hooks |

## Testing

```bash
pytest repos/edtd/tests/test_basic.py
```

## How It Works

1. **Hook Attachment**: Forward hooks capture attention weights from transformer layers
2. **Entropy Computation**: Shannon entropy (-Σ p log p) calculated per token
3. **Dynamic Thresholding**: Threshold = median(entropy) × factor
4. **Mask Generation**: Tokens with entropy ≥ threshold are retained
5. **Cache Management**: Boolean mask used to prune KV cache during inference

## Performance

Expected benefits:
- **Memory**: 30-50% reduction in KV cache size for long sequences
- **Speed**: Proportional improvement in inference time for large batches
- **Quality**: Minimal accuracy degradation on well-tuned threshold factors

## License

MIT License
