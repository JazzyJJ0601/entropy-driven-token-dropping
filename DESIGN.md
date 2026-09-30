# DESIGN.md: Entropy-Driven Token Dropping for Streaming LLM Inference

## Novel Idea
This repository proposes a novel inference optimization technique called Entropy-Driven Token Dropping (EDTD). Unlike existing KV cache eviction methods that rely on static recency heuristics (e.g., H2O) or require retraining (e.g., SnapKV), EDTD dynamically prunes Key-Value (KV) cache entries during runtime based on real-time attention entropy. By measuring the uncertainty of attention scores across attention heads, we identify and discard low-information tokens that contribute minimally to downstream predictions. This approach operates purely during inference, requiring no changes to model weights or architecture.

## Related Work Gap
Current approaches to long-context inference are limited by either memory capacity or computational overhead. Traditional eviction policies (e.g., Least Recently Used) ignore semantic importance, leading to information loss on critical tokens when memory fills up. Fine-tuning based methods are computationally expensive and brittle across different base models and domains. There is no model-agnostic, training-free method that adapts pruning strictly to the dynamic attention patterns of the specific input sequence. Most existing research focuses on static compression ratios or learned projections rather than dynamic, token-level adaptivity based on information-theoretic metrics.

## Approach Sketch
1.  **Hook Injection:** Intercept attention outputs in the transformer layers using PyTorch forward hooks. This captures the attention matrix before softmax normalization for entropy analysis.
2.  **Entropy Calculation:** For each query-token pair, compute Shannon entropy over the attention distribution: H = -sum(p * log(p + epsilon)). This quantifies how focused the model is on specific context tokens. High entropy indicates attention is spread across many tokens (important context), while low entropy indicates attention is concentrated on a few tokens (potential redundancy).
3.  **Thresholding:** Define a dynamic threshold based on the median entropy of the current sequence window. Tokens with entropy below this threshold are flagged as redundant and removed from the KV cache immediately.
4.  **Memory Management:** Maintain a compact boolean mask to track valid cache indices, skipping computations for pruned tokens in subsequent layers to save GPU cycles and memory bandwidth.

## Why It's Technically Impressive
This approach directly targets the quadratic memory complexity of standard transformer inference without requiring model weight changes. It is mathematically grounded in information theory, offering a theoretical guarantee of preserving high-information pathways. Achieving 50%+ VRAM reduction on 32k context windows with <1% perplexity degradation would be a significant contribution to edge deployment research. It allows larger contexts on consumer hardware without specialized inference engines.

## Test Model
We will test this on Llama-2-7B and Mistral-7B using the HuggingFace `transformers` library. We will validate on the Needle In A Haystack benchmark to ensure retrieval accuracy is maintained while measuring peak memory usage and tokens-per-second throughput. We will provide a standalone script to reproduce the memory savings on standard consumer GPUs like the RTX 3090.