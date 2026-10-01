"""
Entropy-Driven Token Dropping (EDTD) Prototype

This module implements a mechanism to dynamically prune tokens based on
their Shannon entropy in attention layers, reducing memory usage during
long-sequence inference while preserving important context.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Any


class EntropyTokenDropper:
    """
    Applies entropy-based token dropping to transformer attention layers.
    
    Tracks attention weights via forward hooks, computes Shannon entropy
    per token, and generates a boolean mask for cache pruning.
    """
    
    def __init__(self, threshold_factor: float = 1.0):
        """
        Args:
            threshold_factor: Multiplier for median entropy to set dynamic threshold.
                              Tokens with entropy < median * factor are considered low-importance.
        """
        self.threshold_factor = threshold_factor
        self.attention_masks: List[torch.Tensor] = []
        self.entropies: List[torch.Tensor] = []
        self._hooks: List[Any] = []
    
    def compute_shannon_entropy(self, probs: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Compute Shannon entropy of probability distribution.
        
        Args:
            probs: Probability distribution (e.g., attention weights)
            dim: Dimension along which to compute entropy
        
        Returns:
            Entropy values per token
        """
        # Clamp probabilities to avoid log(0)
        probs = torch.clamp(probs, min=1e-10, max=1.0)
        # Shannon entropy: -sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs), dim=dim)
        return entropy
    
    def compute_dynamic_threshold(self, entropies: torch.Tensor) -> torch.Tensor:
        """
        Compute dynamic threshold based on sequence median entropy.
        
        Args:
            entropies: Tensor of entropy values per token
        
        Returns:
            Dynamic threshold value
        """
        median_entropy = torch.median(entropies)
        return median_entropy * self.threshold_factor
    
    def create_token_mask(self, entropies: torch.Tensor, 
                          num_tokens: Optional[int] = None) -> torch.Tensor:
        """
        Create boolean mask for token dropping based on entropy threshold.
        
        Args:
            entropies: Tensor of entropy values per token
            num_tokens: Number of tokens to keep (if None, keeps all above threshold)
        
        Returns:
            Boolean mask where True means token is kept
        """
        threshold = self.compute_dynamic_threshold(entropies)
        mask = entropies >= threshold
        
        if num_tokens is not None and mask.sum() > num_tokens:
            # Keep only top-k tokens by entropy
            _, indices = torch.topk(entropies, num_tokens)
            mask = torch.zeros_like(entropies, dtype=torch.bool)
            mask[indices] = True
        
        return mask.to(entropies.device)
    
    def _attention_hook(self, module: nn.Module, 
                        input: Tuple, 
                        output: Tuple) -> None:
        """
        Forward hook to capture attention weights and compute entropy.
        
        Args:
            module: The attention module
            input: Input to the module
            output: Output from the module (typically (attention_weights, values))
        """
        if isinstance(output, tuple) and len(output) > 0:
            attn_weights = output[0]
            
            # Compute entropy per token (usually over heads and position dim)
            # Shape: (batch, heads, seq_len, seq_len) -> compute over last dim
            if attn_weights.dim() >= 3:
                entropies = self.compute_shannon_entropy(attn_weights, dim=-1)
                # Average over heads if multiple
                if entropies.dim() >= 3:
                    entropies = entropies.mean(dim=1)  # Average over heads
                
                self.entropies.append(entropies)
                
                # Create mask for this layer
                num_tokens = self.entropies[-1].shape[-1]
                mask = self.create_token_mask(self.entropies[-1], num_tokens)
                self.attention_masks.append(mask)
    
    def register_hooks(self, model: nn.Module, 
                       attention_layer_names: Optional[List[str]] = None) -> None:
        """
        Register forward hooks on attention layers.
        
        Args:
            model: The transformer model
            attention_layer_names: Names of attention layers to hook (if None, hooks all)
        """
        if attention_layer_names is None:
            # Auto-detect attention layers
            attention_layer_names = []
            for name, module in model.named_modules():
                if 'attention' in name.lower() or 'attn' in name.lower():
                    attention_layer_names.append(name)
        
        for name, module in model.named_modules():
            if any(attr in name.lower() for attr in ['attention', 'attn', 'mlp']):
                if 'mlp' not in name.lower():
                    handle = module.register_forward_hook(self._attention_hook)
                    self._hooks.append(handle)
    
    def reset(self) -> None:
        """Clear all collected entropies and masks."""
        self.attention_masks.clear()
        self.entropies.clear()
    
    def get_combined_mask(self, seq_len: int) -> torch.Tensor:
        """
        Get combined boolean mask for all layers.
        
        Args:
            seq_len: Sequence length
        
        Returns:
            Combined boolean mask (True = keep token)
        """
        if not self.attention_masks:
            return torch.ones(seq_len, dtype=torch.bool)
        
        # Combine masks across layers (keep token if kept in any layer)
        combined = torch.zeros(seq_len, dtype=torch.bool, device=self.attention_masks[0].device)
        for mask in self.attention_masks:
            if mask.numel() == seq_len:
                combined = combined | mask
        
        if not combined.any():
            # Fallback: keep all tokens if none survived
            combined = torch.ones(seq_len, dtype=torch.bool)
        
        return combined
    
    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
