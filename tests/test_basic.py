"""
Basic tests for the entropy-driven token dropping prototype.
"""

import torch
from repos.edtd.core import EntropyTokenDropper


def test_shannon_entropy_computation():
    """Test that Shannon entropy is computed correctly."""
    dropper = EntropyTokenDropper()
    
    # Test with uniform distribution (max entropy)
    probs = torch.tensor([0.25, 0.25, 0.25, 0.25])
    entropy = dropper.compute_shannon_entropy(probs)
    expected = -4 * 0.25 * torch.log(torch.tensor(0.25))  # Should be ~1.386
    assert torch.allclose(entropy, expected, atol=1e-5)
    
    # Test with deterministic distribution (zero entropy)
    probs = torch.tensor([1.0, 0.0, 0.0, 0.0])
    entropy = dropper.compute_shannon_entropy(probs)
    assert entropy < 1e-5


def test_dynamic_threshold():
    """Test dynamic threshold computation."""
    dropper = EntropyTokenDropper(threshold_factor=1.5)
    
    entropies = torch.tensor([0.5, 1.0, 1.5, 2.0])
    threshold = dropper.compute_dynamic_threshold(entropies)
    
    # Median of [0.5, 1.0, 1.5, 2.0] is 1.25
    expected = 1.25 * 1.5
    assert torch.allclose(threshold, torch.tensor(expected))


def test_token_mask_creation():
    """Test boolean mask creation based on entropy threshold."""
    dropper = EntropyTokenDropper(threshold_factor=1.0)
    
    entropies = torch.tensor([0.5, 1.0, 1.5, 2.0])
    mask = dropper.create_token_mask(entropies)
    
    # All tokens with entropy >= median should be kept
    assert mask.sum() > 0
    assert mask.dtype == torch.bool


def test_topk_mask_creation():
    """Test token mask when limiting to top-k tokens."""
    dropper = EntropyTokenDropper()
    
    entropies = torch.tensor([0.5, 1.0, 1.5, 2.0])
    mask = dropper.create_token_mask(entropies, num_tokens=2)
    
    # Should keep exactly 2 tokens (highest entropy)
    assert mask.sum() == 2
    assert mask[3] == True  # Highest entropy token kept
    assert mask[2] == True  # Second highest kept


def test_mask_reset():
    """Test that reset clears all collected data."""
    dropper = EntropyTokenDropper()
    dropper.entropies.append(torch.tensor([1.0]))
    dropper.attention_masks.append(torch.tensor([True]))
    
    dropper.reset()
    
    assert len(dropper.entropies) == 0
    assert len(dropper.attention_masks) == 0


def test_get_combined_mask():
    """Test combined mask generation."""
    dropper = EntropyTokenDropper()
    dropper.attention_masks.append(torch.tensor([True, False, True, False]))
    dropper.attention_masks.append(torch.tensor([False, True, False, True]))
    
    mask = dropper.get_combined_mask(seq_len=4)
    
    # Should combine via OR (keep if any layer keeps it)
    assert mask.sum() == 4  # All tokens kept after OR operation


def test_no_hooks_registered_by_default():
    """Test that no hooks are registered initially."""
    dropper = EntropyTokenDropper()
    assert len(dropper._hooks) == 0


def test_get_combined_mask_empty():
    """Test combined mask when no masks collected."""
    dropper = EntropyTokenDropper()
    mask = dropper.get_combined_mask(seq_len=10)
    
    assert mask.sum() == 10  # All tokens kept as fallback
    assert mask.dtype == torch.bool
