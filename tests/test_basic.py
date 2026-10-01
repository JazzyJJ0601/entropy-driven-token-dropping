"""
Basic tests for the entropy-driven token dropping prototype.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import EntropyTokenDropper


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
    
    # PyTorch median of [0.5, 1.0, 1.5, 2.0] returns 1.0 (2nd element)
    expected = 1.0 * 1.5
    assert torch.allclose(threshold, torch.tensor(expected, dtype=threshold.dtype))


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


def test_drop_rate_affects_logits():
    """Test that different drop rates produce different loss values on a tiny model."""
    import torch.nn as nn
    
    # Create a tiny random model for testing
    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(100, 32)
            self.linear = nn.Linear(32, 100)
        
        def forward(self, input_ids, labels=None):
            x = self.embedding(input_ids)
            logits = self.linear(x)
            if labels is not None:
                # Compute loss: cross-entropy, ignoring -100 labels
                mask = (labels != -100).float()
                nll = nn.functional.cross_entropy(
                    logits.view(-1, 100), labels.view(-1), reduction='none'
                ).view(input_ids.shape)
                loss = (nll * mask).sum() / mask.sum()
                return type('obj', (object,), {'logits': logits, 'loss': loss})()
            return type('obj', (object,), {'logits': logits, 'loss': None})()

    model = TinyModel()
    model.eval()
    
    # Create deterministic input
    torch.manual_seed(42)
    input_ids = torch.randint(0, 100, (1, 10))
    
    # Get loss with no drop (rate=0) - all tokens contribute
    with torch.no_grad():
        out0 = model(input_ids, labels=input_ids)
        loss0 = out0.loss
    
    # Simulate drop rate=30: mask out 3 lowest-entropy tokens (first 3 for test)
    # Tokens with -100 labels are excluded from loss computation
    with torch.no_grad():
        labels = input_ids.clone()
        labels[:, :3] = -100  # Mark first 3 tokens as ignored (30%)
        out30 = model(input_ids, labels=labels)
        loss30 = out30.loss
    
    # The losses should differ because different tokens contribute to the loss
    assert not torch.allclose(loss0, loss30), "Loss should differ between drop rates 0 and 30"
