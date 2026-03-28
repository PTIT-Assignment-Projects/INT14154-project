"""
Learning Rate Scheduler with Linear Warmup + Cosine Annealing.

Why do we need a learning rate schedule?
    - A CONSTANT learning rate is rarely optimal.
    - Too HIGH at the start → training is unstable, loss explodes.
    - Too LOW → training is slow and gets stuck.
    - Too HIGH at the end → model oscillates around the minimum.

The Strategy:
    1. WARMUP Phase (first few steps):
       - Learning rate starts from nearly 0 and linearly increases to the target LR.
       - This gives the model a gentle start: random weights need small careful steps.

    2. COSINE DECAY Phase (remaining steps):
       - Learning rate smoothly decreases following a cosine curve.
       - This lets the model make fine-grained adjustments near the end.

    Visual representation:
        LR │    /‾‾‾‾\
           │   /      \
           │  /        \
           │ /          \
           │/            \
           └──────────────→ Steps
           ↑ warmup ↑ cosine decay
"""

import math
from torch.optim.lr_scheduler import LambdaLR


class WarmupCosineScheduler:
    """
    Creates a learning rate scheduler with linear warmup and cosine annealing.
    """

    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.0):
        """
        Args:
            optimizer: The PyTorch optimizer
            warmup_steps: Number of warmup steps (typically 5-10% of total steps)
            total_steps: Total number of training steps
            min_lr_ratio: Minimum LR as a fraction of the initial LR (default: 0 = decay to 0)
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        # LambdaLR multiplies the base LR by the value returned by lr_lambda
        self.scheduler = LambdaLR(optimizer, lr_lambda=self._lr_lambda)

    def _lr_lambda(self, step):
        """
        Compute the learning rate multiplier for a given step.

        Returns a value between 0 and 1 that the base LR is multiplied by.
        """
        if step < self.warmup_steps:
            # Warmup: linear increase from 0 to 1
            # At step 0: return ~0; at step warmup_steps: return 1.0
            return float(step) / float(max(1, self.warmup_steps))
        else:
            # Cosine decay: smooth decrease from 1 to min_lr_ratio
            progress = float(step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Scale between min_lr_ratio and 1.0
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_decay

    def step(self):
        """Advance the scheduler by one step. Call this every training step (batch)."""
        self.scheduler.step()

    def get_last_lr(self):
        """Get the last computed learning rate."""
        return self.scheduler.get_last_lr()
