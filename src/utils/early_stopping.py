"""
Early Stopping - A regularization technique to prevent overfitting.

The Concept:
    During training, both training loss and validation loss decrease.
    At some point, the validation loss starts INCREASING while training loss
    keeps decreasing — this means the model is memorizing training data
    (overfitting) instead of learning general patterns.

    Early Stopping monitors the validation loss and stops training
    when it hasn't improved for 'patience' epochs.

    Example timeline:
        Epoch 1: val_loss = 0.50  ← best (saved)
        Epoch 2: val_loss = 0.45  ← new best (saved)
        Epoch 3: val_loss = 0.46  ← worse, counter = 1
        Epoch 4: val_loss = 0.47  ← worse, counter = 2
        Epoch 5: val_loss = 0.48  ← worse, counter = 3 → STOP! (patience=3)
        → Restore the model from Epoch 2 (the best one)
"""

import torch
import numpy as np


class EarlyStopping:
    """
    Stop training when validation loss stops improving.
    Optionally saves the best model checkpoint.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min', verbose: bool = True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement.
                           Higher patience = more chances for the model to recover.
                           Typical values: 3-10.
            min_delta (float): Minimum change to qualify as an improvement.
                              Prevents stopping due to tiny random fluctuations.
                              E.g., min_delta=0.001 means val_loss must drop by at least 0.001.
            mode (str): 'min' for loss (lower is better), 'max' for accuracy (higher is better)
            verbose (bool): Whether to print messages when status changes.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, score, model):
        """
        Call this at the end of each epoch with the validation metric.

        Args:
            score: The validation metric (loss or accuracy)
            model: The PyTorch model (to save its state)

        Returns:
            bool: True if training should stop
        """
        if self.mode == 'min':
            # For loss: lower is better, so we negate to make "higher is better"
            current_score = -score
        else:
            # For accuracy/AUC: higher is already better
            current_score = score

        if self.best_score is None:
            # First epoch — set baseline
            self.best_score = current_score
            self._save_checkpoint(model)
        elif current_score < self.best_score + self.min_delta:
            # No improvement
            self.counter += 1
            if self.verbose:
                print(f"  ⏳ EarlyStopping: {self.counter}/{self.patience} (no improvement)")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"  🛑 EarlyStopping triggered! Restoring best model...")
                self._restore_checkpoint(model)
        else:
            # Improvement found
            if self.verbose:
                if self.mode == 'min':
                    print(f"  ✅ EarlyStopping: val_loss improved ({-self.best_score:.4f} → {-current_score:.4f})")
                else:
                    print(f"  ✅ EarlyStopping: metric improved ({self.best_score:.4f} → {current_score:.4f})")
            self.best_score = current_score
            self.counter = 0
            self._save_checkpoint(model)

        return self.early_stop

    def _save_checkpoint(self, model):
        """Save model state dict (in-memory copy)."""
        # .state_dict() returns a reference, so we need deepcopy
        import copy
        self.best_model_state = copy.deepcopy(model.state_dict())

    def _restore_checkpoint(self, model):
        """Restore the model to the best checkpoint."""
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
