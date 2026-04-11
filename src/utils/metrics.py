"""
Evaluation metrics for multi-label classification.

In multi-label classification (like toxic comment classification),
each sample can belong to multiple classes simultaneously.
We need metrics that handle this properly.
"""

import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    accuracy_score,
)


def compute_metrics(
    all_labels: np.ndarray, all_probs: np.ndarray, threshold: float = 0.5
):
    """
    Compute comprehensive metrics for multi-label classification.

    Args:
        all_labels: Ground truth labels, shape (num_samples, num_classes)
        all_probs: Predicted probabilities (after sigmoid), shape (num_samples, num_classes)
        threshold: Threshold to convert probabilities to binary predictions

    Returns:
        dict: Dictionary containing various metrics
    """
    # Convert probabilities to binary predictions using the threshold
    all_preds = (all_probs >= threshold).astype(int)

    metrics = {}

    # --- ROC-AUC Score ---
    # This is the PRIMARY metric for the Kaggle Toxic Comment competition.
    # It measures the model's ability to RANK positive examples higher than negatives.
    # AUC = 1.0 means perfect ranking; AUC = 0.5 means random guessing.
    try:
        # 'macro' averages the AUC across all 6 labels equally
        metrics["roc_auc_macro"] = roc_auc_score(all_labels, all_probs, average="macro")
        # Per-class AUC so you can see which toxic category is hardest
        per_class_auc = roc_auc_score(all_labels, all_probs, average=None)
        metrics["roc_auc_per_class"] = per_class_auc
    except ValueError:
        # This happens if a class has no positive samples in the batch
        metrics["roc_auc_macro"] = 0.0
        metrics["roc_auc_per_class"] = np.zeros(all_labels.shape[1])

    # --- F1 Score ---
    # Harmonic mean of precision and recall. Good for imbalanced datasets.
    # 'macro' treats each class equally; 'micro' treats each sample equally.
    metrics["f1_macro"] = f1_score(
        all_labels, all_preds, average="macro", zero_division=0
    )
    metrics["f1_micro"] = f1_score(
        all_labels, all_preds, average="micro", zero_division=0
    )

    # --- Precision & Recall ---
    # Precision: of all predicted positives, how many are actually positive?
    # Recall: of all actual positives, how many did we find?
    metrics["precision_macro"] = precision_score(
        all_labels, all_preds, average="macro", zero_division=0
    )
    metrics["recall_macro"] = recall_score(
        all_labels, all_preds, average="macro", zero_division=0
    )

    # --- Subset Accuracy ---
    # The strictest metric: a sample is "correct" only if ALL labels match exactly.
    metrics["subset_accuracy"] = accuracy_score(all_labels, all_preds)

    return metrics


def print_classification_report(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    label_names: list,
    threshold: float = 0.5,
):
    """
    Print a detailed classification report with per-class metrics.

    Args:
        all_labels: Ground truth labels
        all_probs: Predicted probabilities
        label_names: List of label names (e.g., ['toxic', 'severe_toxic', ...])
        threshold: Classification threshold
    """
    all_preds = (all_probs >= threshold).astype(int)

    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)

    # Sklearn's built-in report
    print(
        classification_report(
            all_labels, all_preds, target_names=label_names, zero_division=0
        )
    )

    # Per-class AUC
    try:
        per_class_auc = roc_auc_score(all_labels, all_probs, average=None)
        print("\nPer-Class ROC-AUC:")
        for name, auc in zip(label_names, per_class_auc):
            print(f"  {name:20s}: {auc:.4f}")
        print(f"  {'MACRO AVERAGE':20s}: {np.mean(per_class_auc):.4f}")
    except ValueError:
        print("\nCould not compute per-class AUC (possibly missing classes in batch)")

    print("=" * 60)


def find_optimal_thresholds(all_labels: np.ndarray, all_probs: np.ndarray):
    """
    Find optimal classification thresholds for each class to maximize F1 score.

    Args:
        all_labels: Ground truth labels, shape (num_samples, num_classes)
        all_probs: Predicted probabilities, shape (num_samples, num_classes)

    Returns:
        np.ndarray: Optimal threshold for each class
    """
    optimal_thresholds = []
    for i in range(all_labels.shape[1]):
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.01):
            preds = (all_probs[:, i] >= thresh).astype(int)
            f1 = f1_score(all_labels[:, i], preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        optimal_thresholds.append(best_thresh)
    return np.array(optimal_thresholds)
