import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW  # Changed from Adam → AdamW (see explanation.md)
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.constant import (
    TRAIN_CSV_PATH, TEST_CSV_PATH, SUBMISSION_CSV_PATH,
    LABEL_COLUMNS,
    LSTM_MODEL_PATH, BILSTM_MODEL_PATH,
    ATTENTION_BILSTM_MODEL_PATH, BATCH_SIZE, MAX_LEN, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS,
    DROPOUT, LEARNING_RATE, EPOCHS, BILSTM_MODEL, ATTENTION_BILSTM_MODEL, GRU_MODEL, GRU_MODEL_PATH, RCNN_MODEL_PATH,
    RCNN_MODEL, LSTM_MODEL, TRANSFORMER_MODEL, TRANSFORMER_MODEL_PATH,
    WEIGHT_DECAY, WARMUP_RATIO, EARLY_STOPPING_PATIENCE, GRADIENT_CLIP_MAX_NORM
)
from src.preprocessing import TextProcessor
from src.custom_dataset.toxic_dataset import ToxicDataset
from src.models import OwnLSTM, OwnBiLSTM, AttentionBiLSTM, OwnGRU, OwnRCNN, OwnTransformer
from src.utils import compute_metrics, print_classification_report, EarlyStopping, WarmupCosineScheduler


def generate_submission(model, processor, device, submission_path):
    """Generate predictions for the test set and save as a submission CSV."""
    print(f"Generating submission to {submission_path}...")
    test_df = pd.read_csv(TEST_CSV_PATH)

    test_dataset = ToxicDataset(test_df, processor=processor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()

    submission_df = pd.DataFrame(all_preds, columns=LABEL_COLUMNS)
    submission_df.insert(0, 'id', test_df['id'].values)

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved as '{submission_path}'.")


def evaluate_model(model, data_loader, criterion, device):
    """
    Evaluate the model on a validation set.

    Returns:
        avg_loss: Average loss over the dataset
        all_labels: Ground truth labels (numpy array)
        all_probs: Predicted probabilities (numpy array)
    """
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Convert logits → probabilities using sigmoid
            probs = torch.sigmoid(logits)
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    avg_loss = total_loss / len(data_loader)
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    return avg_loss, all_labels, all_probs


def compute_class_weights(df, label_columns):
    """
    Compute class weights for handling imbalanced data.

    The toxic comment dataset is heavily IMBALANCED:
      - ~90% of comments are NON-toxic
      - Some categories like 'threat' have very few positive examples

    Without weighting, the model learns to just predict "not toxic" for everything
    because that gives 90% accuracy — but it's useless!

    pos_weight tells BCEWithLogitsLoss to penalize missed positive examples more heavily.
    Formula: pos_weight = num_negatives / num_positives

    Example: if 'threat' has 478 positive and 159,093 negative examples:
      pos_weight = 159,093 / 478 ≈ 332.8
      → Missing a 'threat' costs 332x more than a false alarm
    """
    pos_counts = df[label_columns].sum()
    neg_counts = len(df) - pos_counts
    pos_weights = neg_counts / pos_counts.clip(lower=1)  # clip to avoid division by zero

    print("\nClass weights (pos_weight):")
    for name, weight in zip(label_columns, pos_weights):
        print(f"  {name:20s}: {weight:.2f}")

    return torch.tensor(pos_weights.values, dtype=torch.float)


def train_model(model_type="bilstm"):
    # ========== 0. Configuration ==========
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure directories exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    # ========== 1. Load Data ==========
    print("Loading data...")
    df = pd.read_csv(TRAIN_CSV_PATH)
    # Using a subset for faster demonstration. Comment out for full training.
    df = df.sample(10000, random_state=42)

    # ========== 2. Preprocessing & Datasets ==========
    print("Initializing components...")
    processor = TextProcessor(model_name="distilbert-base-uncased", max_len=MAX_LEN)
    full_dataset = ToxicDataset(df, processor=processor)

    # ========== 3. Train/Val Split (80/20) ==========
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # ========== 4. Initialize Model ==========
    print(f"Building {model_type} model...")
    if model_type == BILSTM_MODEL:
        model = OwnBiLSTM(
            vocab_size=processor.tokenizer.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=len(LABEL_COLUMNS),
            dropout=DROPOUT
        ).to(device)
        model_path = BILSTM_MODEL_PATH
    elif model_type == ATTENTION_BILSTM_MODEL:
        model = AttentionBiLSTM(
            vocab_size=processor.tokenizer.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=len(LABEL_COLUMNS),
            dropout=DROPOUT
        ).to(device)
        model_path = ATTENTION_BILSTM_MODEL_PATH
    elif model_type == GRU_MODEL:
        model = OwnGRU(
            vocab_size=processor.tokenizer.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=len(LABEL_COLUMNS),
            dropout=DROPOUT
        ).to(device)
        model_path = GRU_MODEL_PATH
    elif model_type == RCNN_MODEL:
        model = OwnRCNN(
            vocab_size=processor.tokenizer.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=len(LABEL_COLUMNS),
            dropout=DROPOUT
        ).to(device)
        model_path = RCNN_MODEL_PATH
    elif model_type == TRANSFORMER_MODEL:
        model = OwnTransformer(
            vocab_size=processor.tokenizer.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=len(LABEL_COLUMNS),
            dropout=DROPOUT
        ).to(device)
        model_path = TRANSFORMER_MODEL_PATH
    elif model_type == LSTM_MODEL:
        model = OwnLSTM(
            vocab_size=processor.tokenizer.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=len(LABEL_COLUMNS),
            dropout=DROPOUT
        ).to(device)
        model_path = LSTM_MODEL_PATH
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    # Print model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ========== 5. Loss & Optimizer ==========
    # NEW: Compute class weights for the imbalanced dataset
    pos_weights = compute_class_weights(df, LABEL_COLUMNS).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # NEW: AdamW with weight decay (L2 regularization)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # NEW: Learning Rate Scheduler (Warmup + Cosine Annealing)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)
    print(f"LR Schedule: warmup for {warmup_steps} steps, then cosine decay over {total_steps} total steps")

    # NEW: Early Stopping
    early_stopper = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, mode='min', verbose=True)

    # ========== 6. Training Loop ==========
    print(f"\nStarting training for {model_type}...")
    train_losses = []
    val_losses = []
    val_aucs = []
    learning_rates = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # NEW: Gradient Clipping — prevents exploding gradients
            # RNNs (LSTM/GRU) are especially prone to gradient explosion
            # because gradients are multiplied through many time steps.
            # Clipping limits the gradient norm to a maximum value.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)

            optimizer.step()

            # NEW: Update learning rate (per step, not per epoch)
            scheduler.step()

            total_loss += loss.item()
            loop.set_postfix({"loss": f"{loss.item():.4f}"})

        # Record current learning rate
        current_lr = scheduler.get_last_lr()[0]
        learning_rates.append(current_lr)

        # ========== Validation with Metrics ==========
        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss, val_labels, val_probs = evaluate_model(model, val_loader, criterion, device)

        # Compute comprehensive metrics
        metrics = compute_metrics(val_labels, val_probs)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_aucs.append(metrics['roc_auc_macro'])

        print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Train Loss:     {avg_train_loss:.4f}")
        print(f"  Val Loss:       {avg_val_loss:.4f}")
        print(f"  Val ROC-AUC:    {metrics['roc_auc_macro']:.4f}")
        print(f"  Val F1 (macro): {metrics['f1_macro']:.4f}")
        print(f"  Val Precision:  {metrics['precision_macro']:.4f}")
        print(f"  Val Recall:     {metrics['recall_macro']:.4f}")
        print(f"  Learning Rate:  {current_lr:.6f}")

        # Check Early Stopping
        if early_stopper(avg_val_loss, model):
            print(f"\n🛑 Early stopping at epoch {epoch+1}!")
            break

    # ========== 7. Final Evaluation on Best Model ==========
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON VALIDATION SET (Best Model)")
    print("=" * 60)
    final_val_loss, final_labels, final_probs = evaluate_model(model, val_loader, criterion, device)
    print_classification_report(final_labels, final_probs, LABEL_COLUMNS)

    # ========== 8. Plotting ==========
    actual_epochs = len(train_losses)

    # Plot 1: Training and Validation Loss
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(range(1, actual_epochs + 1), train_losses, label='Train Loss', marker='o')
    axes[0].plot(range(1, actual_epochs + 1), val_losses, label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{model_type}: Train vs Val Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Validation ROC-AUC over epochs
    axes[1].plot(range(1, actual_epochs + 1), val_aucs, label='Val ROC-AUC', marker='D', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].set_title(f'{model_type}: Validation ROC-AUC')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Learning Rate Schedule
    axes[2].plot(range(1, actual_epochs + 1), learning_rates, label='Learning Rate', marker='^', color='red')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title(f'{model_type}: Learning Rate Schedule')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"plots/{model_type}_training_dashboard.png"
    plt.savefig(plot_path, dpi=150)
    print(f"\nTraining dashboard saved as {plot_path}")

    # ========== 9. Save Model ==========
    torch.save(model.state_dict(), model_path)
    print(f"Training complete! Model saved as {model_path}.")

    # ========== 10. Generate Submission ==========
    sub_path = SUBMISSION_CSV_PATH.replace(".csv", f"_{model_type}.csv")
    generate_submission(model, processor, device, sub_path)


if __name__ == "__main__":
    # You can choose which model to train here
    # train_model(model_type="lstm")
    train_model(model_type="bilstm")
