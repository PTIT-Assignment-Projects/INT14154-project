import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm

from src.constant import (
    TRAIN_CSV_PATH, TEST_CSV_PATH, SUBMISSION_CSV_PATH,
    LABEL_COLUMNS,
    LSTM_MODEL_PATH, BILSTM_MODEL_PATH,
    ATTENTION_BILSTM_MODEL_PATH, BATCH_SIZE, MAX_LEN, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS,
    DROPOUT, LEARNING_RATE, EPOCHS
)
from src.preprocessing import TextProcessor
from src.custom_dataset.toxic_dataset import ToxicDataset
from src.models import OwnLSTM, OwnBiLSTM, AttentionBiLSTM, OwnGRU

# Hyperparameters


def generate_submission(model, processor, device, submission_path):
    print(f"Generating submission to {submission_path}...")
    test_df = pd.read_csv(TEST_CSV_PATH)

    # Optional: use subset for testing
    # test_df = test_df.sample(1000, random_state=42)

    test_dataset = ToxicDataset(test_df, processor=processor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Both models now accept attention_mask even if they don't use it
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()

    submission_df = pd.DataFrame(all_preds, columns=LABEL_COLUMNS)
    submission_df.insert(0, 'id', test_df['id'].values)

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved as '{submission_path}'.")

def train_model(model_type="bilstm"):
    # 0. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Ensure model directory exists
    os.makedirs("models", exist_ok=True)

    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(TRAIN_CSV_PATH)
    # Using a subset for faster demonstration. Comment out for full training.
    df = df.sample(10000, random_state=42) 

    # 2. Preprocessing & Datasets
    print("Initializing components...")
    processor = TextProcessor(model_name="distilbert-base-uncased", max_len=MAX_LEN)
    full_dataset = ToxicDataset(df, processor=processor)

    # 3. Train/Val Split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize Model
    print(f"Building {model_type} model...")
    if model_type == "bilstm":
        model = OwnBiLSTM(
            vocab_size=processor.tokenizer.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=len(LABEL_COLUMNS),
            dropout=DROPOUT
        ).to(device)
        model_path = BILSTM_MODEL_PATH
    elif model_type == "attention_bilstm":
        model = AttentionBiLSTM(
            vocab_size=processor.tokenizer.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=len(LABEL_COLUMNS),
            dropout=DROPOUT
        ).to(device)
        model_path = ATTENTION_BILSTM_MODEL_PATH
    elif model_type == "gru":
        model = OwnGRU(
            vocab_size=processor.tokenizer.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=len(LABEL_COLUMNS),
            dropout=DROPOUT
        ).to(device)
        # Using a default path for GRU if not defined in constant.py
        model_path = "models/gru_model.pth"
    else:
        model = OwnLSTM(
            vocab_size=processor.tokenizer.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            num_classes=len(LABEL_COLUMNS),
            dropout=DROPOUT
        ).to(device)
        model_path = LSTM_MODEL_PATH

    # 5. Loss & Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop
    print(f"Starting training for {model_type}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # 7. Save Model
    torch.save(model.state_dict(), model_path)
    print(f"Training complete! Model saved as {model_path}.")
    
    # 8. Generate Submission
    # Use a specific submission path for each model
    sub_path = SUBMISSION_CSV_PATH.replace(".csv", f"_{model_type}.csv")
    generate_submission(model, processor, device, sub_path)

if __name__ == "__main__":
    # You can choose which model to train here
    # train_model(model_type="lstm")
    train_model(model_type="bilstm")
