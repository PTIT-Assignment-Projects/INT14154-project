import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from tqdm import tqdm

from src.custom_dataset import ToxicDataset
from src.preprocessing import TextProcessor
from src.models import ToxicBiLSTM, EarlyStopping
from src.constant import TEXT_COLUMN, LABEL_COLUMNS, TRAIN_CSV_PATH, TEST_CSV_PATH, SAMPLE_SUBMISSION_PATH, BATCH_SIZE, LEARNING_RATE, EPOCHS, PATIENCE


def train():
    # 0. Configuration

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(TRAIN_CSV_PATH)
    # Using a subset for faster demonstration or full training if you want. 
    # For initial test, let's take a sample or just run the whole thing.
    # df = df.sample(20000, random_state=42) # Comment this out for full training

    # 2. Preprocessing & Datasets
    print("Initializing components...")
    processor = TextProcessor(model_name="distilbert-base-uncased", max_len=128)
    full_dataset = ToxicDataset(df, processor=processor)
    
    # 3. Train/Val Split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 4. Initialize Model
    print("Building model...")
    model = ToxicBiLSTM(
        vocab_size=processor.tokenizer.vocab_size,
        embedding_dim=128,
        hidden_dim=256,
        n_layers=2,
        dropout=0.3
    ).to(device)

    # 5. Loss & Optimizer
    # BCEWithLogitsLoss is perfect for multi-label classification.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # 6. Training Loop with Early Stopping
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path="toxic_bilstm_model.pth")
    
    print("Starting training...")
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

        # Early Stopping check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered. Training halted.")
            break

    # 7. Load Best Model (saved by EarlyStopping)
    model.load_state_dict(torch.load("toxic_bilstm_model.pth"))
    print("Training complete! Best model weights loaded.")
    
    # 8. Generate Submission
    generate_submission(model, processor, device)

def generate_submission(model, processor, device):
    print("Generating submission...")
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
            
            logits = model(input_ids, attention_mask)
            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu())
            
    all_preds = torch.cat(all_preds, dim=0).numpy()
    
    submission_df = pd.DataFrame(all_preds, columns=LABEL_COLUMNS)
    submission_df.insert(0, 'id', test_df['id'].values)
    
    submission_file = "submission.csv"
    submission_df.to_csv(submission_file, index=False)
    print(f"Submission file saved as '{submission_file}'.")

if __name__ == "__main__":
    train()
