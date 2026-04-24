"""
Unified training script for both Jigsaw and HateXplain datasets.
Supports: AMP, gradient accumulation, multiple loss types, DataLoader tuning.
"""
import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm

from src.datasets.jigsaw_dataset import ToxicDataset
from src.datasets.hatexplain_dataset import HateXplainDataset
from src.models.lstm.lstm_classifier import OwnLSTM
from src.models.lstm.bilstm.bilstm_classifier import OwnBiLSTM
from src.models.lstm.attention_lstm import AttentionLSTM
from src.models.lstm.bilstm.attention_bilstm import AttentionBiLSTM
from src.models.gru.gru_classifier import OwnGRU
from src.models.rcnn.rcnn_classifier import OwnRCNN, EnhancedRCNN
from src.models.transformer.transformer_classifier import OwnTransformer
from src.utils.focal_loss import FocalLoss, smooth_bce_with_logits


LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
MODEL_CONFIGS = {
    'lstm':          {'model_class': OwnLSTM,          'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3},
    'bilstm':        {'model_class': OwnBiLSTM,        'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3},
    'attentionlstm': {'model_class': AttentionLSTM,    'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3},
    'attentionbilstm': {'model_class': AttentionBiLSTM, 'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3},
    'gru':           {'model_class': OwnGRU,           'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3},
    'rcnn':          {'model_class': OwnRCNN,          'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3},
    'enhancedrcnn':  {'model_class': EnhancedRCNN,     'hidden_size': 256, 'num_layers': 2, 'dropout': 0.3},
    'transformer':   {'model_class': OwnTransformer,   'hidden_size': 256, 'num_layers': 4, 'dropout': 0.3},
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    metrics = {}
    try:
        metrics['auc'] = roc_auc_score(labels, probs, average='macro')
    except:
        metrics['auc'] = 0.0
    try:
        metrics['f1_macro'] = f1_score(labels, preds, average='macro', zero_division=0)
        metrics['f1_micro'] = f1_score(labels, preds, average='micro', zero_division=0)
    except:
        metrics['f1_macro'] = 0.0
        metrics['f1_micro'] = 0.0
    return metrics


def get_loss_function(loss_type='bce', pos_weight=None, gamma=2.0, smoothing=0.1):
    if loss_type == 'focal':
        return FocalLoss(gamma=gamma, pos_weight=pos_weight)
    elif loss_type == 'smooth':
        return lambda logits, targets: smooth_bce_with_logits(logits, targets, smoothing=smoothing, pos_weight=pos_weight)
    else:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler=None, grad_accum=1):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc='Training')
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels) / grad_accum
            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels) / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        pbar.set_postfix({'loss': f'{loss.item() * grad_accum:.4f}'})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)

    all_labels = np.vstack(all_labels)
    all_probs = np.vstack(all_probs)
    metrics = compute_metrics(all_labels, all_probs)
    metrics['loss'] = total_loss / len(dataloader)

    return metrics, all_labels, all_probs


def get_jigsaw_dataset(tokenizer, max_len, data_dir=None):
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    train_texts = train_df['comment_text'].fillna('').tolist()
    train_labels = train_df[LABEL_COLS].values.tolist()
    full_dataset = ToxicDataset(train_texts, train_labels, tokenizer, max_len)

    val_size = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset


def get_hatexplain_dataset(tokenizer, max_len, csv_path):
    from src.datasets.hatexplain_dataset import load_hatexplain_from_csv
    dataset = load_hatexplain_from_csv(csv_path, tokenizer, max_len)
    val_size = int(len(dataset) * 0.2)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    return train_dataset, val_dataset


def train_model(
    model_name,
    train_dataset,
    val_dataset,
    tokenizer,
    device,
    epochs=10,
    batch_size=64,
    lr=1e-3,
    loss_type='bce',
    use_amp=True,
    num_workers=4,
    grad_accum=1,
    patience=3
):
    config = MODEL_CONFIGS[model_name]
    vocab_size = len(tokenizer)

    model = config['model_class'](
        vocab_size=vocab_size,
        embedding_dim=128,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        num_classes=6,
        dropout=config['dropout']
    ).to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    pos_weight = torch.tensor([1.0, 5.0, 2.0, 10.0, 2.0, 5.0]).to(device)
    criterion = get_loss_function(loss_type, pos_weight=pos_weight)
    val_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs // grad_accum
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.1
    )

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    best_auc = 0
    best_metrics = None
    patience_counter = 0

    for epoch in range(epochs):
        print(f'\n=== Epoch {epoch+1}/{epochs} ===')
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, scaler, grad_accum)
        val_metrics, _, _ = evaluate(model, val_loader, val_criterion, device)

        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_metrics["loss"]:.4f} | '
              f'Val AUC: {val_metrics["auc"]:.4f} | Val F1 Macro: {val_metrics["f1_macro"]:.4f}')

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            best_metrics = val_metrics
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    return best_metrics, best_auc


def run_all_models(dataset_name, data_dir=None, hatexplain_csv=None, epochs=10, batch_size=64, lr=1e-3, loss_type='bce'):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    max_len = 128

    if dataset_name == 'jigsaw':
        train_dataset, val_dataset = get_jigsaw_dataset(tokenizer, max_len, data_dir)
    elif dataset_name == 'hatexplain':
        train_dataset, val_dataset = get_hatexplain_dataset(tokenizer, max_len, hatexplain_csv)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    print(f'Train size: {len(train_dataset)}, Val size: {len(val_dataset)}')

    results = {}
    for model_name in MODEL_CONFIGS:
        print(f'\n{"="*50}')
        print(f'Training {model_name} on {dataset_name}')
        print(f'{"="*50}')
        metrics, best_auc = train_model(
            model_name, train_dataset, val_dataset, tokenizer, device,
            epochs=epochs, batch_size=batch_size, lr=lr, loss_type=loss_type
        )
        results[model_name] = {'auc': best_auc, 'f1_macro': metrics['f1_macro'], 'f1_micro': metrics['f1_micro']}
        print(f'{model_name} Best Val AUC: {best_auc:.4f}')

    print('\n' + '='*60)
    print(f'Results for {dataset_name}')
    print('='*60)
    for model_name, m in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
        print(f'{model_name:20s} | AUC: {m["auc"]:.4f} | F1 Macro: {m["f1_macro"]:.4f} | F1 Micro: {m["f1_micro"]:.4f}')

    return results


if __name__ == '__main__':
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='jigsaw', choices=['jigsaw', 'hatexplain'])
    parser.add_argument('--data_dir', type=str, default='datasets')
    parser.add_argument('--hatexplain_csv', type=str, default='datasets/hatexplain.csv')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'focal', 'smooth'])
    args = parser.parse_args()

    run_all_models(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        hatexplain_csv=args.hatexplain_csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_type=args.loss
    )