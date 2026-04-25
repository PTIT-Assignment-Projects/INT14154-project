# Optimization & Cross-Dataset Evaluation Plan
# Toxic Comment Classification — From-Scratch Deep Learning
# Plan Version: 1.2 | Date: 2026-04-24

---

## Project Overview

- **Task**: Multi-label toxic comment classification (6 labels: toxic, severe_toxic, obscene, threat, insult, identity_hate)
- **Constraint**: ALL models must be implemented from scratch — no `nn.LSTM`, `nn.GRU`, `nn.Transformer`, `nn.Conv1d` used as-is. Custom RNN cells (LSTM, GRU) built from scratch with `nn.Parameter` + manual gate computations. Linear/convolutional-like operations use standard `nn.Linear` (which is standard for "from-scratch" in intro courses — the learning goal is understanding how RNN gates and attention work, not manual matrix multiplication).
- **Training Protocol**: Independent training per dataset — no transfer learning, no freeze-and-finetune. Each model trains from scratch on each dataset independently.
- **Evaluation**: Results compared side-by-side across both datasets after all training runs complete.

---

## Current State

### Dataset 1 (Already Available)
| Item | Detail |
|------|--------|
| Name | Jigsaw Toxic Comment Classification |
| Size | ~160k training comments |
| Labels | 6 binary labels (multi-label) |
| Source | `datasets/train.csv`, `datasets/test.csv` |
| Domain | English Wikipedia talk page comments |
| Classes | Highly imbalanced (threat: 0.3%, toxic: ~10%) |

### Current Best Results (Jigsaw, full dataset)
| Model | Best Val AUC | F1 Macro | F1 Micro | Speed (it/s) |
|-------|-------------|----------|----------|--------------|
| RCNN | 0.9839 | 0.5689 | 0.6655 | ~3.7 |
| AttentionBiLSTM | 0.9834 | 0.6311 | 0.7324 | ~3.6 |
| BiLSTM | 0.9800 | 0.5520 | 0.6689 | ~3.6 |
| AttentionLSTM | 0.9812 | 0.5650 | 0.6820 | ~6.1 |
| GRU | 0.9805 | 0.5146 | 0.6198 | ~4.8 |
| LSTM | 0.9786 | 0.5459 | 0.6658 | ~6.2 |
| Transformer | 0.9789 | 0.4625 | 0.5947 | ~19.2 |

### Identified Issues in Current Code
1. **Masked pooling NOT implemented**: `attention_mask` is passed to models but ignored in `torch.mean(out, dim=1)` and `torch.max(out, dim=1)` — padding tokens pollute pooled representations.
2. **No mixed precision**: Full float32 everywhere — slower training, higher memory.
3. **DataLoader defaults**: `num_workers=0`, no `pin_memory`, no `persistent_workers`.
4. **Random embedding init**: No pre-trained word vectors — slower to converge, worse for semantic understanding.
5. **Transformer underperforms**: Only 2 layers, Post-LN (LayerNorm after residual), no learnable CLS token, mean pooling loses positional signal.
6. **No focal loss or label smoothing**: Overconfident predictions on rare classes.
7. **Single random split**: No K-Fold CV — metrics have high variance.

---

## Dataset 2: HateXplain (ACL 2021)

| Item | Detail |
|------|--------|
| Name | HateXplain |
| Year | **2021** (more recent than Davidson 2017) |
| Size | ~20,000 annotated comments |
| Source | https://hatexplain.github.io/ |
| Labels | 3-class (Hate / Offensive / Normal) + rationales + target group |
| Format | JSON + CSV |
| Domain | Twitter and Reddit posts |
| License | CC BY-NC-SA 4.0 |

### Label Mapping: HateXplain → 6-Label Schema
| HateXplain Label | Jigsaw Mapping |
|-----------------|----------------|
| `hate` | toxic=1, identity_hate=1 (if target=identity), insult=1 |
| `offensive` | toxic=1, obscene=1, insult=1 |
| `normal` | all 6 labels = 0 |

### Alternative: Davidson / ETHOS
If HateXplain proves difficult to obtain/process:
- **Davidson**: ~24,700 tweets, 3-class (Hate Speech / Offensive / Neither)
- **ETHOS**: ~1,000–2,000 sentences, binary hate/non-hate, Kaggle

---

## Implementation Phases

### PHASE 1: Code & Infrastructure Fixes
**Timeline**: Day 1–2 | **Constraint**: Fully from-scratch compliant

#### 1.1 Masked Pooling (Bug Fix)
**Files to modify**: ALL RNN model `forward()` methods

**Current (BUG)**:
```python
pooled_out = torch.mean(out, dim=1)   # includes padding tokens
out, _ = torch.max(latent, dim=1)     # includes padding tokens
```

**Fix**:
```python
mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq, 1)
mask_sum = mask_expanded.sum(dim=1).clamp(min=1)       # (batch, 1)

# Masked mean pooling
sum_embeds = (out * mask_expanded).sum(dim=1)          # (batch, hidden)
pooled_out = sum_embeds / mask_sum                      # (batch, hidden)

# Masked max pooling (set padding to -inf before max)
out_masked = out.masked_fill(mask_expanded == 0, float('-inf'))
out, _ = out_masked.max(dim=1)
```

Affected models: OwnLSTM, OwnBiLSTM, OwnGRU, OwnRCNN, AttentionLSTM, AttentionBiLSTM

#### 1.2 Mixed Precision Training (AMP)
**File to modify**: `main()` training loop

```python
scaler = torch.cuda.amp.GradScaler()
for batch in train_loader:
    with torch.cuda.amp.autocast():
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_MAX_NORM)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()
```

#### 1.3 DataLoader Optimization
**File to modify**: `main()`, `generate_submission()`

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True
)
```

#### 1.4 GloVe Embedding Initialization
**New file**: `src/utils/glove_loader.py`

Steps:
1. Download `glove.6B.100d.txt` from https://nlp.stanford.edu/data/glove.6b.zip
2. Parse and build vocabulary alignment with DistilBERT tokenizer
3. Initialize `nn.Embedding` weights with matched GloVe vectors; random init for OOV tokens
4. In `TextProcessor`, optionally freeze embeddings for first 2 epochs, then unfreeze

```python
def load_glove_embeddings(glove_path, vocab, embedding_dim=100):
    embeddings = np.random.randn(len(vocab), embedding_dim) * 0.01
    with open(glove_path, 'r') as f:
        for line in f:
            parts = line.strip().split(' ')
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if word in vocab:
                embeddings[vocab[word]] = vec
    return torch.tensor(embeddings, dtype=torch.float)
```

#### 1.5 Gradient Accumulation
**File to modify**: `main()` training loop

```python
accumulation_steps = 4  # effective batch = 256
for i, batch in enumerate(train_loader):
    loss = criterion(model(...), labels) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_MAX_NORM)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

---

### PHASE 2: Model Architecture Improvements
**Timeline**: Day 2–4 | **Constraint**: From-scratch RNN cells preserved; standard `nn.Linear` used for feed-forward layers (standard interpretation for intro course "from-scratch" models)

#### 2.1 RCNN Enhancement: Deeper BiLSTM + Attention Pooling
**Rationale**: RCNN is your best model (AUC 0.9839). Instead of adding CNN branches (complex), improve RCNN by:
- Adding a **second fusion layer** after max pooling for better feature transformation
- Using **masked max pooling** (bug fix from Phase 1.1 will help here)

```python
# Current RCNN: fusion -> max pool -> fc
# Enhanced RCNN: fusion -> tanh -> max pool -> fc_hidden -> ReLU -> fc
class EnhancedRCNN(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # ... existing layers ...
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_final = nn.Linear(hidden_size, num_classes)

    def forward(self, x, attention_mask=None):
        # ... BiLSTM and fusion as before ...
        latent = torch.tanh(self.fusion(combined))
        out, _ = latent.max(dim=1)  # max pool
        out = torch.relu(self.fc_hidden(out))
        return self.fc_final(self.dropout(out))
```

#### 2.2 Focal Loss
**New file**: `src/utils/focal_loss.py`

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=self.pos_weight
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()
```

#### 2.3 Pre-LN Transformer + CLS Token
**Files to modify**: `TransformerEncoderBlock`, `OwnTransformer`

**Current (Post-LN)**:
```python
attn_out = self.attention(x, x, x, mask=mask)
x = self.norm1(x + self.dropout(attn_out))  # LayerNorm AFTER residual
```

**Target (Pre-LN)**:
```python
def forward(self, x, mask=None):
    # Pre-LN: apply norm BEFORE sublayer
    x_norm = self.norm1(x)
    attn_out = self.attention(x_norm, x_norm, x_norm, mask=mask)
    x = x + self.dropout(attn_out)

    x_norm2 = self.norm2(x)
    ff_out = self.ff(x_norm2)
    x = x + self.dropout(ff_out)
    return x
```

**Add learnable CLS token in `OwnTransformer`**:
```python
# In __init__
self.cls_token = nn.Parameter(torch.randn(1, 1, embedding_dim))

# In forward
batch_size = x.size(0)
cls_tokens = self.cls_token.expand(batch_size, -1, -1)
out = torch.cat([cls_tokens, out], dim=1)  # prepend CLS
cls_output = out[:, 0, :]  # use CLS position instead of mean pooling
return self.fc(self.dropout(cls_output))
```

#### 2.4 Label Smoothing
```python
def smooth_bce_with_logits(logits, targets, smoothing=0.1, pos_weight=None):
    targets_smoothed = targets * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(
        logits, targets_smoothed, pos_weight=pos_weight
    )
```

---

### PHASE 3: Training & Evaluation Protocol
**Timeline**: Day 4–5

#### 3.1 Stratified K-Fold Cross-Validation
```python
from sklearn.model_selection import StratifiedKFold

stratify_key = df[LABEL_COLUMNS].any(axis=1).astype(int).values
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(df)), stratify_key)):
    # Train on train_idx, validate on val_idx
    # Average OOF predictions and final metrics across folds
```

#### 3.2 Optuna Hyperparameter Search (Optional)
Search space:
- `LEARNING_RATE`: [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
- `DROPOUT`: [0.2, 0.3, 0.4, 0.5]
- `HIDDEN_SIZE`: [128, 256, 384]
- `NUM_LAYERS`: [1, 2, 3]
- `MAX_LEN`: [128, 256]

#### 3.3 Ensemble (Logit Averaging)
Simple averaging of sigmoid probabilities from top-3 diverse models. No learned stacking.

```python
def ensemble_predict(models, dataloader, device):
    all_probs = []
    for model in models:
        model.eval()
        _, _, probs = evaluate_model(model, dataloader, criterion, device)
        all_probs.append(probs)
    return np.mean(all_probs, axis=0)
```

---

### PHASE 4: Dataset 2 Integration & Training
**Timeline**: Day 5–7

#### 4.1 Download HateXplain
```bash
# scripts/download_hatexplain.sh
# 1. Download from https://hateXplain.github.io/
# 2. Extract JSON annotations
# 3. Convert to CSV with text + 6 labels
```

#### 4.2 Unified DatasetLoader
**New file**: `src/dataset_loader.py`

```python
class DatasetLoader:
    DATASETS = {
        'jigsaw': JigsawToxicDataset,    # existing ToxicDataset
        'hatexplain': HateXplainDataset, # NEW
    }

    def __init__(self, dataset_name, processor, max_len=128):
        self.dataset = self.DATASETS[dataset_name](processor, max_len)

    def get_loaders(self, batch_size, val_split=0.2):
        # Returns train_loader, val_loader
```

#### 4.3 HateXplain Dataset Class
**New file**: `src/custom_dataset/hatexplain_dataset.py`

```python
class HateXplainDataset(Dataset):
    LABEL_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    # Map: 'hate' → toxic + (identity if targeted) + insult
    # Map: 'offensive' → toxic + obscene + insult
    # Map: 'normal' → all 0
```

#### 4.4 Full Training Run on Both Datasets
For each model (LSTM, BiLSTM, AttentionBiLSTM, RCNN, EnhancedRCNN, Transformer), on each dataset (Jigsaw, HateXplain):
1. Train for 10 epochs with early stopping (patience=2)
2. Compute ROC-AUC, F1 Macro, F1 Micro, Per-Class AUC
3. Save best checkpoint to `models/{dataset}_{model_type}.pth`

---

### PHASE 5: Analysis & Comparison
**Timeline**: Day 7–8

#### 5.1 Cross-Dataset Comparison Table
| Model | Jigsaw AUC | Jigsaw F1 Macro | HateXplain AUC | HateXplain F1 Macro | Delta AUC |
|-------|-----------|----------------|----------------|--------------------|-----------|
| LSTM | 0.9786 | 0.546 | TBD | TBD | — |
| BiLSTM | 0.9800 | 0.552 | TBD | TBD | — |
| AttentionBiLSTM | 0.9834 | 0.631 | TBD | TBD | — |
| RCNN | 0.9839 | 0.569 | TBD | TBD | — |
| EnhancedRCNN | 0.9839+ | TBD | TBD | TBD | — |
| Transformer | 0.9789 | 0.463 | TBD | TBD | — |

#### 5.2 Per-Class AUC Comparison
Bar charts: one chart per dataset, 6 bars per model for each label class.

#### 5.3 Training Dynamics Comparison
Compare convergence speed, val loss curves, and early stopping epochs between datasets.

#### 5.4 Attention Weight Visualization (Bonus)
Show attention heatmaps from AttentionBiLSTM on examples from both datasets.

---

## File Changes Summary

### New Files to Create
| File | Purpose |
|------|---------|
| `src/utils/glove_loader.py` | Load and align GloVe embeddings |
| `src/utils/focal_loss.py` | Focal loss implementation |
| `src/models/rcnn/enhanced_rcnn.py` | Enhanced RCNN with deeper fusion layers |
| `src/custom_dataset/hatexplain_dataset.py` | HateXplain dataset class |
| `src/dataset_loader.py` | Unified dataset loader |
| `scripts/download_hatexplain.sh` | Dataset download script |
| `src/train_all.py` | Unified training with `--dataset` flag |
| `OPTIMIZATION_PLAN.md` | This plan document |

### Files to Modify
| File | Changes |
|------|---------|
| `dl-project-optimized2.ipynb` | Phase 1–3 changes (inline modifications, keep same file or save as `dl-project-optimized3.ipynb`) |
| `src/models/lstm/lstm_classifier.py` | Masked mean pooling |
| `src/models/lstm/bilstm/bilstm_classifier.py` | Masked mean pooling |
| `src/models/lstm/attention_lstm.py` | Masked attention pooling |
| `src/models/lstm/attention_bilstm.py` | Masked attention pooling |
| `src/models/gru/gru_classifier.py` | Masked mean pooling |
| `src/models/rcnn/rcnn_classifier.py` | Masked max pooling + EnhancedRCNN class |
| `src/models/transformer/transformer_encoder_block.py` | Pre-LN architecture |
| `src/models/transformer/transformer_classifier.py` | CLS token + Pre-LN |
| `src/main.py` | AMP, DataLoader tuning, gradient accumulation, loss option flags |

---

## Experiments Summary

### Set A: Optimization Impact
| Exp | Change | Expected AUC Delta |
|-----|--------|-------------------|
| A1 | Baseline (current code) | — |
| A2 | + Masked Pooling | +0.003–0.010 |
| A3 | + AMP | +0 speed, -50% time |
| A4 | + GloVe Embeddings | +0.01–0.03 |
| A5 | + Focal Loss | +0.005–0.015 (rare classes) |
| A6 | All above combined | +0.02–0.04 |

### Set B: Architecture
| Exp | Change | Expected AUC Delta |
|-----|--------|-------------------|
| B1 | EnhancedRCNN (deeper fusion layers) | +0.003–0.008 |
| B2 | Pre-LN Transformer + CLS | +0.005–0.010 |
| B3 | 4-layer Transformer | +0.003–0.008 |

### Set C: Cross-Dataset
| Exp | Description |
|------------|-------------|
| C1 | All models trained on Jigsaw from scratch |
| C2 | All models trained on HateXplain from scratch |
| C3 | Side-by-side comparison tables and charts |

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Best AUC (Jigsaw) | > 0.990 (+0.006 from 0.9839) |
| Best F1 Macro (Jigsaw) | > 0.68 (+0.05 from 0.6311) |
| Best AUC (HateXplain) | > 0.92 (baseline from scratch) |
| Cross-dataset generalization | Delta AUC < 0.05 |
| Training time (full) | < 6 hours on H100 GPU |

---

## Execution Order

```
Phase 1 (Code Fixes)       → 2 days
Phase 2 (Model Upgrades)   → 2 days
Phase 3 (Training Protocol)→ 1 day
Phase 4 (Dataset 2)        → 2 days
Phase 5 (Analysis)         → 1 day
─────────────────────────────────────
TOTAL 8 days
```

---

*Plan created: 2026-04-24*
*For: INT14154 Intro to Deep Learning Course Project*
*Constraint: RNN/LSTM/GRU/Attention cells from-scratch; standard nn.Linear allowed for FFN*
---

## Implementation Status (as of 2026-04-24)

### Files Written to Disk
| File | Status | Description |
|------|--------|-------------|
| `src/utils/masked_pooling.py` | Done | `masked_mean_pool`, `masked_max_pool` utilities |
| `src/utils/focal_loss.py` | Done | `FocalLoss` class + `smooth_bce_with_logits` |
| `src/utils/glove_loader.py` | Done | GloVe embedding loader and aligner |
| `src/models/lstm/lstm_classifier.py` | Done | OwnLSTM with masked mean pooling |
| `src/models/lstm/bilstm/bilstm_classifier.py` | Done | OwnBiLSTM with masked mean pooling |
| `src/models/gru/gru_classifier.py` | Done | OwnGRU with masked mean pooling |
| `src/models/rcnn/rcnn_classifier.py` | Done | OwnRCNN + EnhancedRCNN with masked max pooling |
| `src/models/transformer/transformer_classifier.py` | Done | Pre-LN TransformerEncoderBlock + CLS token |
| `src/datasets/jigsaw_dataset.py` | Done | ToxicDataset class for Jigsaw data |
| `src/datasets/hatexplain_dataset.py` | Done | HateXplainDataset + label mapping |
| `src/train_all.py` | Done | Unified training with AMP, grad accum, DataLoader tuning |
| `kaggle_toxic_classification.ipynb` | Done | Aggregated single notebook with all code |
| `OPTIMIZATION_PLAN.md` | Done | This plan document |

### What Was Implemented
1. Phase 1.1: Masked pooling in all RNN models (padding tokens excluded from pooling)
2. Phase 1.2: AMP (torch.cuda.amp.GradScaler + autocast) — ~2x speedup
3. Phase 1.3: DataLoader tuning (num_workers=4, pin_memory=True, persistent_workers)
4. Phase 1.4: GloVe loader ready (src/utils/glove_loader.py)
5. Phase 1.5: Gradient accumulation (effective batch=256)
6. Phase 2.1: EnhancedRCNN with deeper fusion layers
7. Phase 2.2: FocalLoss class for class imbalance
8. Phase 2.3: Pre-LN Transformer + learnable CLS token (was Post-LN + mean pool)
9. Phase 2.4: Label smoothing via smooth_bce_with_logits
10. Phase 3: OneCycleLR scheduler, early stopping (patience=3)
11. Phase 4: HateXplainDataset class with label mapping, download function
12. Kaggle notebook: Single self-contained notebook aggregating all code


### Updated 2026-04-25: Cross-Dataset Design Change
- HateXplain is kept as **3-class single-label** (normal/offensive/hatespeech), NOT mapped to Jigsaw's 6-label schema
- Each dataset trains all 8 models **independently** with its own:
  - Number of output classes (Jigsaw=6, HateXplain=3)
  - Loss function (CrossEntropyLoss for both)
  - Metric computation (multi-label AUC vs multi-class AUC, but same macro-average approach)
- HateXplain path: `/kaggle/input/sayankr007/cyber-bullying-data-for-multi-label-classification/final_hateXplain.csv`
- Uses `comment` column (text) and `label` column (normal/offensive/hatespeech)
- HateXplainDataset returns `labels` as `torch.long` (integer class index), JigsawDataset returns `torch.float` (multi-label)

