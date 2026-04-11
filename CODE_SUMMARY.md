# Toxic Comment Classification Project - Code Summary

## Overview
This project implements various deep learning models from scratch for toxic comment classification, based on the Jigsaw Toxic Comment Classification Challenge. The project includes LSTM, BiLSTM, Attention BiLSTM, GRU, RCNN, and Transformer models built using PyTorch.

## Project Structure

```
INT14154-project/
├── datasets/                 # Train/test CSV files
├── docs/                     # Documentation
├── src/
│   ├── constant.py           # Hyperparameters and constants
│   ├── main.py               # Training script
│   ├── preprocessing/        # Text processing utilities
│   │   └── preprocessing.py  # TextProcessor class
│   ├── custom_dataset/       # Dataset classes
│   │   └── toxic_dataset.py  # ToxicDataset implementation
│   ├── models/               # Model implementations
│   │   ├── lstm/             # LSTM variants
│   │   ├── gru/              # GRU models
│   │   ├── rcnn/             # RCNN models
│   │   └── transformer/      # Transformer models
│   └── utils/                # Utility functions
│       ├── metrics.py        # Evaluation metrics
│       ├── early_stopping.py # Early stopping implementation
│       └── lr_scheduler.py   # Learning rate scheduler
├── generate_nb.py            # Notebook generation script
├── kaggle_toxic_comment_notebook.ipynb # Jupyter notebook
├── README.md                 # Project overview
└── CODE_SUMMARY.md           # This file
```

## Detailed Component Explanations

### 1. Configuration (`src/constant.py`)
Defines all hyperconstants and paths used throughout the project:

- **Label Columns**: `["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]`
- **Data Paths**: Paths to train/test CSV files and submission template
- **Model Paths**: Save paths for each model variant
- **Core Hyperparameters**:
  - `BATCH_SIZE = 32`
  - `LEARNING_RATE = 1e-3`
  - `EPOCHS = 10` (with early stopping)
  - `MAX_LEN = 128` (sequence length)
  - `EMBEDDING_DIM = 128`
  - `HIDDEN_SIZE = 256`
  - `NUM_LAYERS = 2`
  - `DROPOUT = 0.3`
- **Optimization Hyperparameters**:
  - `WEIGHT_DECAY = 1e-4` (L2 regularization)
  - `WARMUP_RATIO = 0.1` (LR warmup)
  - `EARLY_STOPPING_PATIENCE = 3`
  - `GRADIENT_CLIP_MAX_NORM = 1.0`

### 2. Text Processing (`src/preprocessing/preprocessing.py`)
The `TextProcessor` class handles tokenization using HuggingFace's `AutoTokenizer`:

- Uses `distilbert-base-uncased` tokenizer by default
- Converts text to `input_ids` and `attention_mask` tensors
- Handles padding, truncation, and tensor conversion
- Properly handles various input types (str, list, pandas Series)

### 3. Dataset (`src/custom_dataset/toxic_dataset.py`)
The `ToxicDataset` class implements PyTorch's Dataset interface:

- Takes a dataframe and text processor
- Handles both labeled (train/val) and unlabeled (test) data
- Returns dictionary with `input_ids`, `attention_mask`, and `labels`
- Automatically flattens tensors for single-item batches

### 4. Model Implementations

#### LSTM Models (`src/models/lstm/`)
- **OwnLSTMCell**: Basic LSTM cell implementation from scratch
- **OwnLSTM**: Unidirectional LSTM with:
  - Embedding layer
  - Stacked LSTM layers
  - Global average pooling
  - Dropout and classification head
- **OwnBiLSTM**: Bidirectional LSTM (two OwnLSTM layers processing forward/backward)
- **AttentionLSTM**: LSTM with self-attention mechanism for weighted pooling
- **AttentionBiLSTM**: Bidirectional LSTM with attention

#### GRU Models (`src/models/gru/`)
- **OwnGRUCell**: GRU cell implementation from scratch
- **OwnGRU**: Stacked GRU layers with similar architecture to LSTM

#### RCNN Models (`src/models/rcnn/`)
- **OwnRCNN**: Recurrent Convolutional Neural Network combining:
  - Bidirectional LSTM for context
  - CNN for local feature extraction
  - Max pooling and classification

#### Transformer Models (`src/models/transformer/`)
- **OwnTransformer**: Transformer encoder implementation from scratch:
  - Positional encoding
  - Multi-head self-attention
  - Feed-forward networks
  - Layer normalization and residual connections

### 5. Utility Functions (`src/utils/`)

#### Metrics (`src/utils/metrics.py`)
- Computes ROC-AUC, F1, precision, recall for multi-label classification
- Supports both macro and micro averaging
- Includes threshold optimization for F1 score

#### Early Stopping (`src/utils/early_stopping.py`)
- Implements patience-based early stopping
- Saves best model state when validation loss improves
- Supports both min and max modes

#### Learning Rate Scheduler (`src/utils/lr_scheduler.py`)
- **WarmupCosineScheduler**: Combines linear warmup with cosine annealing
- Gradually increases LR during warmup phase
- Then follows cosine decay to minimum LR

### 6. Main Training Script (`src/main.py`)

#### Key Features:
1. **Device Management**: Automatically uses CUDA if available
2. **Data Loading**: Loads and optionally samples training data
3. **Class Weighting**: Computes pos_weights for BCEWithLogitsLoss to handle class imbalance
4. **Model Building**: Factory pattern for creating different model types
5. **Training Loop**:
   - Training with gradient clipping (prevents exploding gradients)
   - Validation after each epoch
   - Learning rate scheduling (step-wise)
   - Early stopping based on validation loss
6. **Evaluation**: Comprehensive metrics reporting
7. **Visualization**: Creates training dashboard plots:
   - Train/Val Loss
   - Validation ROC-AUC
   - Learning Rate Schedule
8. **Submission Generation**: Creates CSV files for Kaggle submission
9. **Model Saving**: Stores best model state_dict

#### Training Process:
1. Load and preprocess data
2. Split into train/validation sets (80/20)
3. Initialize model, loss function (with class weights), optimizer (AdamW), and scheduler
4. Train for specified epochs with:
   - Forward pass
   - Loss computation
   - Backward pass with gradient clipping
   - Optimizer step
   - Learning rate update
5. Validate and compute metrics
6. Apply early stopping if no improvement
7. Plot training progress
8. Save best model
9. Generate submission file

### 7. Notebook Generation (`generate_nb.py`)
Script to convert the project structure into a Jupyter notebook for demonstration/educational purposes.

### 8. Jupyter Notebook (`kaggle_toxic_comment_notebook.ipynb`)
Interactive notebook that demonstrates:
- Data exploration
- Model training
- Evaluation
- Submission generation

## Key Technical Innovations

1. **From-Scratch Implementations**: All RNN cells (LSTM, GRU) and Transformer components built manually using PyTorch primitives
2. **Advanced Optimization**:
   - AdamW optimizer with weight decay
   - Learning rate warmup + cosine annealing
   - Gradient clipping for stable RNN training
3. **Class Imbalance Handling**:
   - Computes pos_weights for BCEWithLogitsLoss
   - Addresses severe imbalance (~90% non-toxic samples)
4. **Comprehensive Evaluation**:
   - Multi-label metrics (ROC-AUC, F1, precision, recall)
   - Per-class and aggregate reporting
5. **Modular Design**:
   - Separate concerns (processing, dataset, models, utils)
   - Easy to extend with new model variants
6. **Production-Ready Features**:
   - Early stopping
   - Model checkpointing
   - Submission generation
   - Reproducible results (random seeds)

## Usage Instructions

1. **Data Preparation**:
   - Download train.csv and test.csv from Kaggle Jigsaw Toxic Comment Classification Challenge
   - Place them in the `datasets/` directory

2. **Training**:
   ```bash
   python src/main.py
   ```
   - Modify `train_model()` call in main.py to select different model types:
     - `"lstm"` - Unidirectional LSTM
     - `"bilstm"` - Bidirectional LSTM (default)
     - `"attention_bilstm"` - BiLSTM with attention
     - `"gru"` - GRU model
     - `"rcnn"` - RCNN model
     - `"transformer"` - Transformer model

3. **Evaluation**:
   - Training script automatically computes and displays validation metrics
   - Generates training dashboard plots in `plots/` directory
   - Saves best model to `models/` directory
   - Creates submission file in `datasets/` directory

## Model Architectures Summary

| Model | Key Characteristics |
|-------|-------------------|
| LSTM | Unidirectional, global average pooling |
| BiLSTM | Bidirectional, concatenates forward/backward outputs |
| Attention BiLSTM | BiLSTM with self-attention for weighted temporal pooling |
| GRU | Gated Recurrent Unit, simpler than LSTM but similar performance |
| RCNN | Combines BiLSTM (context) with CNN (local features) |
| Transformer | Self-attention based, parallel processing, positional encoding |

## Dependencies
- torch
- torchvision
- torchaudio
- transformers (for tokenization)
- pandas
- numpy
- matplotlib
- tqdm

All dependencies are managed through `pyproject.toml` and `uv.lock` files.

## Results
The project achieves competitive performance on the toxic comment classification task, with Attention BiLSTM and Transformer models typically performing best due to their ability to capture long-range dependencies and focus on relevant text segments.

## Future Improvements
1. Experiment with different embedding strategies (GloVe, FastText)
2. Try hierarchical attention mechanisms
3. Implement ensemble methods
4. Add more sophisticated data augmentation
5. Experiment with different loss functions (focal loss, etc.)
6. Add hyperparameter tuning capabilities