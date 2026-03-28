LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TEXT_COLUMN = 'comment_text'

# Data paths
TRAIN_CSV_PATH = "datasets/train.csv"
TEST_CSV_PATH = "datasets/test.csv"
SUBMISSION_CSV_PATH = "datasets/submission.csv"

# Model paths
LSTM_MODEL_PATH = "models/lstm_params.pth"
BILSTM_MODEL_PATH = "models/bilstm_params.pth"
ATTENTION_BILSTM_MODEL_PATH = "models/attention_bilstm_params.pth"
GRU_MODEL_PATH = 'models/gru_params.pth'
RCNN_MODEL_PATH = 'models/rcnn_params.pth'
TRANSFORMER_MODEL_PATH = 'models/transformer_params.pth'

# ========== Core Hyperparameters ==========
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 10          # Increased from 3 (Early Stopping will handle when to really stop)
MAX_LEN = 128
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3

# ========== NEW: Optimization Hyperparameters ==========
WEIGHT_DECAY = 1e-4            # L2 regularization strength for AdamW
WARMUP_RATIO = 0.1             # 10% of total steps used for LR warmup
EARLY_STOPPING_PATIENCE = 3   # Stop after 3 epochs without improvement
GRADIENT_CLIP_MAX_NORM = 1.0   # Maximum gradient norm for clipping

# Model type identifiers
BILSTM_MODEL = 'bilstm'
ATTENTION_BILSTM_MODEL = 'attention_bilstm'
LSTM_MODEL = 'lstm'
GRU_MODEL = 'gru'
RCNN_MODEL = 'rcnn'
TRANSFORMER_MODEL = 'transformer'