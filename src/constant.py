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





BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 3
MAX_LEN = 128
EMBEDDING_DIM = 128
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.3




# model type
BILSTM_MODEL = 'bilstm'
ATTENTION_BILSTM_MODEL = 'attention_bilstm'
LSTM_MODEL = 'lstm'