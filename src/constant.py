LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TEXT_COLUMN = 'comment_text'

# Data paths
TRAIN_CSV_PATH = "datasets/train.csv"
TEST_CSV_PATH = "datasets/test.csv"
SUBMISSION_CSV_PATH = "datasets/submission.csv"

# Model paths
LSTM_MODEL_PATH = "models/lstm_params.pth"
BILSTM_MODEL_PATH = "models/bilstm_params.pth"
ATTENTION_LSTM_MODEL_PATH = "models/attention_lstm_params.pth"
ATTENTION_BILSTM_MODEL_PATH = "models/attention_bilstm_params.pth"