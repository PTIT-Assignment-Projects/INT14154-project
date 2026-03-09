LABEL_COLUMNS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
TEXT_COLUMN = 'comment_text'

# TRAIN
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
PATIENCE = 3
TRAIN_CSV_PATH = '/home/duongvct/Documents/workspace/PTIT/Y4T2/y4t2_intro_deep_learning_project/datasets/train.csv'
TEST_CSV_PATH = '/home/duongvct/Documents/workspace/PTIT/Y4T2/y4t2_intro_deep_learning_project/datasets/test.csv'
SAMPLE_SUBMISSION_PATH = '/home/duongvct/Documents/workspace/PTIT/Y4T2/y4t2_intro_deep_learning_project/datasets/sample_submission.csv'