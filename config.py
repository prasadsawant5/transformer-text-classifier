import os

VOCAB_SIZE = 11736              # Number of unique words in the dataset
N_CLASSES = 3                   # Number of output classes
MAX_LEN = 81                    # Maximum length of a sentence in the dataset

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 100

NUM_HEADS = 6                  # Number of attention heads
DIM_MODEL = 64                 # Output dimension of the last dense layer of feed forward module
DIM_FF = 32                    # Output dimension of 1st dense layer of feed forward module
NUM_LAYERS = 4                 # Number of encoder layers
DROPOUT = 0.5

PAD = '[PAD]'
EOS = '[EOS]'

LOGS = './logs'
SAVED_MODELS = './saved_models'
DATA = 'data'
TRAIN = 'finance.csv'
FINANCE_DATA = 'financial_data'
DATASET_PATH = os.path.join(DATA, FINANCE_DATA)
PATHS = [
    os.path.join(DATASET_PATH, 'Sentences_AllAgree.txt'), 
    os.path.join(DATASET_PATH, 'Sentences_75Agree.txt'), 
    os.path.join(DATASET_PATH, 'Sentences_66Agree.txt'), 
    os.path.join(DATASET_PATH, 'Sentences_50Agree.txt')
]