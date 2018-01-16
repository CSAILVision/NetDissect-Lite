# global settings
GPU = True
TEST_MODE = True
MODEL = 'resnet18'
NUM_CLASSES = 1000
QUANTILE = 0.05
DATA_DIRECTORY = '../NetDissect/dataset/broden1_224'
IMG_SIZE = 224
TOPN = 5



# sub settings
if MODEL == 'resnet18':
    FEATURE_NAMES = ['layer4']
    MODEL_FILE = None
    OUTPUT_FOLDER = "result/pytorch_resnet18_places365"

if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4
    INDEX_FILE = 'index_sm.csv'
else:
    WORKERS = 8
    BATCH_SIZE = 128
    INDEX_FILE = 'index.csv'