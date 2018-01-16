# global settings
GPU = True
TEST_MODE = False
MODEL = 'resnet18'
DATASET = 'places365'
QUANTILE = 0.05
TOPN = 20
PARALLEL = 1


# sub settings
if MODEL == 'resnet18':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        NUM_CLASSES = 365
        DATA_DIRECTORY = '../NetDissect/dataset/broden1_224'
        IMG_SIZE = 224
        MODEL_FILE = 'zoo/resnet18_places365.pth.tar'
        MODEL_PARALLEL = True
        OUTPUT_FOLDER = "result/pytorch_resnet18_places365"
    elif DATASET == 'imagenet':
        NUM_CLASSES = 1000
        DATA_DIRECTORY = '../NetDissect/dataset/broden1_224'
        IMG_SIZE = 22
        MODEL_FILE = None
        MODEL_PARALLEL = False
        OUTPUT_FOLDER = "result/pytorch_resnet18_imagenet"

if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    INDEX_FILE = 'index_sm.csv'
else:
    WORKERS = 8
    BATCH_SIZE = 128
    TALLY_BATCH_SIZE = 16
    TALLY_AHEAD = 4
    INDEX_FILE = 'index.csv'