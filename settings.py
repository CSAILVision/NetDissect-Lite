# global settings
GPU = True
TEST_MODE = False
MODEL = 'resnet18' #resnet18, alexnet, resnet50, densenet161
DATASET = 'places365'
QUANTILE = 0.005
SEG_THRESHOLD = 0.04
SCORE_THRESHOLD = 0.04
TOPN = 10
PARALLEL = 1
OUTPUT_FOLDER = "result/pytorch_"+MODEL+"_"+DATASET

# sub settings
if MODEL != 'alexnet':
    DATA_DIRECTORY = '../NetDissect/dataset/broden1_224'
    IMG_SIZE = 224
else:
    DATA_DIRECTORY = '../NetDissect/dataset/broden1_227'
    IMG_SIZE = 227
if DATASET == 'places365':
    NUM_CLASSES = 365
elif DATASET == 'imagenet':
    NUM_CLASSES = 1000
if MODEL == 'resnet18':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/resnet18_places365.pth.tar'
        MODEL_PARALLEL = True
    elif DATASET == 'imagenet':
        MODEL_FILE = None
        MODEL_PARALLEL = False
elif MODEL == 'densenet161':
    FEATURE_NAMES = ['features']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/whole_densenet161_places365_python36.pth.tar'
        MODEL_PARALLEL = False
elif MODEL == 'resnet50':
    FEATURE_NAMES = ['layer4']
    if DATASET == 'places365':
        MODEL_FILE = 'zoo/whole_resnet50_places365_python36.pth.tar'
        MODEL_PARALLEL = False

if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    INDEX_FILE = 'index_sm.csv'
    OUTPUT_FOLDER += "_test"
else:
    WORKERS = 8
    BATCH_SIZE = 128
    TALLY_BATCH_SIZE = 16
    TALLY_AHEAD = 4
    INDEX_FILE = 'index.csv'
