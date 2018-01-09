# global settings
GPU = True
MODEL = 'resnet18'
NUM_CLASSES = 1000

WORKERS = 8
BATCH_SIZE = 4
QUANTILE = 0.005
DATA_DIRECTORY = 'dataset/broden1_227'
IMG_SIZE = 227
TOPN = 5

# sub settings
if MODEL == 'resnet18':
    FEATURE_NAMES = ['layer4']
    MODEL_FILE = None
    OUTPUT_FOLDER = "result/pytorch_resnet18_places365"
