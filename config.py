import os
HOME = os.getenv('PROJECT_BASE_PATH','/Users/sebyjacob/My_Projects/OpenSourceGAN')
IMAGE_METADATA_FOLDER = os.path.join(HOME,'OpenImagesV6/image_metadata')
IMAGE_CLASS_DESCRIPTIONS_FILE = os.path.join(IMAGE_METADATA_FOLDER, 'oidv6-class-descriptions.csv')
TRAIN_BBOX_ANNOTATIONS_FILE = os.path.join(IMAGE_METADATA_FOLDER, 'oidv6-train-annotations-bbox.csv')
TEST_BBOX_ANNOTATIONS_FILE = os.path.join(IMAGE_METADATA_FOLDER, 'test-annotations-bbox.csv')
VAL_BBOX_ANNOTATIONS_FILE = os.path.join(IMAGE_METADATA_FOLDER, 'validation-annotations-bbox.csv')
IMAGES_DICT_FOR_LABEL_META = os.path.join(IMAGE_METADATA_FOLDER,'images-for-label-meta')
IMAGE_IDS_FOLDER=os.path.join(IMAGE_METADATA_FOLDER,'image-ids')
LIST_IMAGE_URL_FILES = [os.path.join(IMAGE_IDS_FOLDER, i) for i in os.listdir(IMAGE_IDS_FOLDER) if 'csv' in i]
LABEL_HIERARCHY_JSON=os.path.join(HOME,'OpenImagesV6/bbox_labels_600_hierarchy.json')

IMAGES_ROOT = os.path.join(HOME,'OpenImagesV6/images')

DISCRIMINATOR_BASEMODEL = 'INCEPTION_V3'

INITIAL_DISCRIMINATOR_LEARNING_RATE = 1e-5
DISCRIMINATOR_CLIP_VALUE = 1.0
DISCRIMINATOR_DECAY = 6e-8

INITIAL_ADVERSARIAL_LEARNING_RATE = 1e-5
ADVERSARIAL_CLIP_VALUE = 1.0
ADVERSARIAL_DECAY = 6e-8
NOISE_LABEL_PERCENTAGE = 0.1

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNELS = 3
BATCH_SIZE=32

NUM_DISCRIMINATOR_STEPS = 1e5
NUM_TRAINING_STEPS = 1e10

def intialize_directories(interested_class):
    global DISCRIMINATOR_SAVE_DIR
    global DISCRIMINATOR_SAVE_PATH
    DISCRIMINATOR_SAVE_DIR = os.path.join(HOME,'GANS',interested_class)
    DISCRIMINATOR_SAVE_PATH = os.path.join(interested_class,'discriminator')


