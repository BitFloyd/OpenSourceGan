import os

INTERESTED_CLASS = 'Furniture'
HOME = os.getenv('PROJECT_BASE_PATH', '/Users/sebyjacob/My_Projects/OpenSourceGAN')
DATA_HOME = os.getenv('DATA_PATH','/data')
IMAGE_METADATA_FOLDER = os.path.join(HOME, 'OpenImagesV6/image_metadata')
IMAGE_CLASS_DESCRIPTIONS_FILE = os.path.join(IMAGE_METADATA_FOLDER, 'oidv6-class-descriptions.csv')
TRAIN_BBOX_ANNOTATIONS_FILE = os.path.join(IMAGE_METADATA_FOLDER, 'oidv6-train-annotations-bbox.csv')
TEST_BBOX_ANNOTATIONS_FILE = os.path.join(IMAGE_METADATA_FOLDER, 'test-annotations-bbox.csv')
VAL_BBOX_ANNOTATIONS_FILE = os.path.join(IMAGE_METADATA_FOLDER, 'validation-annotations-bbox.csv')
IMAGES_DICT_FOR_LABEL_META = os.path.join(IMAGE_METADATA_FOLDER, 'images-for-label-meta')
IMAGE_IDS_FOLDER = os.path.join(IMAGE_METADATA_FOLDER, 'image-ids')
LIST_IMAGE_URL_FILES = [os.path.join(IMAGE_IDS_FOLDER, i) for i in os.listdir(IMAGE_IDS_FOLDER) if 'csv' in i]
LABEL_HIERARCHY_JSON = os.path.join(HOME, 'OpenImagesV6/bbox_labels_600_hierarchy.json')

IMAGES_ROOT = os.path.join(DATA_HOME, 'OpenImages/images')

DISCRIMINATOR_BASEMODEL = 'INCEPTION_V3'

INITIAL_DISCRIMINATOR_LEARNING_RATE = 1e-3
DISCRIMINATOR_CLIP_VALUE = 1.0
DISCRIMINATOR_DECAY = 6e-8

INPUT_GENERATOR_NOISE_DIM = 100

INITIAL_ADVERSARIAL_LEARNING_RATE = 1e-5
ADVERSARIAL_CLIP_VALUE = 1.0
ADVERSARIAL_DECAY = 6e-8
NOISE_LABEL_PERCENTAGE = 0.1

MAX_QUEUE_BATCH_SIZE = 100
BATCH_RETRY_LIMIT=10
NUM_BATCH_GEN_THREADS = 10
MIN_ACCEPTABLE_IMAGE_AREA = 32*32

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
IMAGE_CHANNELS = 3
BATCH_SIZE = 32

NUM_DISCRIMINATOR_STEPS = 500
NUM_TRAINING_STEPS = 1e10
