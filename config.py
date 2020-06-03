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

