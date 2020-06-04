import json
import multiprocessing
import os
import random
import time
from queue import Queue

import numpy as np
import wandb
from skimage.io import imread
from skimage.transform import resize
from skimage.util.dtype import img_as_float
from tensorflow import keras
from tqdm import tqdm

import config

w_run = wandb.init(project="OpenSourceGAN")

interested_class = 'Furniture'
GAN_SAVE_DIR = os.path.join(config.HOME, 'GANS', interested_class)

if (not os.path.exists(GAN_SAVE_DIR)):
    os.makedirs(GAN_SAVE_DIR, exist_ok=True)

GENERATOR_SAVE_PATH = os.path.join(GAN_SAVE_DIR, 'generator')
os.makedirs(GENERATOR_SAVE_PATH, exist_ok=True)
DISCRIMINATOR_SAVE_PATH = os.path.join(GAN_SAVE_DIR, 'discriminator')
os.makedirs(DISCRIMINATOR_SAVE_PATH, exist_ok=True)


batch_queue = Queue(maxsize=config.MAX_QUEUE_BATCH_SIZE)


def batch_populator(queue,generator):
    while 1:
        if (queue.full()):
            time.sleep(0.1)
        else:
            queue.put(generator.get_next_item())
            print("QUEUE IS AT....", queue.qsize())

class GANDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, training_dictionary, batch_size=32, shuffle=True):
        """
        training_dictionary: A dict with local image-paths and bboxes
        """
        self.training_dictionary = training_dictionary
        self.keys = list(self.training_dictionary.keys())
        self.num_training_images = len(self.keys)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        num_batches = int(np.floor(self.num_training_images / self.batch_size))
        return num_batches

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Get relevant images for the indexes
        sampled_images_list = []
        for k in indexes:
            imageID = self.keys[k]
            filepath = self.training_dictionary[imageID]['filepath']
            bounding_box_list = self.training_dictionary[imageID]['list_bboxes']

            if (not len(bounding_box_list)):
                # This should not happen..........
                print("NO BOUNDING BOXES FOUND FOR {filepath} with ImageID {imageID}".format(filepath=filepath,
                                                                                             imageID=imageID))
                continue
            # Pick a random bounding box
            bbox = random.choice(bounding_box_list)
            xmin, xmax, ymin, ymax = float(bbox['XMin']), float(bbox['XMax']), float(bbox['YMin']), float(bbox['YMax'])

            try:
                image = self.preprocess_frame(imread(filepath))
            except:
                continue

            image_x = image.shape[1]
            xmin = max(0, int(xmin * image_x))
            xmax = min(image_x, int(xmax * image_x))

            image_y = image.shape[0]
            ymin = max(0, int(ymin * image_y))
            ymax = min(image_y, int(ymax * image_y))

            sampled_image = image[ymin:ymax, xmin:xmax, :]
            sampled_image = resize(sampled_image, output_shape=(config.IMAGE_HEIGHT, config.IMAGE_WIDTH))
            sampled_images_list.append(sampled_image)

        sampled_images_list = np.array(sampled_images_list)
        while (len(sampled_images_list) < self.batch_size):
            index = np.random.randint(0, len(sampled_images_list))
            sampled_images_list = np.vstack((sampled_images_list, sampled_images_list[index]))

        labels = np.random.uniform(low=0.6, high=0.99, size=(len(sampled_images_list), 1))

        return sampled_images_list, labels

    def get_next_item(self):
        self.index += 1
        items = self.__getitem__(self.index)

        # Safety
        if (self.index % self.__len__() == 0):
            self.on_epoch_end()

        return items

    def preprocess_frame(self, frame):
        frame = img_as_float(frame)
        if (frame.ndim == 2):
            frame_new = np.zeros((frame.shape[0], frame.shape[1], 3))
            frame_new[:, :, 0] = frame
            frame_new[:, :, 1] = frame
            frame_new[:, :, 2] = frame
        else:
            frame_new = frame

        return frame_new

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.index = -1
        self.indexes = np.arange(len(self.training_dictionary.keys()))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


class TrainGANPipeline:

    def __init__(self, GAN, interested_class='Furniture'):

        self.GAN = GAN

        with open(os.path.join(config.IMAGES_DICT_FOR_LABEL_META, interested_class + '_images_dict.json'), 'r') as f:
            self.IMAGE_BBOX_DICT = json.load(f)

        # Do a sanity check and pop the imageIDs that we cannot read.
        print("DOING A SANITY CHECK TO ONLY KEEP IMAGES WE HAVE SUCCESSFULLY DOWNLOADED")
        keys = list(self.IMAGE_BBOX_DICT.keys())

        for key in tqdm(keys):
            filename = self.IMAGE_BBOX_DICT[key]['filepath']
            if (os.path.exists(filename)):
                continue
            else:
                self.IMAGE_BBOX_DICT.pop(key)

        print("{} IMAGES CAN BE READ SUCCESSFULLY AND WILL BE USED FOR TRAINING.............".format(
            len(self.IMAGE_BBOX_DICT)))

        self.generator = GANDataGenerator(self.IMAGE_BBOX_DICT, batch_size=config.BATCH_SIZE)
        self.batches_per_epoch = len(self.generator)


    def train_Discriminator(self):
        self.GAN.set_discriminator_trainable()
        step = 0
        epoch = 0

        print("STARTING GENERATOR THREADS...............")
        # Start generator threads.
        processes = [multiprocessing.Process(target=batch_populator, args=(batch_queue,self.generator)) for i in
                     range(config.NUM_BATCH_GEN_THREADS)]
        for process in processes:
            process.start()
            time.sleep(5)

        print('GIVE BUFFER SOME TIME......')
        time.sleep(3*60)

        while step < config.NUM_DISCRIMINATOR_STEPS:
            generated_images = self.GAN.get_generated_images(batch_size=32)
            generated_labels = np.random.uniform(low=0.0, high=0.4, size=(len(generated_images), 1))
            print('BATCH_QUEUE_LENGTH:', batch_queue.qsize())
            discriminator_images, discriminator_labels = batch_queue.get()
            image_stack = np.vstack((generated_images, discriminator_images))
            label_stack = np.vstack((generated_labels, discriminator_labels))

            disc_loss = self.GAN.discrimiator.train_on_batch(image_stack, label_stack)
            wandb.log({'disc_initial_loss': disc_loss[0]})

            if not (step % self.batches_per_epoch):
                print("Step {} of {}".format(step, config.NUM_DISCRIMINATOR_STEPS))
                epoch += 1
                disc_save_path = os.path.join(DISCRIMINATOR_SAVE_PATH, 'disc-init-{epoch:04d}.ckpt'.format(epoch=epoch))
                self.GAN.discrimiator.save(disc_save_path)
            step += 1

        print ("STOPPING GENRATOR THREADS.........")
        for process in processes:
            process.terminate()

    def train_GAN(self):

        step = 0
        epoch = 0
        # Start batch threads.
        # Start generator threads.
        print("STARTING GENERATOR THREADS...............")
        processes = [multiprocessing.Process(target=batch_populator, args=(batch_queue,self.generator)) for i in
                     range(config.NUM_BATCH_GEN_THREADS)]

        for process in processes:
            process.start()

        print("TRAINING THE GAN...............")
        while step < config.NUM_TRAINING_STEPS:

            # Train Discriminator

            self.GAN.set_discriminator_trainable()

            generated_images = self.GAN.get_generated_images(batch_size=config.BATCH_SIZE)
            generated_labels = np.random.uniform(low=0.0, high=0.4, size=(len(generated_images), 1))

            discriminator_images, discriminator_labels = batch_queue.get()
            image_stack = np.vstack((generated_images, discriminator_images))
            label_stack = np.vstack((generated_labels, discriminator_labels))
            # Add noise to label_stack by flipping very few labels to reduce mode collapse probability
            for i in np.random.randint(0, len(label_stack), int(config.NOISE_LABEL_PERCENTAGE * len(label_stack))):
                label_stack[i] = np.abs(1 - label_stack[i])

            disc_loss = self.GAN.discrimiator.train_on_batch(image_stack, label_stack)
            wandb.log({'disc_loss': disc_loss[0], 'step': step})

            # Freeze Discriminator and train the adversarial model
            self.GAN.freeze_discriminator_layers()

            noise_to_generate = np.random.rand(config.BATCH_SIZE, config.INPUT_GENERATOR_NOISE_DIM)
            label_stack = np.random.uniform(low=0.0, high=0.4,
                                            size=(config.BATCH_SIZE, 1))

            adv_loss = self.GAN.adversarial.train_on_batch(noise_to_generate, label_stack)
            wandb.log({'adv_loss': adv_loss[0], 'step': step})

            if not (step % self.batches_per_epoch):
                print("Step {} of {}".format(step, config.NUM_TRAINING_STEPS))
                generated_images = self.GAN.get_generated_images(10)
                generated_images = np.uint8(generated_images * 255.0)
                wandb.log(
                    {"examples": [wandb.Image(image, caption=str(idx)) for idx, image in enumerate(generated_images)],
                     "step": step})
                epoch += 1
                discriminator_save_path = os.path.join(DISCRIMINATOR_SAVE_PATH,
                                                       'disc-gan-{epoch:04d}.ckpt'.format(epoch=epoch))
                self.GAN.discrimiator.save(discriminator_save_path)
                generator_save_path = os.path.join(GENERATOR_SAVE_PATH, 'gen-gan-{epoch:04d}.ckpt'.format(epoch=epoch))
                self.GAN.generator.save(generator_save_path)

            step += 1

        print ("STOPPING GENRATOR THREADS.........")
        for process in processes:
            process.terminate()
