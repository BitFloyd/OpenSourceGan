import json
import random
import os
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.util.dtype import img_as_float
from tensorflow import keras

import config
import wandb
w_run = wandb.init(project="OpenSourceGAN")


class GANDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, training_dictionary, batch_size=32, shuffle=True):
        """
        training_dictionary: A dict with local image-paths and bboxes
        """
        self.training_dictionary = training_dictionary
        self.keys = list(self.training_dictionary.keys())
        self.num_training_images = len(training_dictionary.keys())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        num_batches = int(np.floor(self.num_training_files / self.batch_size))
        print("NUMBER_OF_BATCHS:", num_batches)
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
            xmin, xmax, ymin, ymax = bbox['XMin'], bbox['XMax'], bbox['YMin'], bbox['YMax']

            image = self.preprocess_frame(imread(filepath))

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

    def preprocess_frame(self, frame):
        frame = img_as_float(frame)
        return frame

    def on_epoch_end(self):
        'Updates indexes after each epoch'
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
        keys = self.IMAGE_BBOX_DICT.keys()

        for key in keys:
            filename = self.IMAGE_BBOX_DICT[key]['filepath']
            try:
                imread(filename)
            except:
                self.IMAGE_BBOX_DICT.pop(key)

        print("{} IMAGES CAN BE READ SUCCESSFULLY AND WILL BE USED FOR TRAINING.............".format(
            len(self.IMAGE_BBOX_DICT)))

        self.generator = GANDataGenerator(self.IMAGE_BBOX_DICT, batch_size=config.BATCH_SIZE)
        self.batches_per_epoch = len(self.generator)

    def train_Discriminator(self):
        self.GAN.set_discriminator_trainable()
        step = 0

        while step < config.NUM_DISCRIMINATOR_STEPS:
            generated_images = self.GAN.get_generated_images(batch_size=32)
            generated_labels = np.random.uniform(low=0.0,high=0.4,size=(len(generated_images),1))

            discriminator_images, discriminator_labels = next(self.generator)
            image_stack = np.vstack((generated_images, discriminator_images))
            label_stack = np.vstack((generated_labels,discriminator_labels))

            disc_loss = self.GAN.discrimiator.train_on_batch(image_stack,label_stack)

            wandb.log({'disc_initial_loss':disc_loss, 'step': step})
            if not (step % len(self.generator)):
                self.generator.on_epoch_end()


    def train_GAN(self):

        step = 0

        while step < config.NUM_TRAINING_STEPS:

            #Train Discriminator

            self.GAN.set_discriminator_trainable()

            generated_images = self.GAN.get_generated_images(batch_size=config.BATCH_SIZE)
            generated_labels = np.random.uniform(low=0.0, high=0.4, size=(len(generated_images), 1))

            discriminator_images, discriminator_labels = next(self.generator)
            image_stack = np.vstack((generated_images, discriminator_images))
            label_stack = np.vstack((generated_labels, discriminator_labels))
            # Add noise to label_stack by flipping very few labels to reduce mode collapse probability
            for i in np.random.randint(0, len(label_stack), int(config.NOISE_LABEL_PERCENTAGE * len(label_stack))):
                label_stack[i] = np.abs(1 - label_stack[i])

            disc_loss = self.GAN.discrimiator.train_on_batch(image_stack, label_stack)

            wandb.log({'disc_loss': disc_loss, 'step': step})

            #Freeze Discriminator and train the adversarial model
            self.GAN.freeze_discriminator_layers()

            generated_images = self.GAN.get_generated_images(batch_size=config.BATCH_SIZE)
            generated_labels = np.random.uniform(low=0.0, high=0.4, size=(len(generated_images), 1))

            discriminator_images, discriminator_labels = next(self.generator)
            image_stack = np.vstack((generated_images, discriminator_images))
            label_stack = np.vstack((generated_labels, discriminator_labels))

            adv_loss = self.GAN.adversarial.train_on_batch(image_stack, label_stack)

            wandb.log({'adv_loss': adv_loss, 'step': step})


            if not (step % len(self.generator)):
                self.generator.on_epoch_end()




