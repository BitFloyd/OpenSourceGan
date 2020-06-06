import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, BatchNormalization, MaxPooling2D, Dropout, GaussianNoise,
    Flatten, Reshape, Activation, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import config


class GAN:

    def __init__(self):
        self.discriminator = None
        self.generator = None
        self.full_model = None

        self.adv_lr = config.INITIAL_ADVERSARIAL_LEARNING_RATE
        self.disc_lr = config.INITIAL_DISCRIMINATOR_LEARNING_RATE
        self.initialize_disc()
        self.initialize_gen()
        self.initialize_adversarial()

    def switch_model(self, kwargs):

        model = None
        if (config.DISCRIMINATOR_BASEMODEL in ['INCEPTION_V3', 'RESNET50', 'INCEPTIONRESNET_V2', 'XCEPTION']):
            if (config.DISCRIMINATOR_BASEMODEL == 'INCEPTION_V3'):
                model = InceptionV3(**kwargs)
            elif (config.DISCRIMINATOR_BASEMODEL == 'RESNET50'):
                model = ResNet50(**kwargs)
            elif (config.DISCRIMINATOR_BASEMODEL == 'INCEPTIONRESNET_V2'):
                model = InceptionResNetV2(**kwargs)
            elif (config.DISCRIMINATOR_BASEMODEL == 'XCEPTION'):
                model = Xception(**kwargs)
        else:
            raise Exception('model_config.BASEMODEL is not defined correctly')

        return model

    def create_base_model(self):
        kwargs = {'weights': 'imagenet', 'include_top': False, 'pooling': None,
                  'input_shape': (config.IMAGE_HEIGHT,
                                  config.IMAGE_WIDTH,
                                  config.IMAGE_CHANNELS)}
        pret_model = self.switch_model(kwargs)
        return pret_model

    def initialize_disc(self):

        pt_model = self.create_base_model()
        mod_out = Conv2D(filters=64, kernel_size=1, padding='same', activation='relu')(pt_model.output)
        mod_out = MaxPooling2D(2)(mod_out)
        mod_out = Flatten()(mod_out)
        mod_out = Dense(32, activation='relu')(mod_out)
        mod_out = Dense(1, activation='sigmoid')(mod_out)

        self.discrimiator = Model(inputs=pt_model.input, outputs=mod_out)
        self.optimizer_disc = Adam(lr=self.disc_lr, clipvalue=config.DISCRIMINATOR_CLIP_VALUE,
                                   decay=config.DISCRIMINATOR_DECAY)

        self.discrimiator.compile(optimizer=self.optimizer_disc, loss='binary_crossentropy', metrics=['accuracy'])
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("DISCRIMINATOR SUMMARY")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        self.discrimiator.summary()

        return self.discrimiator

    def upsample_bank(self, filters, tensor, kernel_shape, upsample_factor):
        tensor = Conv2D(filters, kernel_shape, padding='same')(tensor)
        tensor = BatchNormalization()(tensor)
        tensor = Activation('relu')(tensor)
        tensor = Conv2D(filters, kernel_shape, padding='same')(tensor)
        tensor = BatchNormalization()(tensor)
        tensor = Activation('relu')(tensor)
        tensor = Conv2D(filters, kernel_shape, padding='same')(tensor)
        tensor = BatchNormalization()(tensor)
        tensor = Activation('relu')(tensor)
        tensor = UpSampling2D(upsample_factor)(tensor)

        return tensor

    def initialize_gen(self):
        generator_input_noise = Input(shape=config.INPUT_GENERATOR_NOISE_DIM, )
        gen = Dense(8 * 8 * 256)(generator_input_noise)
        gen = BatchNormalization(momentum=0.0)(gen)
        gen = Activation('relu')(gen)
        gen = Reshape((8, 8, 256))(gen)
        gen = Dropout(0.4)(gen)  # Shape = 8,8,256

        gen = GaussianNoise(0.02)(gen)
        gen = self.upsample_bank(filters=128, tensor=gen, kernel_shape=3, upsample_factor=2)  # Shape = 16,16,128
        gen = GaussianNoise(0.02)(gen)
        gen = self.upsample_bank(filters=64, tensor=gen, kernel_shape=3, upsample_factor=2)  # Shape = 32,32,64
        gen = GaussianNoise(0.02)(gen)
        gen = self.upsample_bank(filters=32, tensor=gen, kernel_shape=3, upsample_factor=2)  # Shape = 64,64,32
        gen = GaussianNoise(0.01)(gen)
        gen = self.upsample_bank(filters=16, tensor=gen, kernel_shape=3, upsample_factor=2)  # Shape = 128,128,16
        gen = GaussianNoise(0.01)(gen)
        gen = self.upsample_bank(filters=8, tensor=gen, kernel_shape=3, upsample_factor=2)  # Shape = 256,256,8

        gen = Conv2D(3, 1, activation='sigmoid')(gen)  # Shape = 256,256,3

        self.generator = Model(inputs=generator_input_noise, outputs=gen)

        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        print("GENERATOR SUMMARY")
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        self.generator.summary()

        return self.generator

    def initialize_adversarial(self):

        adversarial = self.discrimiator(self.generator.output)
        self.adversarial = Model(inputs=self.generator.input, outputs=adversarial)
        self.optimizer_adv = Adam(lr=self.adv_lr, clipvalue=config.ADVERSARIAL_CLIP_VALUE,
                                  decay=config.ADVERSARIAL_DECAY)
        self.adversarial.compile(optimizer=self.optimizer_adv, loss='binary_crossentropy', metrics=['accuracy'])
        return self.adversarial

    def set_discriminator_trainable(self):
        self.discrimiator.trainable = True
        for layer in self.discrimiator.layers:
            layer.trainable = True

    def freeze_discriminator_layers(self):
        self.discrimiator.trainable = False
        for layer in self.discrimiator.layers:
            layer.trainable = False

    def get_generated_images(self, batch_size=config.BATCH_SIZE):

        input_array = np.random.normal(size=(config.BATCH_SIZE, config.INPUT_GENERATOR_NOISE_DIM))
        generated_images = self.generator.predict(input_array)
        return generated_images

    def update_disc_learning_rate(self, decay_factor=0.99):
        self.disc_lr = self.disc_lr * decay_factor
        K.set_value(self.discrimiator.optimizer.learning_rate, self.disc_lr)

        return True

    def update_adv_learning_rate(self, decay_factor=0.99):
        self.adv_lr = self.adv_lr * decay_factor
        K.set_value(self.adversarial.optimizer.learning_rate, self.adv_lr)

        return True
