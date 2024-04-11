# Credit To: https://github.com/eriklindernoren/Keras-GAN/blob/master/wagn_gp/wgan_gp.py paper: Cheng, Keyang,
# Rabia Tahir, Lubamba Kasangu Eric, and Maozhen Li, ‘An Analysis of Generative Adversarial Networks and Variants for

import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import LeakyReLU, GaussianNoise
from keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate
from keras.models import Sequential, Model, load_model
from keras.optimizers.legacy import Adam
from keras.losses import mean_squared_error, mean_absolute_error

import keras.backend as K

import numpy as np

import matplotlib.pyplot as plt
from keras.src.layers import concatenate
from keras.src.saving.object_registration import register_keras_serializable
from tqdm import tqdm
import random

# goofy import
import sys

sys.path.append('../processing')
from mask2 import Mask




class WGAN():
    def __init__(self):
        pass

    def build_generator(self):
        image = Input(self.input_shape)
        mask = Input(self.input_shape)

        # "Encoder" part of the model
        e1i = Conv2D(8, kernel_size=4, strides=1, padding='same')(image)
        e1m = Conv2D(4, kernel_size=4, strides=1, padding='same')(mask)
        e1 = Concatenate()([e1i, e1m])
        e1 = LeakyReLU(alpha=0.2)(e1)

        e2 = Conv2D(16, kernel_size=4, strides=2, padding='same')(e1)
        e2 = LeakyReLU(alpha=0.2)(e2)
        e2 = Dropout(0.1)(e2)

        e3 = Conv2D(32, kernel_size=4, strides=2, padding='same')(e2)
        e3 = LeakyReLU(alpha=0.2)(e3)
        e3 = Dropout(0.1)(e3)

        e4 = Dense(256)(Flatten()(e3))
        e4 = LeakyReLU(alpha=0.2)(e4)

        # latent representation
        lr = Dense(49)(e4)
        lr = LeakyReLU(alpha=0.2)(lr)
        lr = Reshape((7, 7, 1))(lr)

        # "Decoder" part of the model
        d1 = Conv2DTranspose(32, kernel_size=4, strides=2, padding='same')(lr)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d1 = Dropout(0.1)(d1)

        # skip connection
        d1 = concatenate([d1, e2])

        d2 = Conv2DTranspose(16, kernel_size=4, strides=2, padding='same')(d1)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = Dropout(0.1)(d2)

        d3 = Conv2DTranspose(8, kernel_size=4, strides=1, padding='same')(d2)
        d3 = LeakyReLU(alpha=0.2)(d3)

        # skip connection
        d3 = concatenate([d3, e1])

        d4 = Conv2D(32, kernel_size=4, strides=2, padding='same')(d3)
        d4 = LeakyReLU(alpha=0.2)(d4)

        d5 = Conv2DTranspose(16, kernel_size=4, strides=2, padding='same')(d4)
        d5 = LeakyReLU(alpha=0.2)(d5)

        d4 = Conv2DTranspose(1, kernel_size=4, strides=1, padding='same')(d3)
        d4 = Activation('tanh')(d4)

        unet_model = Model(inputs=[image, mask], outputs=d4)

        return unet_model

    def build_critic(self):
        pass

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):

