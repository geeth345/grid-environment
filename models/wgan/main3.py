from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D, Layer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow import random as tf_random
import tensorflow as tf
from functools import partial
import numpy as np
import matplotlib.pyplot as plt

class RandomWeightedAverage(Layer):
    """Provides a (random) weighted average between real and generated image samples"""
    def call(self, inputs, **kwargs):
        alpha = tf_random.uniform((K.shape(inputs[0])[0], 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

class WGANGP(Model):
    def __init__(self):
        super(WGANGP, self).__init__()
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        self.n_critic = 5
        optimizer = RMSprop(learning_rate=0.00005)

        self.generator = self.build_generator()
        self.critic = self.build_critic()

        self.generator.trainable = False

        real_img = Input(shape=self.img_shape)
        z_disc = Input(shape=(self.latent_dim,))
        fake_img = self.generator(z_disc)

        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        interpolated_img = RandomWeightedAverage()([real_img, fake_img])
        validity_interpolated = self.critic(interpolated_img)

        partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty'

        self.critic_model = Model(inputs=[real_img, z_disc], outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss, self.wasserstein_loss, partial_gp_loss],
                                  optimizer=optimizer, loss_weights=[1, 1, 10])

        self.critic.trainable = False
        self.generator.trainable = True

        z_gen = Input(shape=(self.latent_dim,))
        img = self.generator(z_gen)
        valid = self.critic(img)
        self.generator_model = Model(z_gen, valid)
        self.generator_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)

    # Define a custom train_step
    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and on what you pass to `fit()`.
        real_imgs, noise = data

        # Constants for the training process
        batch_size = tf.shape(real_imgs)[0]
        valid = -tf.ones((batch_size, 1))
        fake = tf.ones((batch_size, 1))
        dummy = tf.zeros((batch_size, 1))  # Dummy gt for gradient penalty

        with tf.GradientTape() as tape:
            # Generate fake images
            fake_imgs = self.generator(noise, training=True)
            # Discriminator decisions for real and fake images
            logits_real = self.critic(real_imgs, training=True)
            logits_fake = self.critic(fake_imgs, training=True)

            # Calculate the discriminator loss using real and fake images
            d_loss_real = self.wasserstein_loss(valid, logits_real)
            d_loss_fake = self.wasserstein_loss(fake, logits_fake)

            # Calculate the gradient penalty
            with tf.GradientTape() as gp_tape:
                alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
                interpolated = alpha * real_imgs + (1 - alpha) * fake_imgs
                gp_tape.watch(interpolated)
                pred = self.critic(interpolated, training=True)
            grads = gp_tape.gradient(pred, [interpolated])[0]
            norm_grads = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            gradient_penalty = tf.reduce_mean((norm_grads - 1.) ** 2)
            d_loss = d_loss_real + d_loss_fake + 10.0 * gradient_penalty

        # Compute discriminator gradients
        d_gradients = tape.gradient(d_loss, self.critic.trainable_variables)

        # Update discriminator weights
        self.optimizer.apply_gradients(zip(d_gradients, self.critic.trainable_variables))

        return {'d_loss': d_loss}


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """Computes the gradient penalty loss for a batch of "averaged" samples."""
        with tf.GradientTape() as tape:
            # Compute the gradients of the predictions with respect to the input.
            tape.watch(averaged_samples)
            prediction = self.critic(averaged_samples)
        gradients = tape.gradient(prediction, averaged_samples)

        # Compute the euclidean norm by squaring...
        gradients_sqr = tf.square(gradients)
        # Summing over the rows...
        gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        # and taking the square root...
        gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
        # Compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = tf.square(1 - gradient_l2_norm)
        # Return the mean as loss over all the batch samples
        return tf.reduce_mean(gradient_penalty)


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):
        model = Sequential()
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=4, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(self.channels, kernel_size=4, padding="same"))
        model.add(Activation("tanh"))
        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def build_critic(self):
        model = Sequential()
        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def train(self, epochs, batch_size=32, sample_interval=50):
        (X_train, _), (_, _) = mnist.load_data()
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty
        for epoch in range(epochs):
            for _ in range(self.n_critic):
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]
                noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
                d_loss = self.critic_model.train_on_batch([imgs, noise], [valid, fake, dummy])
            g_loss = self.generator_model.train_on_batch(noise, valid)
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    wgan = WGANGP()
    wgan.train(epochs=30000, batch_size=32, sample_interval=100)
