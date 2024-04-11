# keras imports
from functools import partial

from keras.layers import Input, Dense, MaxPooling2D, Conv2D, LeakyReLU, Concatenate, Reshape
from keras.layers import Conv2DTranspose, Flatten, UpSampling2D, Activation, BatchNormalization
from keras.layers import GaussianNoise, Dropout, Lambda, Layer
from keras.models import Model, load_model
from keras.optimizers.legacy import Adam, RMSprop
from keras.models import Model
from keras import backend as K
import tensorflow as tf

# other module imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

class CriticModel(Model):

    def __init__(self, critic, generator, **kwargs):
        super(CriticModel, self).__init__(**kwargs)
        self.critic = critic
        self.generator = generator

    def compute_gradient_penalty(self, real_images, fake_images):
        """Calculates the gradient penalty for a batch of real and fake images."""
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform([batch_size, 1, 1, 1], 0., 1.)
        interpolated_images = real_images * alpha + fake_images * (1 - alpha)

        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            prediction = self.critic(interpolated_images, training=True)

        gradients = tape.gradient(prediction, [interpolated_images])[0]
        gradients_sqr = tf.square(gradients)
        gradients_sqr_sum = tf.reduce_sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = tf.sqrt(gradients_sqr_sum)
        gradient_penalty = tf.reduce_mean(tf.square(1 - gradient_l2_norm))
        return gradient_penalty

    def train_step(self, data):
        real_images, masks, masked_images = data

        with tf.GradientTape() as tape:
            fake_images = self.generator([masked_images, masks], training=True)
            real_score = self.critic(real_images, training=True)
            fake_score = self.critic(fake_images, training=True)
            gradient_penalty = self.compute_gradient_penalty(real_images, fake_images)
            # WGAN-GP loss
            critic_loss = tf.reduce_mean(fake_score) - tf.reduce_mean(
                real_score) + 10.0 * gradient_penalty  # Lambda is 10 for gradient penalty

        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        return {"critic_loss": critic_loss}


class UNet():

    def __init__(self):

        self.input_shape = (28, 28, 1)
        self.data_source = '../../data/masked100_600_0.7.npz'

        # build the generator and critic models
        self.generator = self.build_generator()
        self.generator.compile()

        self.critic = self.build_critic()

        ############################
        # compile the critic model #
        ############################

        self.generator.trainable = False

        image = Input(self.input_shape)
        mask = Input(self.input_shape)
        masked_image = Input(self.input_shape)

        self.critic_model = CriticModel(self.critic, self.generator)
        self.critic_model.build(image, mask, masked_image)
        self.critic_model.compile(optimizer=RMSprop(0.00005))


        ###############################
        # compile the generator model #
        ###############################

        self.critic.trainable = False
        self.generator.trainable = True

        # build the combined model (generator with adversarial loss for training)
        masked_image = Input(shape=self.input_shape)
        mask = Input(shape=self.input_shape)
        generated_image = self.generator([masked_image, mask])
        validity = self.critic(generated_image)
        self.combined = Model([masked_image, mask], [validity])
        self.combined.compile(loss=self.wasserstein_loss, optimizer=RMSprop(0.00005), metrics=['accuracy'])


        print("Generator Summary")
        print(self.generator.summary())
        print("Critic Summary")
        print(self.critic.summary())
        print("Critic Model Summary")
        print(self.critic_model.summary())
        print("Combined Model Summary")
        print(self.combined.summary())

        # load the dataset from the file
        data = np.load('../../data/masked100_600_0.7.npz')
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        self.X_train_masked = data['X_train_masked']
        self.X_masks = data['X_masks']
        self.X_test_masked = data['X_test_masked']
        self.X_test_masks = data['X_test_masks']

        self.metrics_file = open('metrics.csv', 'w')

        # for generating samples
        # create a list of lists of the indexes of the test set images with each label
        self.test_label_indices = []
        self.masked_img_samples = None
        self.masked_img_masks = None
        for i in range(10):
            self.test_label_indices.append(np.where(self.y_test == i)[0])

        # for model evaluation
        # load a pre-trained CNN model and see how well it performs on the reconstructed images
        self.cnn = load_model('../_classifier/mnist_cnn.h5')
        self.cnn.trainable = False
        self.cnn_accuracies = []

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def compute_gradient_penalty(self, y_true, y_pred, averaged_samples):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)


    def build_critic(self):

        image = Input(self.input_shape)
        x = Conv2D(16, kernel_size=3, strides=2, input_shape=self.input_shape, padding="same")(image)
        x = GaussianNoise(0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.2)(x)

        x = Conv2D(32, kernel_size=3, strides=2, padding="same")(x)
        x = GaussianNoise(0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.2)(x)

        x = BatchNormalization(momentum=0.8)(x)
        x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
        x = GaussianNoise(0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.2)(x)

        x_flat = Flatten()(x)

        validity_dense = Dense(128)(x_flat)
        validity_dense = LeakyReLU(alpha=0.2)(validity_dense)
        validity = Dense(1, name='validity')(validity_dense)

        model = Model(inputs=[image], outputs=[validity])

        return model

    def build_generator(self):

        image = Input(shape=self.input_shape)
        mask = Input(shape=self.input_shape)

        e1_image = Conv2D(8, kernel_size=(4, 4), strides=(1, 1), padding='same')(image)
        e1_mask = Conv2D(8, kernel_size=(4, 4), strides=(1, 1), padding='same')(mask)
        e1 = Concatenate()([e1_image, e1_mask])
        e1 = BatchNormalization()(e1)
        e1 = LeakyReLU(alpha=0.2)(e1)

        e2 = Conv2D(16, kernel_size=(4, 4), strides=(2, 2), padding='same')(e1)
        e2 = BatchNormalization()(e2)
        e2 = LeakyReLU(alpha=0.2)(e2)

        e3 = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), padding='same')(e2)
        e3 = BatchNormalization()(e3)
        e3 = LeakyReLU(alpha=0.2)(e3)

        e4 = Dense(100)(Flatten()(e3))
        e4 = BatchNormalization()(e4)
        e4 = LeakyReLU(alpha=0.2)(e4)

        lr = Dense(64)(e4)
        lr = BatchNormalization()(lr)
        lr = LeakyReLU(alpha=0.2)(lr)

        d4 = Dense(196)(lr)
        d4 = BatchNormalization()(d4)
        d4 = LeakyReLU(alpha=0.2)(d4)

        d3 = Reshape((7, 7, 4))(d4)
        d3 = Conv2DTranspose(32, kernel_size=(4, 4), strides=(1, 1), padding='same')(d3)
        d3 = BatchNormalization()(d3)
        d3 = LeakyReLU(alpha=0.2)(d3)

        # d2 = UpSampling2D(size=(2, 2))(d3)
        # d2 = Concatenate()([d2, e2])
        # d2 = Conv2DTranspose(16, kernel_size=(4, 4), strides=(1, 1), padding='same')(d2)
        # d2 = LeakyReLU(alpha=0.2)(d2)

        d2 = Conv2DTranspose(16, kernel_size=(4, 4), strides=(2, 2), padding='same')(d3)
        d2 = BatchNormalization()(d2)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = Concatenate()([d2, e2])

        # d1 = UpSampling2D(size=(2, 2))(d2)
        # d1 = Concatenate()([d1, e1])
        # d1 = Conv2DTranspose(8, kernel_size=(4, 4), strides=(1, 1), padding='same')(d1)
        # d1 = LeakyReLU(alpha=0.2)(d1)

        d1 = Conv2DTranspose(8, kernel_size=(4, 4), strides=(2, 2), padding='same')(d2)
        d1 = BatchNormalization()(d1)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d1 = Concatenate()([d1, e1])

        d0 = Conv2DTranspose(4, kernel_size=(4, 4), strides=(1, 1), padding='same')(d1)
        d0 = BatchNormalization()(d0)
        d0 = LeakyReLU(alpha=0.2)(d0)

        d0 = Conv2DTranspose(2, kernel_size=(4, 4), strides=(1, 1), padding='same')(d0)
        d0 = BatchNormalization()(d0)
        d0 = LeakyReLU(alpha=0.2)(d0)

        d0 = Conv2DTranspose(1, kernel_size=(4, 4), strides=(1, 1), padding='same')(d0)

        output = Activation('tanh')(d0)

        model = Model(inputs=[image, mask], outputs=output)

        return model

    def train(self, epochs, batch_size, sample_interval):

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))
        dummy = np.zeros((batch_size, 1))

        for epoch in tqdm(range(epochs)):

            ########################
            # Prepare Data         #
            ########################

            idx = np.random.randint(0, self.X_train.shape[0], batch_size)
            images = self.X_train[idx]
            masked_images = self.X_train_masked[idx]
            masks = self.X_masks[idx]
            labels = self.y_train[idx]

            ########################
            # Train Critic         #
            ########################

            # generate a batch of images using the generator
            gen_images = self.generator.predict([masked_images, masks], verbose=0)

            d_loss = self.critic_model.train_on_batch([images, masks, masked_images], [valid, fake, dummy])

            #########################
            # Train Generator       #
            #########################

            g_loss = self.combined.train_on_batch([masked_images, masks], valid)

            #########################
            # Record Progress       #
            #########################

            print(f"{epoch} [Generator Loss: {g_loss}] [Critic Loss: {d_loss}]")
            # TODO: write to metrics file
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.evaluate_using_cnn(epoch)
                self.backup_model(epoch)

    def sample_images(self, epoch):
        r, c = 10, 10
        if epoch == 0:
            # generate masked images from the test set
            sampled_labels = np.array([num for _ in range(r) for num in range(c)])
            imgs = []
            masked_imgs = []
            masks = []
            for label in sampled_labels:
                index = random.choice(self.test_label_indices[label])
                maskedimg = self.X_test_masked[index]
                msk = self.X_test_masks[index]
                imgs.append(self.X_test[index])
                masked_imgs.append(maskedimg)
                masks.append(msk)

            self.masked_img_samples = np.array(masked_imgs)
            self.masked_img_masks = np.array(masks)
            imgs = np.array(imgs)

            # save a plot of the masked sample images for comparison
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(self.masked_img_samples[cnt, :, :], interpolation='nearest')
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("images/0_masked.png")
            plt.close()
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(self.masked_img_masks[cnt, :, :], interpolation='nearest')
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("images/0_masks.png")
            plt.close()
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(imgs[cnt, :, :], interpolation='nearest')
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("images/0_images.png")
            plt.close()

        # generate images using the generator model
        gen_imgs = self.generator.predict([self.masked_img_samples, self.masked_img_masks], verbose=0)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], interpolation='nearest')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def evaluate_using_cnn(self, epoch):
        generated_images = self.generator.predict([self.X_test_masked, self.X_test_masks], verbose=0)
        classifications = self.cnn.predict(generated_images, verbose=0)
        accuracy = np.mean(np.argmax(classifications, axis=1) == self.y_test)
        self.cnn_accuracies.append((accuracy, epoch))

        # save a plot of the accuracy scores so far
        plt.plot([x[1] for x in self.cnn_accuracies], [x[0] for x in self.cnn_accuracies])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of Pre-Trained CNN on Reconstructed Images')
        plt.savefig('images/cnn_accuracy.png')
        plt.close()

    def backup_model(self, epoch):
        self.generator.save(f'saved_model/gen.keras')
        self.critic.save(f'saved_model/disc.keras')


if __name__ == '__main__':
    unet = UNet()
    unet.train(epochs=3001, batch_size=64, sample_interval=200)
