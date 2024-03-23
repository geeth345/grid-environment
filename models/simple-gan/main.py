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


# combined loss function
@register_keras_serializable()
def combined_loss(alpha=0.5, beta=0.5):
    def loss(y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        loss = alpha * mae + beta * mse
        return loss

    return loss


class GAN():
    def __init__(self):

        self.input_shape = (28, 28, 1)
        self.latent_dims = 40

        self.masking_function = Mask(walk_length_min=100, walk_length_max=600, visible_radius=1, direction_change_chance=0.7, inverted_mask=False,
                                     add_noise=False)


        # build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
                                   optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        # The generator unet
        self.generator = self.build_generator()
        self.generator.compile(loss='mean_squared_error', optimizer=Adam(0.0008, 0.5))

        print("Generator Summary")
        print(self.generator.summary())
        print("Discriminator Summary")
        print(self.discriminator.summary())



        # The full model (unet + discriminator)
        self.discriminator.trainable = False
        masked_img = Input(shape=self.input_shape)
        mask = Input(shape=self.input_shape)
        gen_img = self.generator([masked_img, mask])
        valid = self.discriminator(gen_img)
        self.full_model = Model([masked_img, mask], valid)
        self.full_model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'],
                                optimizer=Adam(0.0008, 0.5))

        # load the mnist dataset
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        # rescale -1 to 1
        self.X_train = (self.X_train.astype(np.float32) - 127.5) / 127.5
        self.X_train = np.expand_dims(self.X_train, axis=3)
        self.X_test = (self.X_test.astype(np.float32) - 127.5) / 127.5
        self.X_test = np.expand_dims(self.X_test, axis=3)

        ix = np.random.randint(0, self.X_train.shape[0], 250000)
        self.X_train = self.X_train[ix]
        self.y_train = self.y_train[ix]
        self.X_train_masked, self.X_masks = self.masking_function.mask(self.X_train)
        self.X_test_masked, self.X_test_masks = self.masking_function.mask(self.X_test)

        # reshape back to (28, 28, 1)
        self.X_train = self.X_train.reshape(self.X_train.shape[0], 28, 28, 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 28, 28, 1)
        self.X_train_masked = self.X_train_masked.reshape(self.X_train_masked.shape[0], 28, 28, 1)
        self.X_test_masked = self.X_test_masked.reshape(self.X_test_masked.shape[0], 28, 28, 1)
        self.X_masks = self.X_masks.reshape(self.X_masks.shape[0], 28, 28, 1)
        self.X_test_masks = self.X_test_masks.reshape(self.X_test_masks.shape[0], 28, 28, 1)

        # # expose the data to the class
        # self.X_train = X_train
        # self.X_test = X_test
        # self.X_train_masked = X_train_masked
        # self.X_test_masked = X_test_masked
        # self.X_masks = X_masks
        # self.X_test_masks = X_test_masks
        # self.y_test = y_test
        # self.y_train = y_train

        # for generating samples
        # create a list of lists of the indexes of the test set images with each label
        self.test_label_indices = []
        self.masked_img_samples = None
        self.masked_img_masks = None
        for i in range(10):
            self.test_label_indices.append(np.where(self.y_test == i)[0])

        # for model evaluation
        # load a pre-trained CNN model and see how well it performs on the reconstructed images
        self.cnn = load_model('../mnist-cnn/mnist_cnn.h5')
        self.cnn.trainable = False
        self.cnn_accuracies = []

        # for model evaluation
        # evaluate the discriminator's accuracy on the reconstructed images
        self.accuracy_scores = []


    def build_generator(self):

        image = Input(self.input_shape)
        mask = Input(self.input_shape)

        x = concatenate([image, mask])
        x = Flatten()(x)

        # three hidden layers
        x = Dense(500, activation='leaky_relu')(x)
        x = BatchNormalization()(x)
        x = Dense(500, activation='leaky_relu')(x)
        x = BatchNormalization()(x)
        x = Dense(500, activation='leaky_relu')(x)
        x = BatchNormalization()(x)

        # output layer + reshape
        x = Dense(28 * 28, activation='tanh')(x)
        x = Reshape((28, 28, 1))(x)

        gen = Model(inputs=[image, mask], outputs=x)

        return gen



    def build_discriminator(self):

        # Input
        image = Input(self.input_shape)
        x0 = Flatten()(image)

        # classifier
        x1 = Dense(1000, activation='leaky_relu')(x0)
        x1 = GaussianNoise(0.1)(x1)
        x1 = BatchNormalization()(x1)

        x1 = Dense(500, activation='leaky_relu')(x1)
        x1 = GaussianNoise(0.1)(x1)
        x1 = BatchNormalization()(x1)

        x1 = Dense(250, activation='leaky_relu')(x1)
        x1 = GaussianNoise(0.1)(x1)
        x1 = BatchNormalization()(x1)

        x1 = Dense(250, activation='leaky_relu')(x1)
        x1 = GaussianNoise(0.1)(x1)
        x1 = BatchNormalization()(x1)

        x1 = Dense(250, activation='leaky_relu')(x1)
        x1 = GaussianNoise(0.1)(x1)
        x1 = BatchNormalization()(x1)

        x1 = Dense(10, activation='softmax')(x1)
        x1 = GaussianNoise(0.1)(x1)
        x1 = BatchNormalization()(x1)


        # discriminator
        x2 = Concatenate()([x0, x1])
        x2 = Dense(1000, activation='leaky_relu')(x2)
        x2 = GaussianNoise(0.1)(x2)
        x2 = BatchNormalization()(x2)

        x2 = Dense(500, activation='leaky_relu')(x2)
        x2 = GaussianNoise(0.1)(x2)
        x2 = BatchNormalization()(x2)

        x2 = Dense(250, activation='leaky_relu')(x2)
        x2 = GaussianNoise(0.1)(x2)
        x2 = BatchNormalization()(x2)

        x2 = Dense(250, activation='leaky_relu')(x2)
        x2 = GaussianNoise(0.1)(x2)
        x2 = BatchNormalization()(x2)

        x2 = Dense(250, activation='leaky_relu')(x2)
        x2 = GaussianNoise(0.1)(x2)
        x2 = BatchNormalization()(x2)

        x2 = Dense(1, activation='sigmoid')(x2)
        x2 = GaussianNoise(0.1)(x2)
        x2 = BatchNormalization()(x2)

        model = Model(inputs=[image], outputs=[x2, x1])

        return model



    def train(self, epochs, batch_size=128, sample_interval=50):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # select a random batch of images
            idx = np.random.randint(0, self.X_train_masked.shape[0], batch_size)
            masked_imgs = self.X_train_masked[idx]
            imgs = self.X_train[idx]
            masks = self.X_masks[idx]
            labels = self.y_train[idx]

            # generate some images and use them to train the discriminator
            gen_imgs = self.generator.predict([masked_imgs, masks], verbose=0)
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # train the full model (generator and locked discriminator)
            loss2 = self.full_model.train_on_batch([masked_imgs, masks], [valid, labels])

            # print the progress
            print(f"{epoch} [Gen Loss: {loss2[0]}, D Loss Real: {d_loss_real}, D Loss Fake: {d_loss_fake}]")

            # if at save interval, save generated sample images, and save a copy of the model
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.evaluate_using_cnn(epoch)
                self.evaluate_using_discriminator(epoch)
                self.backup_model()

    def sample_images(self, epoch):
        r, c = 10, 10
        if epoch == 0:
            # generate masked images from the test set
            sampled_labels = np.array([num for _ in range(r) for num in range(c)])
            masked_imgs = []
            masks = []
            for label in sampled_labels:
                index = random.choice(self.test_label_indices[label])
                img = self.X_test_masked[index]
                msk = self.X_test_masks[index]
                masked_imgs.append(img)
                masks.append(msk)

            self.masked_img_samples = np.array(masked_imgs)
            self.masked_img_masks = np.array(masks)

            # save a plot of hte masked images for comparison
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

        # generate images using the generator model
        gen_imgs = self.generator.predict([self.masked_img_samples, self.masked_img_masks])

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


    def evaluate_using_discriminator(self, epoch):
        _, generated_images = self.full_model.predict([self.X_test_masked, self.X_test_masks])
        accuracy = np.mean(np.argmax(generated_images, axis=1) == self.y_test)
        self.accuracy_scores.append((accuracy, epoch))

        # save a plot of the accuracy scores so far
        plt.plot([x[1] for x in self.accuracy_scores], [x[0] for x in self.accuracy_scores])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of Discriminator\'s Classifications')
        plt.savefig('images/dsc_accuracy.png')
        plt.close()

    def evaluate_using_cnn(self, epoch):
        generated_images = self.generator.predict([self.X_test_masked, self.X_test_masks])
        classifications = self.cnn.predict(generated_images)
        accuracy = np.mean(np.argmax(classifications, axis=1) == self.y_test)
        self.cnn_accuracies.append((accuracy, epoch))

        # save a plot of the accuracy scores so far
        plt.plot([x[1] for x in self.cnn_accuracies], [x[0] for x in self.cnn_accuracies])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of Pre-Trained CNN on Reconstructed Images')
        plt.savefig('images/cnn_accuracy.png')
        plt.close()

    def backup_model(self):
        self.discriminator.save('saved_model/v2_disc.keras')
        self.generator.save('saved_model/v2_gen.keras')



if __name__ == '__main__':
    model = GAN()
    model.train(epochs=10000, batch_size=100, sample_interval=1000)
