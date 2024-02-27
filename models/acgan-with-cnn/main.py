# code source: https://github.com/eriklindernoren/Keras-GAN/blob/master/acgan/acgan.py paper: Cheng, Keyang,
# Rabia Tahir, Lubamba Kasangu Eric, and Maozhen Li, ‘An Analysis of Generative Adversarial Networks and Variants for
# Image Synthesis on MNIST Dataset’, Multimedia Tools and Applications, 79.19 (2020), 13725–52
# <https://doi.org/10.1007/s11042-019-08600-2>

from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate
from keras.models import Sequential, Model, load_model
from keras.optimizers.legacy import Adam

import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import numpy as np

# goofy import
import sys
sys.path.append('../processing')
from mask import Mask



class ACGAN():
    def __init__(self):
        # Input shape
        self.masked_img_samples = np.array([])
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.num_classes = 10
        self.latent_dim = 100
        
        self.masking_function = Mask(visible_radius=1, direction_change_chance=0.7, inverted_mask=False, add_noise=True)


        optimizer = Adam(0.0008, 0.5)
        optimizer2 = Adam(0.0004, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes a masked image as input
        masked_image = Input(shape=self.img_shape)
        img = self.generator(masked_image)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(masked_image, [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer2)


        # load the cnn model
        self.cnn = load_model('../mnist-cnn/mnist_cnn.h5')

        # create a model that combines the cnn with the generator
        # using the cnn to classify the generated images
        self.cnn.trainable = False

        # the cnn takes the generated image as input and outputs a classification
        cnn_class = self.cnn(img)
        self.combined_with_cnn = Model(masked_image, cnn_class)
        self.combined_with_cnn.compile(loss='sparse_categorical_crossentropy',
            optimizer=optimizer2)






        # load the mnist dataset
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


        # create a normalised and masked version of the test set
        print('Creating masked test set')
        self.X_test = (self.X_test.astype(np.float32) - 127.5) / 127.5
        self.X_test_masked = np.array([self.masking_function.mask(img) for img in tqdm(self.X_test)])


        # create a list of lists of the indexes of the test set images with each label
        self.test_label_indices = []
        for i in range(10):
            self.test_label_indices.append(np.where(y_test == i)[0])

        # for evaluation
        self.accuracy_scores = []
        self.recall_scores = []
        self.classification_scores = []

        # load the cnn model
        classifications = self.cnn.predict(self.X_test_masked)
        accuracy = np.sum(np.argmax(classifications, axis=1) == self.y_test) / len(self.y_test)
        self.cnn_baseline = accuracy


    def build_generator(self):
        # Encoder
        inputs = Input(shape=self.img_shape)

        # Encoder: Downsample 1
        conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = BatchNormalization(momentum=0.8)(conv1)
        conv1 = Activation("relu")(conv1)
        conv1_pool = MaxPooling2D(pool_size=(2, 2))(conv1)

        # Encoder: Downsample 2
        conv2 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv1_pool)
        conv2 = BatchNormalization(momentum=0.8)(conv2)
        conv2 = Activation("relu")(conv2)
        conv2_pool = MaxPooling2D(pool_size=(2, 2))(conv2)

        # Bottleneck
        conv3 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv2_pool)
        conv3 = BatchNormalization(momentum=0.8)(conv3)
        conv3 = Activation("relu")(conv3)

        # Decoder: Upsample 1
        up1 = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(conv3)
        up1 = concatenate([up1, conv2], axis=3)
        conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
        conv4 = BatchNormalization(momentum=0.8)(conv4)
        conv4 = Activation("relu")(conv4)

        # Decoder: Upsample 2
        up2 = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(conv4)
        up2 = concatenate([up2, conv1], axis=3)
        conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
        conv5 = BatchNormalization(momentum=0.8)(conv5)
        conv5 = Activation("relu")(conv5)

        # Output layer
        output_img = Conv2D(self.channels, (3, 3), activation='tanh', padding='same')(conv5)

        return Model(inputs, output_img)


    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.summary()

        img = Input(shape=self.img_shape)

        # Extract feature representation
        features = model(img)

        # Determine validity and label of the image
        validity = Dense(1, activation="sigmoid")(features)
        label = Dense(self.num_classes, activation="softmax")(features)

        return Model(img, [validity, label])

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        X_train = self.X_train
        y_train = self.y_train

        # Configure inputs
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Generating masked images for training the generator
            masked_imgs = np.array([self.masking_function.mask(img) for img in imgs])

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(masked_imgs, verbose=0)

            # Image labels. 0-9
            img_labels = y_train[idx]

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, img_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------



            # Train the generator using both the masked images and the labels, as well as the cnn loss
            g_loss = self.combined.train_on_batch(masked_imgs, [valid, img_labels])
            g_loss_cnn = self.combined_with_cnn.train_on_batch(masked_imgs, img_labels)

            g_loss = 0.5 * np.add(g_loss, g_loss_cnn)


            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_model()
                self.sample_images(epoch)
                self.test_masked_accuracy(epoch)


    def test_masked_accuracy(self, epoch):
        # evaluate the combined model on the masked test set, looking at the accuracy of the labels produced by the discriminator
        # get classifications
        classifications = self.combined.predict(self.X_test_masked)[1]
        # get the predicted labels
        predicted_labels = np.argmax(classifications, axis=1)

        accuracy = np.sum(predicted_labels == self.y_test) / len(self.y_test)
        self.accuracy_scores.append((accuracy, epoch))


        # save a plot of the accuracy scores
        plt.plot([score[1] for score in self.accuracy_scores], [score[0] for score in self.accuracy_scores])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of the discriminator on generated images')
        plt.savefig('images/accuracy.png')
        plt.close()


        # use predicted labels to calculate recall for each class
        recalls = []
        for i in range(10):
            # get the indices of the test images with the current label
            indices = np.where(self.y_test == i)[0]
            predicted_labels = np.argmax(classifications[indices], axis=1)
            recall = np.sum(predicted_labels == i) / len(indices)
            recalls.append(recall)

        self.recall_scores.append((recalls, epoch))

        # save a plot of the recall scores, one line for each class
        for i in range(10):
            plt.plot([score[1] for score in self.recall_scores], [score[0][i] for score in self.recall_scores])
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.title('Recall for each class')
        plt.savefig('images/recall.png')
        plt.close()

        # calculate the classification score
        generated_images = self.generator.predict(self.X_test_masked)
        classifications = self.cnn.predict(generated_images)
        accuracy = np.sum(np.argmax(classifications, axis=1) == self.y_test) / len(self.y_test)
        self.classification_scores.append((accuracy, epoch))
        baselines = [(self.cnn_baseline, epoch) for (_, epoch) in self.classification_scores]

        # save a plot of the classification scores, including the baseline
        plt.plot([score[1] for score in self.classification_scores], [score[0] for score in self.classification_scores])
        plt.plot([score[1] for score in baselines], [score[0] for score in baselines])
        plt.xlabel('Epoch')
        plt.ylabel('Classification Accuracy')
        plt.title('Classification Accuracy of a CNN on Generated Images')
        plt.savefig('images/classification.png')
        plt.close()


    def sample_images(self, epoch):
        r, c = 10, 10

        if epoch == 0:
            # generate masked images from the test set
            sampled_labels = np.array([num for _ in range(r) for num in range(c)])
            masked_imgs = []
            for label in sampled_labels:
                index = random.choice(self.test_label_indices[label])
                img = self.X_test_masked[index]
                masked_imgs.append(img)

            self.masked_img_samples = np.array(masked_imgs)

            # save a plot of hte masked images for comparison
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(self.masked_img_samples[cnt, :, :])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("images/0_masked.png")

        gen_imgs = self.generator.predict(self.masked_img_samples)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,0])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "generator")
        save(self.discriminator, "discriminator")



# old masking function


# # parameters
# RANDOM_WALK_LENGTH = 40
# VISIBLE_RADIUS = 5
#
# # given a 28*28 numpy array, apply a mask and return the masked image
# def mask(image, walk_length=RANDOM_WALK_LENGTH, visible_radius=VISIBLE_RADIUS):
#
#     if len(image.shape) == 3:
#         image = image[:,:,0]
#
#
#     # assume the image is a 28x28 numpy array
#     assert image.shape == (28, 28)
#
#
#     # randomly select a starting position
#     agent_position = (random.randint(7, 20), random.randint(7, 20))
#     random_walk_steps = [random.randint(0, 3) for _ in range(RANDOM_WALK_LENGTH)]
#
#     # create the mask
#     mask = np.zeros((28, 28))
#     for i in range(RANDOM_WALK_LENGTH):
#         # update the agent position
#         if random_walk_steps[i] == 0:
#             agent_position = (agent_position[0] - 1, agent_position[1])
#         elif random_walk_steps[i] == 1:
#             agent_position = (agent_position[0], agent_position[1] + 1)
#         elif random_walk_steps[i] == 2:
#             agent_position = (agent_position[0] + 1, agent_position[1])
#         elif random_walk_steps[i] == 3:
#             agent_position = (agent_position[0], agent_position[1] - 1)
#
#         # update the mask
#         for x in range(agent_position[0] - VISIBLE_RADIUS, agent_position[0] + VISIBLE_RADIUS + 1):
#             for y in range(agent_position[1] - VISIBLE_RADIUS, agent_position[1] + VISIBLE_RADIUS + 1):
#                 if 0 <= x < 28 and 0 <= y < 28:
#                     mask[x][y] = 1
#
#     # apply the mask to the image, but invisible pixels are random values between 0 and 1
#     noise = np.clip(np.random.normal(0.0, 0.2, (28, 28)), -1, 1)
#     #noise = np.zeros((28, 28))
#
#
#     masked_image = np.where(mask == 1, image, noise)
#
#     #
#     # # display all three arrays for debugging
#     # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#     # ax1.imshow(image, cmap='gray')
#     # ax2.imshow(mask, cmap='gray')
#     # ax3.imshow(masked_image, cmap='gray')
#     # plt.show()
#
#     return masked_image
#


if __name__ == '__main__':
    acgan = ACGAN()
    acgan.train(epochs=2600, batch_size=32, sample_interval=200)