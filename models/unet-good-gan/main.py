from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Add
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import LeakyReLU
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


class U_Net():
    def __init__(self):

        self.input_shape = (28, 28, 1)
        self.latent_dims = 40

        self.masking_function = Mask(walk_length_min=100, walk_length_max=600, visible_radius=1, direction_change_chance=0.7, inverted_mask=False,
                                     add_noise=False)

        self.f = open('progress.csv', 'w')


        # Build and compile the generator
        self.generator = self.build_unet()
        self.generator.compile(loss='mean_squared_error', optimizer=Adam(0.0008, 0.5))

        # build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy'], optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        # build and compile the classifier
        self.classifier = self.build_classifier()
        self.classifier.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])


        print("Generator Summary")
        print(self.generator.summary())
        print("Discriminator Summary")
        print(self.discriminator.summary())
        print("Classifier Summary")
        print(self.classifier.summary())



        # The full model (unet + discriminator)
        self.discriminator.trainable = False
        masked_img = Input(shape=self.input_shape)
        mask = Input(shape=self.input_shape)
        gen_img = self.generator([masked_img, mask])
        valid = self.discriminator(gen_img)
        self.full_model_g = Model([masked_img, mask], valid)
        self.full_model_g.compile(loss=['binary_crossentropy'], optimizer=Adam(0.0008, 0.5), metrics=['accuracy'])

        # The full model (unet + classifier)
        self.classifier.trainable = False
        masked_img = Input(shape=self.input_shape)
        mask = Input(shape=self.input_shape)
        gen_img = self.generator([masked_img, mask])
        label = self.classifier(gen_img)
        self.full_model_c = Model([masked_img, mask], label)
        self.full_model_c.compile(loss=['sparse_categorical_crossentropy'], optimizer=Adam(0.0008, 0.5), metrics=['accuracy'])


        # load the mnist dataset
        (self.X_train, self.y_train), (self.X_test, self.y_test) = mnist.load_data()

        # rescale -1 to 1
        self.X_train = (self.X_train.astype(np.float32) - 127.5) / 127.5
        self.X_train = np.expand_dims(self.X_train, axis=3)
        self.X_test = (self.X_test.astype(np.float32) - 127.5) / 127.5
        self.X_test = np.expand_dims(self.X_test, axis=3)

        ix = np.random.randint(0, self.X_train.shape[0], 120000)
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


    def build_unet(self):

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



    def build_discriminator(self):

        # Input
        image = Input(self.input_shape)
        x = Conv2D(16, kernel_size=3, strides=2, input_shape=self.input_shape, padding="same")(image)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = Conv2D(32, kernel_size=3, strides=2, padding="same")(x)
        x = ZeroPadding2D(padding=((0, 1), (0, 1)))(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x = BatchNormalization(momentum=0.8)(x)
        x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.25)(x)
        x_flat = Flatten()(x)

        validity_dense = Dense(128)(x_flat)
        validity_dense = LeakyReLU(alpha=0.2)(validity_dense)
        validity = Dense(1, activation='sigmoid', name='validity')(validity_dense)

        model = Model(inputs=[image], outputs=[validity])

        return model


    def build_classifier(self):
        image = Input(self.input_shape)
        x = Conv2D(32, kernel_size=(3, 3), activation="leaky_relu")(image)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, kernel_size=(3, 3), activation="leaky_relu")(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dropout(0.5)(x)
        x = Dense(10, activation="softmax")(x)
        model = Model(inputs=[image], outputs=[x])

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

            # generate some images using the generator
            gen_imgs = self.generator.predict([masked_imgs, masks], verbose=0)

            # train the discriminator on real and fake images
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # train the classifier on both real and fake images
            c_loss_real = self.classifier.train_on_batch(imgs, labels)
            c_loss_fake = self.classifier.train_on_batch(gen_imgs, labels)
            c_loss = 0.5 * np.add(c_loss_real, c_loss_fake)

            # train the generator using the loss of the discriminator and classifier
            g_loss_disc = self.full_model_g.train_on_batch([masked_imgs, masks], valid)
            g_loss_clas = self.full_model_c.train_on_batch([masked_imgs, masks], labels)


            # write the progress to a csv file
            if epoch == 0:
                self.f.write("epoch,d_loss,d_acc,c_loss,c_acc,g_loss_disc,g_acc_disc,g_loss_clas,g_acc_clas\n")
            self.f.write(f"{epoch},{d_loss[0]},{d_loss[1]},{c_loss[0]},{c_loss[1]},{g_loss_disc[0]},{g_loss_disc[1]},{g_loss_clas[0]},{g_loss_clas[1]}\n")

            # summarise progress to terminal
            print(f"{epoch} [D Loss: {d_loss}, C Loss: {c_loss}, G Loss Disc: {g_loss_disc}, G Loss Clas: {g_loss_clas}]")

            # if at save interval, save generated sample images, and save a copy of the model
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.evaluate_using_cnn(epoch)
                self.evaluate_using_discriminator(epoch)
                self.backup_model()

        self.f.close()

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
        image_classes = self.full_model_c.predict([self.X_test_masked, self.X_test_masks])
        accuracy = np.mean(np.argmax(image_classes, axis=1) == self.y_test)
        self.accuracy_scores.append((accuracy, epoch))

        # save a plot of the accuracy scores so far
        plt.plot([x[1] for x in self.accuracy_scores], [x[0] for x in self.accuracy_scores])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy of Auxiliary Classifier')
        plt.savefig('images/cls_accuracy.png')
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
        self.discriminator.save('saved_model/v1_dis.keras')
        self.generator.save('saved_model/v1_gen.keras')
        self.classifier.save('saved_model/v1_clas.keras')


if __name__ == '__main__':
    model = U_Net()
    model.train(epochs=2001, batch_size=64, sample_interval=100)
