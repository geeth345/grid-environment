from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate
from keras.models import Sequential, Model, load_model
from keras.optimizers.legacy import Adam

import numpy as np

import matplotlib.pyplot as plt
from tqdm import tqdm
import random


# goofy import
import sys
sys.path.append('../processing')
from mask2 import Mask


class Autoencoder():
    def __init__(self):

        self.input_shape = (28, 28, 1)
        self.latent_dims = 40

        self.masking_function = Mask(visible_radius=1, direction_change_chance=0.7, inverted_mask=False, add_noise=False)

        optimizer = Adam(0.0002, 0.5)

        # build and compile the encoder
        self.encoder = self.build_encoder()
        self.encoder.compile(loss='mean_squared_error', optimizer=optimizer)

        # build and compile the decoder
        self.decoder = self.build_decoder()
        self.decoder.compile(loss='mean_squared_error', optimizer=optimizer)

        X = Input(shape=(28, 28, 1))
        M = Input(shape=(28, 28, 1))
        encoded = self.encoder([X, M])
        decoded = self.decoder(encoded)
        self.combined_model = Model(inputs=[X, M], outputs=decoded)
        self.combined_model.compile(loss='mean_squared_error', optimizer=optimizer)

        # load the mnist dataset
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

        # rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        X_test = (X_test.astype(np.float32) - 127.5) / 127.5
        X_test = np.expand_dims(X_test, axis=3)

        ix = np.random.randint(0, X_train.shape[0], 360000)
        X_train = X_train[ix]
        X_train_masked, X_masks = self.masking_function.mask(X_train)
        X_test_masked, X_test_masks = self.masking_function.mask(X_test)

        # reshape back to (28, 28, 1)
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
        X_train_masked = X_train_masked.reshape(X_train_masked.shape[0], 28, 28, 1)
        X_test_masked = X_test_masked.reshape(X_test_masked.shape[0], 28, 28, 1)
        X_masks = X_masks.reshape(X_masks.shape[0], 28, 28, 1)
        X_test_masks = X_test_masks.reshape(X_test_masks.shape[0], 28, 28, 1)

        # expose the data
        self.X_train = X_train
        self.X_test = X_test
        self.X_train_masked = X_train_masked
        self.X_test_masked = X_test_masked
        self.X_masks = X_masks
        self.X_test_masks = X_test_masks
        self.y_test = y_test
        self.y_train = y_train
        
        # for generating samples
        # create a list of lists of the indexes of the test set images with each label
        self.test_label_indices = []
        self.masked_img_samples = None
        self.masked_img_masks = None
        for i in range(10):
            self.test_label_indices.append(np.where(y_test == i)[0])


        # for model evaluation
        # load a pre-trained CNN model and see how well it performs on the reconstructed images
        self.cnn = load_model('../mnist-cnn/mnist_cnn.h5')
        self.cnn.trainable = False
        self.cnn_accuracies = []




    def build_encoder(self):

        image = Input(self.input_shape)
        mask = Input(self.input_shape)

        m = MaxPooling2D(pool_size=(4, 4))(mask)
        m = Flatten()(m)

        x = Conv2D(8, kernel_size=4, padding='same', activation='relu')(image)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(32, kernel_size=4, padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)

        c = Concatenate()([x, m])

        c = Dense(64, activation='relu')(c)

        c = Dense(self.latent_dims, activation='relu')(c)

        return Model(inputs=[image, mask], outputs=c)



    def build_decoder(self):

        latent_rep = Input(shape=(self.latent_dims,))
        c = Dense(49, activation='relu')(latent_rep)
        c = Reshape((7, 7, 1))(c)
        c = Conv2DTranspose(32, kernel_size=4, padding='same', activation='relu')(c)
        c = UpSampling2D(size=(2, 2))(c)
        c = Conv2DTranspose(16, kernel_size=4, padding='same', activation='relu')(c)
        c = UpSampling2D(size=(2, 2))(c)
        c = Conv2DTranspose(1, kernel_size=4, padding='same', activation='tanh')(c)

        model = Model(inputs=[latent_rep], outputs=c)

        return model



    def train(self, epochs, batch_size=128, sample_interval=50):


        for epoch in range(epochs):

            # select a random batch of images
            idx = np.random.randint(0, self.X_train_masked.shape[0], batch_size)
            masked_imgs = self.X_train_masked[idx]
            imgs = self.X_train[idx]
            masks = self.X_masks[idx]

            # train the combined model
            loss = self.combined_model.train_on_batch([masked_imgs, masks], imgs)
            
            # print the progress
            print(f"{epoch} [Loss: {loss}]")
            
            # if at save interval, save generated sample images
            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.evaluate_using_cnn(epoch)
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
                    axs[i, j].imshow(self.masked_img_samples[cnt, :, :])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("images/0_masked.png")
            plt.close()
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(self.masked_img_masks[cnt, :, :])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("images/0_masks.png")
            plt.close()

        # generate images using the combined model
        gen_imgs = self.combined_model.predict([self.masked_img_samples, self.masked_img_masks])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()
        

    def evaluate_using_cnn(self, epoch):
        generated_images = self.combined_model.predict([self.X_test_masked, self.X_test_masks])
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
        self.combined_model.save('saved_model/combined_model.keras')



if __name__ == '__main__':
    ae = Autoencoder()
    ae.train(epochs=20001, batch_size=32, sample_interval=2000)
        
