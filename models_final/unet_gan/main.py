# keras imports
from keras.layers import Input, Dense, MaxPooling2D, Conv2D, LeakyReLU, Concatenate, Reshape
from keras.layers import Conv2DTranspose, Flatten, UpSampling2D, Activation, BatchNormalization
from keras.layers import GaussianNoise, Dropout
from keras.models import Model, load_model
from keras.optimizers.legacy import Adam

# other module imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


class UNet():

    def __init__(self):

        self.input_shape = (28, 28, 1)
        self.data_source = '../../data/masked100_600_0.7.npz'

        # build the generator model
        self.generator = self.build_generator()
        # self.generator = load_model('checkpoints/gen_900.keras')
        self.generator.compile()

        # lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate=0.0001,
        #     decay_steps=1000,
        #     decay_rate=0.85)

        # build the discriminator model
        self.discriminator = self.build_discriminator()
        #self.discriminator = load_model('checkpoints/disc_900.keras')
        self.discriminator.compile(loss='binary_crossentropy', optimizer=Adam(0.00015, 0.5, 0.9), metrics=['accuracy'])


        print("Generator Summary")
        print(self.generator.summary())
        print("Discriminator Summary")
        print(self.discriminator.summary())


        # build the combined model (generator with adversarial loss for training)
        self.discriminator.trainable = False
        masked_image = Input(shape=self.input_shape)
        mask = Input(shape=self.input_shape)
        generated_image = self.generator([masked_image, mask])
        validity = self.discriminator(generated_image)
        self.combined = Model([masked_image, mask], [validity])
        self.combined.compile(loss='binary_crossentropy', optimizer=Adam(0.00020, 0.5, 0.9), metrics=['accuracy'])



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

    def build_discriminator(self):

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
        validity = Dense(1, activation='sigmoid', name='validity')(validity_dense)

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
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # accuracy = 0
        # best = -9999999
        # best_epoch = 1000

        for epoch in tqdm(range(0, epochs)):

            ########################
            # Prepare Data         #
            ########################

            idx = np.random.randint(0, self.X_train.shape[0], batch_size)
            images = self.X_train[idx]
            masked_images = self.X_train_masked[idx]
            masks = self.X_masks[idx]
            labels = self.y_train[idx]

            ########################
            # Train Discriminator #
            ########################

            # generate a batch of images using the generator
            gen_images = self.generator.predict([masked_images, masks], verbose=0)

            # train the discrimnator on the real images
            d_real_loss, d_real_acc = self.discriminator.train_on_batch(images, valid)

            # train the discriminator on the generated images
            d_fake_loss, d_fake_acc = self.discriminator.train_on_batch(gen_images, fake)


            #########################
            # Train Generator       #
            #########################

            idx1 = np.random.randint(0, self.X_train.shape[0], batch_size)
            images1 = self.X_train[idx1]
            masked_images1 = self.X_train_masked[idx1]
            masks1 = self.X_masks[idx1]

            # use the combined model to train the generator (discriminator weights are fixed)
            g_loss, g_acc = self.combined.train_on_batch([masked_images, masks], valid)

            # print the progress
            #print(f"{epoch} [Generator Loss: {g_loss}]")


            # calculate metrics and write to file
            mse = np.mean(np.square(images - gen_images))
            psnr = 10 * np.log10(1 / mse)

            # accuracy
            cnn_accuracy = np.mean(np.argmax(self.cnn.predict(gen_images, verbose=0), axis=1) == labels)

            if epoch == 0:
                self.metrics_file.write("Epoch,mse,psnr,d_real_loss,d_real_acc,d_fake_loss,d_fake_acc,g_loss,g_acc,cnn_accuracy\n")
            self.metrics_file.write(f"{epoch},{mse},{psnr},{d_real_loss},{d_real_acc},{d_fake_loss},{d_fake_acc},{g_loss},{g_acc},{cnn_accuracy}\n")


            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                accuracy = self.evaluate_using_cnn(epoch)
                self.backup_model(epoch)


            # if accuracy > best:
            #     best = accuracy
            #     best_epoch = epoch
            #
            # if best_epoch + 200 <= epoch:
            #     print("Early stopping")
            #     break



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

        return accuracy

    def backup_model(self, epoch):
        self.generator.save(f'saved_model/gen_{epoch}.keras')
        self.discriminator.save(f'saved_model/disc_{epoch}.keras')


if __name__ == '__main__':
    unet = UNet()
    unet.train(epochs=8001, batch_size=32, sample_interval=200)
