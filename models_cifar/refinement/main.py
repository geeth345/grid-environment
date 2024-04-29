# keras imports
from keras.layers import Input, Dense, MaxPooling2D, Conv2D, LeakyReLU, Concatenate, Reshape
from keras.layers import Conv2DTranspose, Flatten, UpSampling2D, Activation, BatchNormalization
from keras.layers import GaussianNoise, Dropout
from keras.models import Model, load_model
from keras.optimizers.legacy import Adam

# other module imports
import numpy as np
import matplotlib.pyplot as plt
from keras.src.layers import MultiHeadAttention, Resizing
from tqdm import tqdm
import random

# set path to current location
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# building a dialated convolution based refinement network
class SAGAN():

    def __init__(self):

        self.input_shape = (32, 32, 3)
        self.mask_shape = (32, 32, 1)
        self.data_source = '../../data/cifar_masked200_700_0.7.npz'

        # load the coarse model
        self.coarse_model = load_model('coarse.keras')
        self.coarse_model.trainable = False

        # build the generator model
        self.gen_fine = self.build_generator_fine()
        masked_image = Input(shape=self.input_shape)
        mask = Input(shape=self.mask_shape)
        coarse_image = self.coarse_model([masked_image, mask])
        generated_image = self.gen_fine(coarse_image)

        self.generator = Model([masked_image, mask], generated_image)


        # build the discriminator model
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=Adam(0.000075, 0.5, 0.9), metrics=['accuracy'], loss_weights=[0.45, 0.55])

        # build the combined model (generator with adversarial loss for training)
        self.discriminator.trainable = False
        validity, label = self.discriminator(generated_image)
        self.combined = Model([masked_image, mask], [validity, label])
        self.combined.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=Adam(0.0001, 0.5, 0.9), metrics=['accuracy'], loss_weights=[0.45, 0.55])


        print("Generator Summary")
        print(self.generator.summary())
        print("Discriminator Summary")
        print(self.discriminator.summary())

        # load the dataset from the file
        data = np.load(self.data_source)
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

        # # for model evaluation
        # # load a pre-trained CNN model and see how well it performs on the reconstructed images
        # self.cnn = load_model('../_classifier/mnist_cnn.h5')
        # self.cnn.trainable = False
        # self.cnn_accuracies = []

    def build_discriminator(self):

        image = Input(self.input_shape)

        # Base path
        x = Conv2D(32, kernel_size=4, strides=1, padding="same")(image)
        x = LeakyReLU(alpha=0.2)(x)
        x = Conv2D(32, kernel_size=4, strides=1, padding="same")(x)
        x = LeakyReLU(alpha=0.2)(x)

        # Parallel path 1 - Small kernel for fine-grained features
        p1 = Conv2D(32, kernel_size=2, strides=1, padding="same")(image)
        p1 = LeakyReLU(alpha=0.2)(p1)
        p1 = Conv2D(32, kernel_size=2, strides=1, padding="same")(p1)
        p1 = LeakyReLU(alpha=0.2)(p1)

        # Parallel path 2 - Large kernel for broad features
        p2 = Conv2D(32, kernel_size=6, strides=1, padding="same")(image)
        p2 = LeakyReLU(alpha=0.2)(p2)
        p2 = Conv2D(32, kernel_size=6, strides=1, padding="same")(p2)
        p2 = LeakyReLU(alpha=0.2)(p2)

        # Merge parallel paths
        x = Concatenate()([x, p1, p2])

        x = Conv2D(32, kernel_size=4, strides=2, input_shape=self.input_shape, padding="same")(x)
        x = GaussianNoise(0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.1)(x)
        x = Conv2D(32, kernel_size=4, strides=1, input_shape=self.input_shape, padding="same")(x)
        x = GaussianNoise(0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.1)(x)


        x = Conv2D(64, kernel_size=4, strides=2, padding="same")(x)
        x = GaussianNoise(0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.1)(x)
        x = Conv2D(64, kernel_size=4, strides=1, padding="same")(x)
        x = GaussianNoise(0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.1)(x)

        #x = BatchNormalization(momentum=0.8)(x)
        x = Conv2D(128, kernel_size=4, strides=2, padding="same")(x)
        x = GaussianNoise(0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.1)(x)
        x = Conv2D(128, kernel_size=4, strides=1, padding="same")(x)
        x = GaussianNoise(0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.1)(x)

        x_flat = Flatten()(x)

        validity_dense = Dense(256)(x_flat)
        validity_dense = LeakyReLU(alpha=0.2)(validity_dense)
        validity_dense = Dropout(0.1)(validity_dense)
        validity_dense = Dense(128)(validity_dense)
        validity_dense = LeakyReLU(alpha=0.2)(validity_dense)
        validity = Dense(1, activation='sigmoid', name='validity')(validity_dense)

        label_dense = Dense(512)(x_flat)
        label_dense = LeakyReLU(alpha=0.2)(label_dense)
        label_dense = Dropout(0.1)(label_dense)
        label_dense = Dense(256)(label_dense)
        label_dense = LeakyReLU(alpha=0.2)(label_dense)
        label = Dense(10, activation='softmax', name='label')(label_dense)

        model = Model(inputs=[image], outputs=[validity, label], name='discriminator')

        return model

    def build_generator_fine(self):

        image = Input(shape=self.input_shape)

        e1 = Conv2D(16, kernel_size=(4, 4), strides=(1, 1), padding='same')(image)
        # e1 = BatchNormalization()(e1)
        e1 = LeakyReLU(alpha=0.2)(e1)

        e2 = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), padding='same')(e1)
        e2 = LeakyReLU(alpha=0.2)(e2)
        e2 = Conv2D(32, kernel_size=(4, 4), strides=(1, 1), padding='same')(e2)
        e2 = LeakyReLU(alpha=0.2)(e2)

        e3 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='same')(e2)
        e3 = LeakyReLU(alpha=0.2)(e3)
        e3 = Conv2D(64, kernel_size=(4, 4), strides=(1, 1), padding='same')(e3)
        e3 = LeakyReLU(alpha=0.2)(e3)

        # dilated convolutions
        # dialated convolution layers
        dial1 = Conv2D(32, kernel_size=(4, 4), strides=(1, 1), padding='same', dilation_rate=1)(e3)
        dial1 = BatchNormalization()(dial1)
        dial1 = LeakyReLU(alpha=0.2)(dial1)

        dial2 = Conv2D(32, kernel_size=(4, 4), strides=(1, 1), padding='same', dilation_rate=1)(dial1)
        dial2 = BatchNormalization()(dial2)
        dial2 = LeakyReLU(alpha=0.2)(dial2)

        dial3 = Conv2D(32, kernel_size=(4, 4), strides=(1, 1), padding='same', dilation_rate=1)(dial2)
        dial3 = BatchNormalization()(dial3)
        dial3 = LeakyReLU(alpha=0.2)(dial3)


        d3 = Conv2D(64, kernel_size=(4, 4), strides=(1, 1), padding='same')(dial3)
        d3 = LeakyReLU(alpha=0.2)(d3)
        d3 = Conv2D(64, kernel_size=(4, 4), strides=(1, 1), padding='same')(d3)
        d3 = LeakyReLU(alpha=0.2)(d3)

        # d2 = UpSampling2D(size=(2, 2))(d3)
        # d2 = Concatenate()([d2, e2])
        # d2 = Conv2DTranspose(16, kernel_size=(4, 4), strides=(1, 1), padding='same')(d2)
        # d2 = LeakyReLU(alpha=0.2)(d2)

        d2 = Resizing(16, 16, interpolation='nearest')(d3)
        d2 = Conv2D(32, kernel_size=(4, 4), strides=(1, 1), padding='same')(d2)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = Conv2D(32, kernel_size=(4, 4), strides=(1, 1), padding='same')(d2)
        d2 = LeakyReLU(alpha=0.2)(d2)
        d2 = Concatenate()([d2, e2])

        # d1 = UpSampling2D(size=(2, 2))(d2)
        # d1 = Concatenate()([d1, e1])
        # d1 = Conv2DTranspose(8, kernel_size=(4, 4), strides=(1, 1), padding='same')(d1)
        # d1 = LeakyReLU(alpha=0.2)(d1)

        d1 = Resizing(32, 32, interpolation='nearest')(d2)
        d1 = Conv2D(16, kernel_size=(4, 4), strides=(1, 1), padding='same')(d1)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d1 = Conv2D(16, kernel_size=(4, 4), strides=(1, 1), padding='same')(d1)
        d1 = LeakyReLU(alpha=0.2)(d1)
        d1 = Concatenate()([d1, e1])

        d0 = Conv2DTranspose(8, kernel_size=(4, 4), strides=(1, 1), padding='same')(d1)
        # d0 = BatchNormalization()(d0)
        d0 = LeakyReLU(alpha=0.2)(d0)

        d0 = Conv2DTranspose(4, kernel_size=(4, 4), strides=(1, 1), padding='same')(d0)
        # d0 = BatchNormalization()(d0)
        d0 = LeakyReLU(alpha=0.2)(d0)

        d0 = Conv2DTranspose(3, kernel_size=(4, 4), strides=(1, 1), padding='same')(d0)

        output = Activation('tanh')(d0)
        output = Reshape(self.input_shape)(output)

        model = Model(inputs=image, outputs=output, name='generator_fine')

        return model


    def train(self, epochs, batch_size, sample_interval):

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        #self.metrics_file.write("epoch,d_real_auth_acc,d_real_class_acc,d_fake_auth_acc,d_fake_class_acc,g_auth_acc,g_class_acc,cnn_accuracy\n")
        self.metrics_file.write("epoch,d_real_auth_acc,d_real_class_acc,d_fake_auth_acc,d_fake_class_acc,g_auth_acc,g_class_acc\n")

        for epoch in range(epochs):

            if epoch == 0:
                # warm up the discriminator
                for _ in range(0):
                    idx = np.random.randint(0, self.X_train.shape[0], batch_size)
                    images = self.X_train[idx]
                    masked_images = self.X_train_masked[idx]
                    masks = self.X_masks[idx]
                    labels = self.y_train[idx]

                    gen_images = self.gen_fine.predict([masked_images, masks], verbose=0)

                    d_real = self.discriminator.train_on_batch(images, [valid, labels])
                    d_fake = self.discriminator.train_on_batch(gen_images, [fake, labels])

                    auth_loss = d_fake[1]
                    class_loss = d_fake[2]
                    auth_acc = d_fake[3]
                    class_acc = d_fake[4]

                    print(f"Warming up Discriminator: [Auth Loss: {auth_loss}, Auth Acc: {auth_acc}, Class Loss: {class_loss}, Class Acc: {class_acc}]")

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
            d_real = self.discriminator.train_on_batch(images, [valid, labels])

            # train the discriminator on the generated images
            d_fake = self.discriminator.train_on_batch(gen_images, [fake, labels])

            # metrics
            #cnn_accuracy = np.mean(np.argmax(self.cnn.predict(gen_images, verbose=0), axis=1) == labels)
            d_real_auth_acc = d_real[3]
            d_real_class_acc = d_real[4]
            d_fake_auth_acc = d_fake[3]
            d_fake_class_acc = d_fake[4]

            #########################
            # Train Generator       #
            #########################

            idx = np.random.randint(0, self.X_train.shape[0], batch_size)
            images = self.X_train[idx]
            masked_images = self.X_train_masked[idx]
            masks = self.X_masks[idx]
            labels = self.y_train[idx]

            # use the combined model to train the generator (discriminator weights are fixed)
            gen = self.combined.train_on_batch([masked_images, masks], [valid, labels])

            # metrics
            auth_loss = gen[1]
            class_loss = gen[2]
            auth_acc = gen[3]
            class_acc = gen[4]

            # print the progress
            print(f"{epoch} [Auth Loss: {auth_loss}, Auth Acc: {auth_acc}, Class Loss: {class_loss}, Class Acc: {class_acc}]")


            # write metrics to file
            #self.metrics_file.write(f"{epoch},{d_real_auth_acc},{d_real_class_acc},{d_fake_auth_acc},{d_fake_class_acc},{auth_acc},{class_acc},{cnn_accuracy}\n")
            self.metrics_file.write(
                f"{epoch},{d_real_auth_acc},{d_real_class_acc},{d_fake_auth_acc},{d_fake_class_acc},{auth_acc},{class_acc}\n")

            # # calculate metrics and write to file
            # mse = np.mean(np.square(images - self.generator.predict([masked_images, masks], verbose=0)))
            # psnr = 10 * np.log10(1 / mse)
            # if epoch == 0:
            #     self.metrics_file.write("Epoch,mse,psnr,d_real_loss,d_real_acc,d_fake_loss,d_fake_acc,g_loss,g_acc\n")
            # self.metrics_file.write(f"{epoch},{mse},{psnr},{d_real_loss},{d_real_acc},{d_fake_loss},{d_fake_acc},{g_loss},{g_acc}\n")

            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                #self.evaluate_using_cnn(epoch)
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
                    axs[i, j].imshow((self.masked_img_samples[cnt, :, :] * 0.5) + 0.5, interpolation='nearest')
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
                    axs[i, j].imshow((imgs[cnt, :, :] * 0.5) + 0.5, interpolation='nearest')
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("images/0_images.png")
            plt.close()

        # generate images using the generator model
        #gen_imgs = self.generator.predict([self.masked_img_samples, self.masked_img_masks])
        # generate coarse images
        coarse_imgs = self.coarse_model.predict([self.masked_img_samples, self.masked_img_masks])
        # generate fine images
        gen_imgs = self.gen_fine.predict(coarse_imgs)

        # Rescale images 0 - 1
        coarse_imgs = 0.5 * coarse_imgs + 0.5
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(coarse_imgs[cnt, :, :], interpolation='nearest')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d_coarse.png" % epoch)
        plt.close()

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :], interpolation='nearest')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d_fine.png" % epoch)
        plt.close()

    # def evaluate_using_cnn(self, epoch):
    #     generated_images = self.generator.predict([self.X_test_masked, self.X_test_masks], verbose=0)
    #     classifications = self.cnn.predict(generated_images, verbose=0)
    #     accuracy = np.mean(np.argmax(classifications, axis=1) == self.y_test)
    #     self.cnn_accuracies.append((accuracy, epoch))
    #
    #     # save a plot of the accuracy scores so far
    #     plt.plot([x[1] for x in self.cnn_accuracies], [x[0] for x in self.cnn_accuracies])
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy')
    #     plt.title('Accuracy of Pre-Trained CNN on Reconstructed Images')
    #     plt.savefig('images/cnn_accuracy.png')
    #     plt.close()

    def backup_model(self, epoch):
        self.generator.save(f'saved_model/gen_{epoch}.keras')
        self.discriminator.save(f'saved_model/disc_{epoch}.keras')


if __name__ == '__main__':
    net = SAGAN()
    net.train(epochs=10001, batch_size=64, sample_interval=500)
