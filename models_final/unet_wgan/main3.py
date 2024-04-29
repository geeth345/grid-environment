# keras imports
import time

from keras.layers import Input, Dense, MaxPooling2D, Conv2D, LeakyReLU, Concatenate, Reshape
from keras.layers import Conv2DTranspose, Flatten, UpSampling2D, Activation, BatchNormalization
from keras.layers import GaussianNoise, Dropout
from keras.models import load_model
from keras import Model
from keras.optimizers.legacy import Adam
import tensorflow as tf

# other module imports
import numpy as np
import matplotlib.pyplot as plt
from keras.src.engine.base_layer import Layer
from keras.src.layers import Add
from tqdm import tqdm
import random


class UNet():

    def __init__(self):

        self.input_shape = (28, 28, 1)
        self.data_source = '../../data/masked100_600_0.7.npz'

        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()
        print("Generator Summary")
        print(self.generator.summary())
        print("Discriminator Summary")
        print(self.discriminator.summary())

        # defining loss functions
        def discriminator_loss(real_img, fake_img):
            real_loss = tf.reduce_mean(real_img)
            fake_loss = tf.reduce_mean(fake_img)
            return fake_loss - real_loss
        def generator_loss(fake_img):
            return -tf.reduce_mean(fake_img)


        # instatntiate opmimisers
        d_optimiser = Adam(0.0002, 0.5, 0.9)
        g_optimiser = Adam(0.0002, 0.5, 0.9)

        # instantiate the WGAN model
        self.wgan = self.WGAN(self.discriminator, self.generator, self)
        self.wgan.compile(
            d_optimiser=d_optimiser,
            g_optimiser=g_optimiser,
            d_loss_fn=discriminator_loss,
            g_loss_fn=generator_loss
        )

        # compile the generator model for the intial mse loss training
        self.generator.compile(loss='mean_squared_error', optimizer=Adam(0.0008, 0.5))


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

        # for model evaluation
        # load a pre-trained CNN model and see how well it performs on the reconstructed images
        self.cnn = load_model('../_classifier/mnist_cnn.h5')
        self.cnn.trainable = False
        self.cnn_accuracies = []

    class WGAN(Model):

        def __init__(self, discriminator, generator, parent, **kwargs):
            super(UNet.WGAN, self).__init__(**kwargs)
            self.discriminator = discriminator
            self.generator = generator
            self.parent = parent

        def compile(self, d_optimiser, g_optimiser, d_loss_fn, g_loss_fn):
            self.d_optimiser = d_optimiser
            self.g_optimiser = g_optimiser
            self.d_loss_fn = d_loss_fn
            self.g_loss_fn = g_loss_fn

        def gradient_penalty(self, batch_size, real_images, fake_images):
            # This function calculates the gradient penalty (instead of weight clipping like
            # in a standard WGAN) to enforce the Lipschitz constraint
            # it is calculated on an interpolated image and added to the disc loss

            # interpolated image generation
            alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
            diff = fake_images - real_images
            interpolated = real_images + alpha * diff

            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                # get discriminator output for interpolated image
                pred = self.discriminator(interpolated, training=True)

            # calculate the gradients of the discriminator output w.r.t the interpolated image
            grads = tape.gradient(pred, [interpolated])[0]
            # calculate the L2 norm of the gradients
            norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
            # calculate the gradient penalty
            gp = tf.reduce_mean((norm - 1.0) ** 2)
            return gp



        def train_step(self, batch_size, trainGenerator=True):

            # train the generator
            # note that the wgan paper suggests training the generator more often than the discriminator
            for i in range(5):

                # collect data
                idx = np.random.randint(0, self.parent.X_train.shape[0], batch_size)
                real_images = self.parent.X_train[idx]
                masked_images = self.parent.X_train_masked[idx]
                masks = self.parent.X_masks[idx]


                with tf.GradientTape() as tape:
                    # generate fake images
                    fake_images = self.generator([masked_images, masks], training=True)
                    # get discriminator output for fake images
                    fake_logits = self.discriminator(fake_images, training=True)
                    # discriminator output on the real images
                    real_logits = self.discriminator(real_images, training=True)

                    # calculator the discriminator loss
                    d_cost = self.d_loss_fn(real_img=real_logits, fake_img=fake_logits)
                    # calculate gradient penalty
                    gp = self.gradient_penalty(batch_size, real_images, fake_images)
                    # add the gradient penalty to the discriminator loss
                    d_loss = d_cost + gp * 10

                # get the gradients w.r.t the discriminator loss
                d_gradient = tape.gradient(d_loss, self.discriminator.trainable_variables)
                # apply the gradients to the discriminator
                self.d_optimiser.apply_gradients(
                    zip(d_gradient, self.discriminator.trainable_variables)
                )

            if trainGenerator:
                # collect data
                idx = np.random.randint(0, self.parent.X_train.shape[0], batch_size)
                masked_images = self.parent.X_train_masked[idx]
                masks = self.parent.X_masks[idx]


                # train the generator
                with tf.GradientTape() as tape:
                    # generate fake images
                    generated_images = self.generator([masked_images, masks], training=True)
                    # get discriminator output for fake images
                    gen_img_logits = self.discriminator(generated_images, training=True)
                    # calculate the generator loss
                    g_loss = self.g_loss_fn(gen_img_logits)

                # get the gradients w.r.t the generator loss
                gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
                # apply the gradients to the generator
                self.g_optimiser.apply_gradients(
                    zip(gen_gradient, self.generator.trainable_variables)
                )
            else:
                g_loss = 0

            return {"d_loss": d_loss, "g_loss": g_loss}



    def build_discriminator(self):

        image = Input(self.input_shape)

        # Base path
        x = Conv2D(16, kernel_size=4, strides=1, padding="same")(image)
        x = LeakyReLU(alpha=0.2)(x)

        # Parallel path 1 - Small kernel for fine-grained features
        p1 = Conv2D(16, kernel_size=2, strides=1, padding="same")(image)
        p1 = LeakyReLU(alpha=0.2)(p1)

        # Parallel path 2 - Large kernel for broad features
        p2 = Conv2D(16, kernel_size=6, strides=1, padding="same")(image)
        p2 = LeakyReLU(alpha=0.2)(p2)

        # Merge parallel paths
        x = Concatenate()([x, p1, p2])

        x = Conv2D(16, kernel_size=4, strides=2, padding="same")(x)
        x = GaussianNoise(0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.05)(x)

        x = Conv2D(32, kernel_size=4, strides=2, padding="same")(x)
        x = GaussianNoise(0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.05)(x)

        x = BatchNormalization(momentum=0.8)(x)
        x = Conv2D(64, kernel_size=4, strides=2, padding="same")(x)
        x = GaussianNoise(0.05)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.05)(x)

        x_flat = Flatten()(x)

        validity_dense = Dense(256)(x_flat)
        validity_dense = LeakyReLU(alpha=0.2)(validity_dense)
        validity_dense = Dropout(0.05)(validity_dense)
        validity_dense = Dense(128)(validity_dense)
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

        d3 = Conv2DTranspose(32, kernel_size=(4, 4), strides=(1, 1), padding='same')(dial3)
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




        for epoch in range(epochs):

            metrics = {"d_loss": 0, "g_loss": 0}

            if epoch <= 50:
                # train the generotor with mse loss
                idx = np.random.randint(0, self.X_train.shape[0], batch_size)
                masked_images = self.X_train_masked[idx]
                masks = self.X_masks[idx]
                images = self.X_train[idx]
                loss = self.generator.train_on_batch([masked_images, masks], images)
                print(f"{epoch} [MSE Loss: {loss}]")

            else:


                ########################
                # Train Model          #
                ########################

                traingenerator = epoch > 60

                metrics = self.wgan.train_step(batch_size, trainGenerator=traingenerator)
                print(f"{epoch} [Discriminator Loss: {metrics['d_loss']}, Generator Loss: {metrics['g_loss']}]")


            ########################
            # Evaluate Model       #
            ########################

            # calculate psnr
            idx = np.random.randint(0, self.X_test.shape[0], batch_size)
            masked_images = self.X_test_masked[idx]
            masks = self.X_test_masks[idx]
            images = self.X_test[idx]
            labels = self.y_test[idx]
            generated_images = self.generator.predict([masked_images, masks], verbose=0)
            psnr = np.mean(10 * np.log10(1 / np.mean(np.square(images - generated_images))))
            cnn_accuracy = np.mean(np.argmax(self.cnn.predict(generated_images, verbose=0), axis=1) == labels)

            # write metrics to file
            if epoch == 0:
                self.metrics_file.write("epoch,d_loss,g_loss,psnr,cnn_accuracy\n")
            self.metrics_file.write(f"{epoch},{metrics['d_loss']},{metrics['g_loss']},{psnr},{cnn_accuracy}\n")


            if epoch % sample_interval == 0:
                self.sample_images(epoch)
                self.evaluate_using_cnn(epoch)
                self.backup_model(epoch)

        self.metrics_file.close()

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
        self.generator.save(f'saved_model/gen2_{epoch}.keras')
        self.discriminator.save(f'saved_model/disc2_{epoch}.keras')


if __name__ == '__main__':
    unet = UNet()

    start1 = time.perf_counter()
    start2 = time.process_time()

    unet.train(epochs=4001, batch_size=64, sample_interval=100)

    end1 = time.perf_counter()
    end2 = time.process_time()
    elapsed1 = end1 - start1
    elapsed2 = end2 - start2

    file = open('time.txt', 'w')
    file.write(str(elapsed1) + '\n' + str(elapsed2))
    file.close()
    print(f"Time taken (total): {elapsed1}")
    print(f"Time taken (process): {elapsed2}")
