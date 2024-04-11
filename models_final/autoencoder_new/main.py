# keras imports
from keras.layers import Input, Dense, MaxPooling2D, Conv2D, LeakyReLU, Concatenate, Reshape
from keras.layers import Conv2DTranspose, Flatten, UpSampling2D, Activation, BatchNormalization
from keras.models import Model, load_model
from keras.optimizers.legacy import Adam

# other module imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random


class Autoencoder():

    def __init__(self):

        self.input_shape = (28, 28, 1)
        self.data_source = '../../data/masked100_600_0.7.npz'

        self.generator = self.build_generator()
        self.generator.compile(loss='mean_squared_error', optimizer=Adam(0.0008, 0.5))

        print("Generator Summary")
        print(self.generator.summary())
        
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


        d2 = Conv2DTranspose(16, kernel_size=(4, 4), strides=(2, 2), padding='same')(d3)
        d2 = BatchNormalization()(d2)
        d2 = LeakyReLU(alpha=0.2)(d2)


        d1 = Conv2DTranspose(8, kernel_size=(4, 4), strides=(2, 2), padding='same')(d2)
        d1 = BatchNormalization()(d1)
        d1 = LeakyReLU(alpha=0.2)(d1)

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
            
            # select a random batch of images
            idx = np.random.randint(0, self.X_train.shape[0], batch_size)
            images = self.X_train[idx]
            masked_images = self.X_train_masked[idx]
            masks = self.X_masks[idx]
            labels = self.y_train[idx]
            
            # train the generator
            loss = self.generator.train_on_batch([masked_images, masks], images)
            
            # print the progress
            print(f"{epoch} [Generator Loss: {loss}]")


            # calculate metrics and write to file
            mse = np.mean(np.square(images - self.generator.predict([masked_images, masks], verbose=0)))
            psnr = 10 * np.log10(1 / mse)
            if epoch == 0:
                self.metrics_file.write("Epoch,GeneratorLoss,MSE,PSNR\n")
            self.metrics_file.write(f"{epoch},{loss},{mse},{psnr}\n")

            
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

    def backup_model(self, epoch):
        self.generator.save(f'saved_model/gen.keras')


if __name__ == '__main__':
    ae = Autoencoder()
    ae.train(epochs=5001, batch_size=64, sample_interval=500)
