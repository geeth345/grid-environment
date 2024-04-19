import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import time

from keras.models import load_model
from keras.datasets import mnist
from keras.losses import sparse_categorical_crossentropy

from skimage.metrics import structural_similarity as ssim

# models
to_evaluate = {
    'Autoencoder': '../autoencoder_new/saved_model/gen.keras',
    'U-Net': '../unet_mse/saved_model/gen.keras',
    'GAN (U-Net)': '../unet_gan/saved_model/gen_4800.keras',
    'ACGAN (U-Net)': '../unet_acgan/saved_model/gen_2600.keras',
    'SAGAN (U-Net)': '../unet_sagan/saved_model/gen_2800.keras',
    'SAGANv2' : '../sagan_v2/saved_model/gen_3000.keras',
    'WGAN-GP (U-Net)': '../unet_wgan/saved_model/gen.keras',
    'ACWGAN (U-Net)': '../unet_acwgan/saved_model/gen.keras'
}

import sys
sys.path.append('../../models/processing')
from mask import Mask

mask_levels = range(0, 800, 40)


eval_results_acc = {name: [] for name in to_evaluate.keys()}
eval_results_acc['baseline'] = []

eval_results_loss = {name: [] for name in to_evaluate.keys()}
eval_results_loss['baseline'] = []

eval_results_psnr = {name: [] for name in to_evaluate.keys()}
eval_results_psnr['baseline'] = []

eval_results_ssim = {name: [] for name in to_evaluate.keys()}
eval_results_ssim['baseline'] = []


eval_results_time = {name: [] for name in to_evaluate.keys()}

# load mnist data and rescale -1 to 1
_, (X_test, y_test) = mnist.load_data()
X_test = (X_test.astype('float32') - 127.5) / 127.5
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# load all the models
models = {name: load_model(path, compile=False) for name, path in to_evaluate.items()}

# load the classifier
classifier = load_model('../_classifier/mnist_cnn.h5')


def get_classifier_score(X, y):
    # calculate the average SCCE loss value
    y_pred = classifier.predict(X)
    loss = np.mean(sparse_categorical_crossentropy(y, y_pred))
    accuracy = np.mean((y_pred.argmax(axis=1) == y))
    return loss, accuracy

def get_psnr(X, y):
    return 10 * np.log10(1 / np.mean(np.square(X - y)))

def get_ssim(X, y):
    # print(X.shape, y.shape)
    res = []
    for i in range(X.shape[0]):
        res.append(ssim(np.squeeze(X[i]), np.squeeze(y[i]), data_range=2))
    return np.mean(res)



for mask_level in mask_levels:
    masking_fn = Mask(walk_length_min=mask_level, walk_length_max=mask_level, visible_radius=1, direction_change_chance=0.7, inverted_mask=False, add_noise=False)
    # create a copy of the test set
    X_test_masked = X_test.copy()
    X_test_masked, X_test_masks = masking_fn.mask(X_test_masked)
    X_test_masked = X_test_masked.reshape(X_test_masked.shape[0], 28, 28, 1)
    X_test_masks = X_test_masks.reshape(X_test_masks.shape[0], 28, 28, 1)

    for name, model in models.items():
        print(f"Evaluating {name} at mask level {mask_level}")

        # generate some images + time taken
        start = time.perf_counter()
        generated_images = model.predict([X_test_masked, X_test_masks])
        end = time.perf_counter()
        elapsed = end - start
        eval_results_time[name].append(elapsed)

        # evaluate the model using the classifier
        loss, accuracy = get_classifier_score(generated_images, y_test)
        eval_results_acc[name].append(accuracy)
        eval_results_loss[name].append(loss)

        # evaluate the model using average PSNR
        psnr = get_psnr(X_test, generated_images)
        eval_results_psnr[name].append(psnr)

        # evaluate the model using SSIM
        ssim_score = get_ssim(X_test, generated_images)
        eval_results_ssim[name].append(ssim_score)


    # evaluate baseline
    print(f"Evaluating baseline at mask level {mask_level}")
    loss, accuracy = get_classifier_score(X_test_masked, y_test)
    eval_results_acc['baseline'].append(accuracy)
    eval_results_loss['baseline'].append(loss)

    psnr = get_psnr(X_test, X_test_masked)
    eval_results_psnr['baseline'].append(psnr)

    ssim_score = get_ssim(X_test, X_test_masked)
    eval_results_ssim['baseline'].append(ssim_score)


# plot the results
# cnn score
for name, results in eval_results_acc.items():
    plt.plot(mask_levels, results, label=name)
plt.xlabel('Mask Level (number of steps)')
plt.ylabel('Accuracy of CNN')
plt.title('Accuracy of CNN on Reconstructed Images at Different Mask Levels')
plt.legend()
plt.savefig("eval_charts/accuracy.png")
#plt.show()
plt.close()

# loss
for name, results in eval_results_loss.items():
    plt.plot(mask_levels, results, label=name)
plt.xlabel('Mask Level (number of steps)')
plt.ylabel('Average Loss')
plt.title('Average Loss of Classifier on Reconstructed Images at Different Mask Levels')
plt.legend()
plt.savefig("eval_charts/loss.png")
#plt.show()
plt.close()


# psnr
for name, results in eval_results_psnr.items():
    plt.plot(mask_levels, results, label=name)
plt.xlabel('Mask Level (number of steps)')
plt.ylabel('PSNR')
plt.title('PSNR of Reconstructed Images at Different Mask Levels')
plt.legend()
plt.savefig("eval_charts/psnr.png")
#plt.show()
plt.close()

# ssim
for name, results in eval_results_ssim.items():
    plt.plot(mask_levels, results, label=name)
plt.xlabel('Mask Level (number of steps)')
plt.ylabel('SSIM')
plt.title('SSIM of Reconstructed Images at Different Mask Levels')
plt.legend()
plt.savefig("eval_charts/ssim.png")
#plt.show()
plt.close()


# time taken
keys = []
times = []
for key, value in eval_results_time.items():
    keys.append(key)
    times.append(np.mean(value))
plt.bar(keys, times)
plt.xlabel('Model')
plt.ylabel('Average Time (seconds)')
plt.title('Average Time for Each Model')
plt.savefig('eval_charts/time.png')
plt.close()
