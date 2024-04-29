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
    'ACGAN': '../unet_acgan/saved_model/gen_6000.keras',
    'WGAN-GP': '../unet_wgan/saved_model/gen2_2900.keras',
    'GAN': '../unet_gan/saved_model/gen_4400.keras',
}

import sys
sys.path.append('../../models/processing')
from mask_cifar import Mask

mask_levels = range(0, 95, 5)

test_on_percentage = 20


eval_results_acc = {name: [] for name in to_evaluate.keys()}
eval_results_acc['baseline'] = []

eval_results_conf = {name: [] for name in to_evaluate.keys()}
eval_results_conf['baseline'] = []

eval_results_psnr = {name: [] for name in to_evaluate.keys()}
eval_results_psnr['baseline'] = []

eval_results_ssim = {name: [] for name in to_evaluate.keys()}
eval_results_ssim['baseline'] = []


eval_results_time = {name: [] for name in to_evaluate.keys()}

# load mnist data and rescale -1 to 1
_, (X_test, y_test) = mnist.load_data()
X_test = (X_test.astype('float32') - 127.5) / 127.5
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# get a subset of the test data
idx = np.random.choice(X_test.shape[0], (test_on_percentage * X_test.shape[0]) // 100, replace=False)
X_test = X_test[idx]
y_test = y_test[idx]



# load all the models
models = {name: load_model(path, compile=False) for name, path in to_evaluate.items()}

# load the classifier
classifier = load_model('../_classifier/mnist_cnn.h5')

image_sums = []


def get_classifier_score(X, y):
    y_pred = classifier.predict(X)
    avg_confidence = np.mean(y_pred.max(axis=1))
    accuracy = np.mean((y_pred.argmax(axis=1) == y))
    return avg_confidence, accuracy

def get_psnr(X, y):
    return 10 * np.log10(1 / np.mean(np.square(X - y)))

def get_ssim(X, y):
    # print(X.shape, y.shape)
    res = []
    for i in range(X.shape[0]):
        res.append(ssim(np.squeeze(X[i]), np.squeeze(y[i]), data_range=2))
    return np.mean(res)



for mask_level in mask_levels:
    masking_fn = Mask(mode='percentage', percentage=mask_level, visible_radius=1, background='black', direction_change_chance=0.7)
    # create a copy of the test set
    print(f"Generating masks for mask level {mask_level}")
    X_test_masked = X_test.copy()
    X_test_masked, X_test_masks = masking_fn.mask(X_test_masked)
    X_test_masked = X_test_masked.reshape(X_test_masked.shape[0], 28, 28, 1)
    X_test_masks = X_test_masks.reshape(X_test_masks.shape[0], 28, 28, 1)

    image_sums.append(np.mean(X_test_masked.sum(axis=(1, 2, 3))))

    for name, model in models.items():
        print(f"Evaluating {name} at mask level {mask_level}")

        # generate some images + time taken
        start = time.perf_counter()
        generated_images = model.predict([X_test_masked, X_test_masks])
        end = time.perf_counter()
        elapsed = end - start
        eval_results_time[name].append(elapsed)

        # evaluate the model using the classifier
        confidence, accuracy = get_classifier_score(generated_images, y_test)
        eval_results_acc[name].append(accuracy)
        eval_results_conf[name].append(confidence)

        # evaluate the model using average PSNR
        psnr = get_psnr(X_test, generated_images)
        eval_results_psnr[name].append(psnr)

        # evaluate the model using SSIM
        ssim_score = get_ssim(X_test, generated_images)
        eval_results_ssim[name].append(ssim_score)


    # evaluate baseline
    print(f"Evaluating baseline at mask level {mask_level}")
    confidence, accuracy = get_classifier_score(X_test_masked, y_test)
    eval_results_acc['baseline'].append(accuracy)
    eval_results_conf['baseline'].append(confidence)

    psnr = get_psnr(X_test, X_test_masked)
    eval_results_psnr['baseline'].append(psnr)

    ssim_score = get_ssim(X_test, X_test_masked)
    eval_results_ssim['baseline'].append(ssim_score)


# plot the results
# cnn score
for name, results in eval_results_acc.items():
    plt.plot(mask_levels, results, label=name)
plt.xlabel('Mask Level (% of image visible)')
plt.ylabel('Accuracy of CNN')
plt.title('Accuracy of CNN on Reconstructed Images at Different Mask Levels')
plt.legend()
plt.savefig("eval_charts/accuracy.png")
#plt.show()
plt.close()

# confidence
for name, results in eval_results_conf.items():
    plt.plot(mask_levels, results, label=name)
plt.xlabel('Mask Level (number of steps)')
plt.ylabel('Average Loss')
plt.title('Classification Confidence on Reconstructed Images at Different Mask Levels')
plt.legend()
plt.savefig("eval_charts/confidence.png")
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


# put them all one one plot for the report

fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# CNN score
axs[0, 0].set_title('Accuracy of Pre-Trained CNN')
for name, results in eval_results_acc.items():
    axs[0, 0].plot(mask_levels, results, label=name)
axs[0, 0].set_xlabel('Mask Level (% of image visible)')
axs[0, 0].set_ylabel('Accuracy of CNN')
axs[0, 0].legend()

# Confidence
axs[0, 1].set_title('Classification Confidence of Pre-Trained CNN')
for name, results in eval_results_conf.items():
    axs[0, 1].plot(mask_levels, results, label=name)
axs[0, 1].set_xlabel('Mask Level (% of image visible)')
axs[0, 1].set_ylabel('Average Loss')
axs[0, 1].legend()

# PSNR
axs[1, 0].set_title('PSNR')
for name, results in eval_results_psnr.items():
    axs[1, 0].plot(mask_levels, results, label=name)
axs[1, 0].set_xlabel('Mask Level (% of image visible)')
axs[1, 0].set_ylabel('PSNR')
axs[1, 0].legend()

# SSIM
axs[1, 1].set_title('SSIM')
for name, results in eval_results_ssim.items():
    axs[1, 1].plot(mask_levels, results, label=name)
axs[1, 1].set_xlabel('Mask Level (% of image visible)')
axs[1, 1].set_ylabel('SSIM')
axs[1, 1].legend()

# Adjust layout to prevent overlap
plt.tight_layout()

#plt.suptitle('Quantitative Evaluation of Reconstructed Images at Different Mask Levels')

# Save the entire figure
plt.savefig("eval_charts/all_metrics.png")
plt.close()

# # sums
# plt.plot(mask_levels, image_sums)
# plt.xlabel('Mask Level (number of steps)')
# plt.ylabel('Sum of Image Pixels')
# plt.title('Sum of Image Pixels at Different Mask Levels')
# plt.savefig('eval_charts/sum.png')

