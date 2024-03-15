import pandas as pd
import matplotlib.pyplot as plt

import sys
from processing.mask2 import Mask

from keras.models import load_model
from keras.datasets import mnist

# models
to_evaluate = {
    'old_AE': 'autoencoder/saved_model/combined_model.keras',
    'new_AE': 'autoencoder-big/saved_model/v1_gen.keras',
    #'new_AE_500e' : 'autoencoder-big/saved_model/v1_gen_500e.keras',
    'unet': 'unet/saved_model/unet_no_discrim_4000e.keras',
    'unet_gan': 'unet/saved_model/unet_gan.keras',
    'unet_combined': 'unet/saved_model/v2_gen.keras',
    #'unet_500e': 'unet/saved_model/unet_no_discrim_500e.keras'
}

mask_levels = range(0, 56, 4)


eval_results = {name: [] for name in to_evaluate.keys()}
eval_results['baseline'] = []

# load mnist data and rescale -1 to 1
_, (X_test, y_test) = mnist.load_data()
X_test = (X_test.astype('float32') - 127.5) / 127.5
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# load all the models
models = {name: load_model(path, compile=False) for name, path in to_evaluate.items()}

# load the classifier
classifier = load_model('mnist-cnn/mnist_cnn.h5')

for mask_level in mask_levels:
    masking_fn = Mask(walk_length_min=mask_level, walk_length_max=mask_level, visible_radius=1, direction_change_chance=0.7, inverted_mask=False, add_noise=False)
    # create a copy of the test set
    X_test_masked = X_test.copy()
    X_test_masked, X_test_masks = masking_fn.mask(X_test_masked)
    X_test_masked = X_test_masked.reshape(X_test_masked.shape[0], 28, 28, 1)
    X_test_masks = X_test_masks.reshape(X_test_masks.shape[0], 28, 28, 1)

    for name, model in models.items():
        print(f"Evaluating {name} at mask level {mask_level}")
        # evaluate the model
        generated_images = model.predict([X_test_masked, X_test_masks])
        y_pred = classifier.predict(generated_images).argmax(axis=1)
        score = (y_pred == y_test).mean()
        eval_results[name].append(score)
    # evaluate baseline
    print(f"Evaluating baseline at mask level {mask_level}")
    y_pred = classifier.predict(X_test_masked).argmax(axis=1)
    score = (y_pred == y_test).mean()
    eval_results['baseline'].append(score)


# plot the results
for name, results in eval_results.items():
    plt.plot(mask_levels, results, label=name)
plt.xlabel('Mask Level (number of steps)')
plt.ylabel('Accuracy of CNN')
plt.title('Accuracy of CNN on Reconstructed Images at Different Mask Levels')
plt.legend()
plt.savefig("comparison_02.png")
plt.show()
plt.close()
