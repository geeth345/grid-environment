import sys
sys.path.append('../processing')
from mask2 import Mask

from keras.models import load_model
from keras.datasets import mnist


# load the mnist dataset
_, (X_test, y_test) = mnist.load_data()

# load the model
autoencoder = load_model('saved_model/v2_gen.keras', compile=False)
classifier = load_model('../mnist-cnn/mnist_cnn.h5')

# rescale -1 to 1
X_test = (X_test.astype('float32') - 127.5) / 127.5
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

mask_levels = range(0, 1000, 50)

f = open('eval_results.txt', 'w')
f.write("mask_level,score\n")

results = []

for mask_level in mask_levels:
    masking_fn = Mask(walk_length_min=mask_level, walk_length_max=mask_level, visible_radius=1, direction_change_chance=0.7, inverted_mask=False, add_noise=False)
    # create a copy of the test set
    X_test_masked = X_test.copy()
    X_test_masked, X_test_masks = masking_fn.mask(X_test_masked)
    X_test_masked = X_test_masked.reshape(X_test_masked.shape[0], 28, 28, 1)
    X_test_masks = X_test_masks.reshape(X_test_masks.shape[0], 28, 28, 1)

    # evaluate the model
    generated_images = autoencoder.predict([X_test_masked, X_test_masks])
    y_pred = classifier.predict(generated_images).argmax(axis=1)
    score = (y_pred == y_test).mean()
    results.append(score)
    f.write(f"{mask_level},{score}\n")

f.close()

# plot the results
import matplotlib.pyplot as plt
plt.plot(mask_levels, results)
plt.xlabel('Mask Level')
plt.ylabel('Accuracy')
plt.title('Accuracy of u-net model at Different Mask Levels')
plt.savefig('eval_results.png')
plt.close()