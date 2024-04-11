import numpy as np
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import gridspec

# load the data
# load the dataset from the file
data = np.load('../../data/masked100_600_0.7.npz')
X_test = data['X_test']
y_test = data['y_test']
X_test_masked = data['X_test_masked']
X_test_masks = data['X_test_masks']

# load the model
model = load_model('saved_model/gen.keras', compile=False)

indexes = []
images = []
masked_images = []
masks = []
generated_images = []

# select a sample image for each label

for i in range(10):
    index = np.where(y_test == i)[0][0]
    indexes.append(index)

for index in indexes:
    images.append(X_test[index])
    masked_images.append(X_test_masked[index])
    masks.append(X_test_masks[index])

# generate the images
for i in range(len(images)):
    generated_images.append(model.predict([np.expand_dims(masked_images[i], axis=0), np.expand_dims(masks[i], axis=0)]))

# save a plot of the masked sample images for comparison
fig = plt.figure(figsize=(10, 4))
gs = gridspec.GridSpec(4, len(images), wspace=0.1, hspace=0.1)

for i in range(len(images)):
    # Original Image
    ax = plt.subplot(gs[0, i])
    ax.imshow(images[i], interpolation='nearest', aspect='auto')
    ax.axis('off')

    # Masked Image
    ax = plt.subplot(gs[1, i])
    ax.imshow(masked_images[i], interpolation='nearest', aspect='auto')
    ax.axis('off')

    # Mask
    ax = plt.subplot(gs[2, i])
    ax.imshow(masks[i], interpolation='nearest', aspect='auto')
    ax.axis('off')

    # Generated Image
    ax = plt.subplot(gs[3, i])
    ax.imshow(generated_images[i][0, :, :], interpolation='nearest', aspect='auto')
    ax.axis('off')

fig.savefig("example.png")

print("Done generating example images")