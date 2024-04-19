import numpy as np
import keras
from keras.models import load_model
import matplotlib.pyplot as plt
from matplotlib import gridspec
from tqdm import tqdm


# models
to_evaluate = {
    'Autoencoder': '../autoencoder_new/saved_model/gen.keras',
    'U-Net': '../unet_mse/saved_model/gen.keras',
    'GAN (U-Net)': '../unet_gan/saved_model/gen_4400.keras',
    'ACGAN (U-Net)': '../unet_acgan/saved_model/gen_2600.keras',
    'SAGAN (U-Net)': '../unet_sagan/saved_model/gen_2800.keras',
    'SAGANv2' : '../sagan_v2/saved_model/gen_3000.keras',
    'WGAN-GP (U-Net)': '../unet_wgan/saved_model/gen.keras',
    'ACWGAN (U-Net)': '../unet_acwgan/saved_model/gen.keras'
}

# load the data
# load the dataset from the file
data = np.load('../../data/masked100_600_0.7.npz')
X_test = data['X_test']
y_test = data['y_test']
X_test_masked = data['X_test_masked']
X_test_masks = data['X_test_masks']

# shuffle the dataset to get a random sample
np.random.seed(42)
idx = np.random.permutation(len(X_test))
X_test = X_test[idx]
y_test = y_test[idx]
X_test_masked = X_test_masked[idx]
X_test_masks = X_test_masks[idx]

indexes = []
images = []
masked_images = []
masks = []

all_models_images = []


for i in range(10):
    index = np.where(y_test == i)[0][0]
    indexes.append(index)

for index in indexes:
    images.append(X_test[index])
    masked_images.append(X_test_masked[index])
    masks.append(X_test_masks[index])

# iterate through the models
print("Generating model images")
for model_name, model in tqdm(to_evaluate.items()):
    model = load_model(model, compile=False)
    generated_images = []

    for i in range(len(images)):
        generated_images.append(model.predict([np.expand_dims(masked_images[i], axis=0), np.expand_dims(masks[i], axis=0)], verbose=0))

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
        ax.imshow(generated_images[i][0], interpolation='nearest', aspect='auto')
        ax.axis('off')

    plt.suptitle(model_name)
    plt.savefig(f'../_evaluation/example_images/{model_name}.png')
    plt.close()

    all_models_images.append(generated_images)


# generate a plot of all the generated images together
# use a grid formation
# label the rows

# Example data setup
num_rows = len(to_evaluate)
num_models = len(all_models_images)
num_rows = 3 + num_rows
fig = plt.figure(figsize=(10, num_rows))
gs = gridspec.GridSpec(num_rows, 10, wspace=0.1, hspace=0.1)

for i in range(10):

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

    # Generated Images for each model
    for j in range(num_models):
        ax = plt.subplot(gs[3 + j, i])
        ax.imshow(all_models_images[j][i][0], interpolation='nearest', aspect='auto')
        ax.axis('off')

# Adding labels to each row
row_labels = ['Original Image', 'Masked Image', 'Mask'] + list(to_evaluate.keys())
for idx, label in enumerate(row_labels):
    ax = plt.subplot(gs[idx, 0])
    ax.text(-0.1, 0.5, label, transform=ax.transAxes, fontsize=12, va='center', ha='right', rotation=0, color='black')
    ax.axis('off')

plt.suptitle('Comparison of Generated Images from Different Models')

plt.tight_layout()
plt.savefig('example_images/all_models.png', bbox_inches='tight')
plt.close()



print("Done generating model images")