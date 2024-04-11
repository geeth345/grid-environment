import numpy as np
from mask3 import Mask
from keras.datasets import cifar10

walk_length_min = 200
walk_length_max = 700
visible_radius = 2
direction_change_chance = 0.7


if __name__ == "__main__":
    masking_function = Mask(walk_length_min=walk_length_min, walk_length_max=walk_length_max, visible_radius=visible_radius, direction_change_chance=direction_change_chance)

    # Load the cifar10 dataset

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5

    ix = np.random.randint(0, X_train.shape[0], 100000)
    X_train = X_train[ix]
    y_train = y_train[ix]
    X_train_masked, X_masks = masking_function.mask(X_train)
    X_test_masked, X_test_masks = masking_function.mask(X_test)

    np.savez(f"../../data/cifar_masked{walk_length_min}_{walk_length_max}_{direction_change_chance}.npz", X_train=X_train, X_test=X_test, X_train_masked=X_train_masked, X_test_masked=X_test_masked, X_masks=X_masks, X_test_masks=X_test_masks, y_train=y_train, y_test=y_test)