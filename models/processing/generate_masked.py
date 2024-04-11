import numpy as np
from tqdm import tqdm
import random
from mask2 import Mask
from keras.datasets import mnist

walk_length_min = 100
walk_length_max = 600
visible_radius = 1
direction_change_chance = 0.7
inverted_mask = False
add_noise = False



if __name__ == "__main__":
    masking_function = Mask(walk_length_min=50, walk_length_max=600, visible_radius=1,
                            direction_change_chance=0.7, inverted_mask=False,
                            add_noise=False)

    # load the mnist dataset
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    X_test = (X_test.astype(np.float32) - 127.5) / 127.5
    X_test = np.expand_dims(X_test, axis=3)

    ix = np.random.randint(0, X_train.shape[0], 300000)
    X_train = X_train[ix]
    y_train = y_train[ix]
    X_train_masked, X_masks = masking_function.mask(X_train)
    X_test_masked, X_test_masks = masking_function.mask(X_test)

    # reshape back to (28, 28, 1)
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
    X_train_masked = X_train_masked.reshape(X_train_masked.shape[0], 28, 28, 1)
    X_test_masked = X_test_masked.reshape(X_test_masked.shape[0], 28, 28, 1)
    X_masks = X_masks.reshape(X_masks.shape[0], 28, 28, 1)
    X_test_masks = X_test_masks.reshape(X_test_masks.shape[0], 28, 28, 1)

    # np.save(f"../../data/masked{walk_length_min}_{walk_length_max}_{direction_change_chance}/train_masks.npy", X_masks)
    # np.save(f"../../data/masked{walk_length_min}_{walk_length_max}_{direction_change_chance}/test_masks.npy", X_test_masks)
    # np.save(f"../../data/masked{walk_length_min}_{walk_length_max}_{direction_change_chance}/train.npy", X_train)
    # np.save(f"../../data/masked{walk_length_min}_{walk_length_max}_{direction_change_chance}/test.npy", X_test)
    # np.save(f"../../data/masked{walk_length_min}_{walk_length_max}_{direction_change_chance}/train_masked.npy", X_train_masked)
    # np.save(f"../../data/masked{walk_length_min}_{walk_length_max}_{direction_change_chance}/test_masked.npy", X_test_masked)


    np.savez(f"../../data/masked{walk_length_min}_{walk_length_max}_{direction_change_chance}.npz", X_train=X_train, X_test=X_test, X_train_masked=X_train_masked, X_test_masked=X_test_masked, X_masks=X_masks, X_test_masks=X_test_masks, y_train=y_train, y_test=y_test)


    print("Data saved")

