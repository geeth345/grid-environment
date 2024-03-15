# Code based on original work: https://github.com/JamesHarcourt7/autoencoder-perception/blob/main/utils.py
# Original code author: James Harcourt

import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

# import matplotlib.pyplot as plt

RANDOM_WALK_LENGTH_MIN = 100
RANDOM_WALK_LENGTH_MAX = 600
VISIBLE_RADIUS = 5
INVERTED_MASK = False
DIRECTION_CHANGE_CHANCE = 1.0
ADD_NOISE = True
LIMIT_START_POSITION = False
DEBUG = False


class Mask:

    def __init__(self, walk_length_min=RANDOM_WALK_LENGTH_MIN, walk_length_max=RANDOM_WALK_LENGTH_MAX,
                 visible_radius=VISIBLE_RADIUS, inverted_mask=INVERTED_MASK,
                 direction_change_chance=DIRECTION_CHANGE_CHANCE, add_noise=ADD_NOISE, limit_start_position=LIMIT_START_POSITION, debug=DEBUG):
        self.walk_length_min = walk_length_min
        self.walk_length_max = walk_length_max
        self.visible_radius = visible_radius
        self.inverted_mask = inverted_mask
        self.direction_change_chance = direction_change_chance
        self.add_noise = add_noise
        self.limit_start_position = limit_start_position
        self.debug = debug

    def mask(self, images):

        masked_images = []
        masks = []

        for image in tqdm(images):

            if len(image.shape) == 3:
                image = image[:, :, 0]

            # assume the image is a 28x28 numpy array
            assert image.shape == (28, 28)

            mask = np.zeros((28, 28)).astype(int)

            # pick random number of walk steps
            steps = np.random.randint(self.walk_length_min, self.walk_length_max + 1)
            # pick random starting point
            pos = np.random.randint(0, 27), np.random.randint(0, 27)

            if self.limit_start_position:
                pos = (np.random.randint(7, 20), np.random.randint(7, 20))

            previous_direction = (0, 0)

            for _ in range(steps):
                for i in range((self.visible_radius * 2) + 1):
                    for j in range((self.visible_radius * 2) + 1):
                        mask[pos[0] + i - self.visible_radius][pos[1] + j - self.visible_radius] = 1

                if np.random.uniform(0, 1) < 0.7:
                    previous_direction = self.random_direction()

                pos = (min(max(previous_direction[0] + pos[0], 1), 26),
                       min(max(previous_direction[1] + pos[1], 1), 26))

                if pos[0] == 0 or pos[0] == 27 or pos[1] == 0 or pos[1] == 27:
                    previous_direction = self.random_direction()

            # apply the mask to the image, but invisible pixels are random values between 0 and 1
            if self.add_noise:
                noise = np.clip(np.random.normal(0.0, 1, (28, 28)), -1, 1)
            else:
                noise = np.zeros((28, 28)) - 1

            masked_image = np.where(mask == 1, image, noise)

            if self.debug:
                # display all three arrays for debugging
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                ax1.imshow(image, cmap='gray')
                ax2.imshow(mask, cmap='gray')
                ax3.imshow(masked_image, cmap='gray')
                plt.show()

            masked_images.append(masked_image)
            masks.append(mask)

        return np.array(masked_images), np.array(masks)

    def random_direction(self):
        n = random.randint(0, 3)
        return [
            (1, 0),
            (0, 1),
            (-1, 0),
            (0, -1)
        ][n]

