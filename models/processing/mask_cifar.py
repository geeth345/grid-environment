# adapting mask2 to work with images with multiple colour channels

# Code uses parts from original work: https://github.com/JamesHarcourt7/autoencoder-perception/blob/main/utils.py
# Original code author: James Harcourt

import random
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


RANDOM_WALK_LENGTH_MIN = 200
RANDOM_WALK_LENGTH_MAX = 700
VISIBLE_RADIUS = 2
DIRECTION_CHANGE_CHANCE = 1.0
DEBUG = False

class Mask:

    def __init__(self, walk_length_min=RANDOM_WALK_LENGTH_MIN, walk_length_max=RANDOM_WALK_LENGTH_MAX,
                 visible_radius=VISIBLE_RADIUS, direction_change_chance=DIRECTION_CHANGE_CHANCE, debug=DEBUG):
        self.walk_length_min = walk_length_min
        self.walk_length_max = walk_length_max
        self.visible_radius = visible_radius
        self.direction_change_chance = direction_change_chance
        self.debug = debug


    def mask(self, images):

        masked_images = []
        masks = []

        image_w = images[0].shape[0]
        image_h = images[0].shape[1]

        for image in tqdm(images):

            # check sizes are the same
            assert image.shape == (image_w, image_h, 3)

            mask = np.zeros((image_w, image_h)).astype(int)

            steps = np.random.randint(self.walk_length_min, self.walk_length_max + 1)

            # define starting position for the random walk based on dimensions and visible radius
            bound_w = (0 + self.visible_radius, image_w - (1 + self.visible_radius))
            bound_h = (0 + self.visible_radius, image_h - (1 + self.visible_radius))
            pos = np.random.randint(bound_w[0], bound_w[1]), np.random.randint(bound_h[0], bound_h[1])

            previous_direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

            for _ in range(steps):
                for i in range((self.visible_radius * 2) + 1):
                    for j in range((self.visible_radius * 2) + 1):
                        mask[pos[0] + i - self.visible_radius][pos[1] + j - self.visible_radius] = 1

                if random.random() < self.direction_change_chance:
                    previous_direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

                pos = np.clip(pos[0] + previous_direction[0], bound_w[0], bound_w[1]), np.clip(pos[1] + previous_direction[1], bound_h[0], bound_h[1])

                # if on edge turn around
                if pos[0] == bound_w[0] or pos[0] == bound_w[1] or pos[1] == bound_h[0] or pos[1] == bound_h[1]:
                    previous_direction = -previous_direction[0], -previous_direction[1]

            bg = np.zeros((image_w, image_h, 3))

            mask = np.expand_dims(mask, axis=-1)
            masked_image = np.where(mask == 1, image, bg)

            if self.debug:
                # display all three arrays for debugging
                fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                ax1.imshow((image * 0.5) + 0.5)
                ax2.imshow(mask)
                ax3.imshow((masked_image * 0.5) + 0.5)
                plt.show()

            masked_images.append(masked_image)
            masks.append(mask)

        return masked_images, masks









