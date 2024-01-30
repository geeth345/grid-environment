# modify the mnist dataset, for each image partially mask the image.
# the only visible part of the image should be what is visible from performing a random walk

import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt



# parameters
RANDOM_WALK_LENGTH = 40
VISIBLE_RADIUS = 5

# given a 28*28 numpy array, apply a mask and return the masked image
def mask(image, walk_length=RANDOM_WALK_LENGTH, visible_radius=VISIBLE_RADIUS):
    # assume the image is a 28x28 numpy array
    assert image.shape == (28, 28)

    # randomly select a starting position
    agent_position = (random.randint(7, 20), random.randint(7, 20))
    random_walk_steps = [random.randint(0, 3) for _ in range(RANDOM_WALK_LENGTH)]

    # create the mask
    mask = np.zeros((28, 28))
    for i in range(RANDOM_WALK_LENGTH):
        # update the agent position
        if random_walk_steps[i] == 0:
            agent_position = (agent_position[0] - 1, agent_position[1])
        elif random_walk_steps[i] == 1:
            agent_position = (agent_position[0], agent_position[1] + 1)
        elif random_walk_steps[i] == 2:
            agent_position = (agent_position[0] + 1, agent_position[1])
        elif random_walk_steps[i] == 3:
            agent_position = (agent_position[0], agent_position[1] - 1)

        # update the mask
        for x in range(agent_position[0] - VISIBLE_RADIUS, agent_position[0] + VISIBLE_RADIUS + 1):
            for y in range(agent_position[1] - VISIBLE_RADIUS, agent_position[1] + VISIBLE_RADIUS + 1):
                if 0 <= x < 28 and 0 <= y < 28:
                    mask[x][y] = 1

    # apply the mask
    masked_image = np.multiply(image, mask)

    # display all three arrays for debugging
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.imshow(image, cmap='gray')
    ax2.imshow(mask, cmap='gray')
    ax3.imshow(masked_image, cmap='gray')
    plt.show()

    return masked_image


# load the mnist dataset
def load_mnist(filename):
    with open(filename, 'rb') as file:
        file.read(16)
        data = np.fromfile(file, dtype=np.uint8).reshape(-1, 28, 28)
    return data

def save_modified_mnist(data, filename):
    magic_number = 2051
    num_images = data.shape[0]
    num_rows = data.shape[1]
    num_cols = data.shape[2]
    with open(filename, 'wb') as file:
        file.write(magic_number.to_bytes(4, 'big'))
        file.write(num_images.to_bytes(4, 'big'))
        file.write(num_rows.to_bytes(4, 'big'))
        file.write(num_cols.to_bytes(4, 'big'))
        bytes = data.tobytes()
        print(len(bytes))
        file.write(bytes)

if __name__ == "__main__":
    original_images = load_mnist('../data/train-images-idx3-ubyte')
    masked_images = np.array([mask(image) for image in tqdm(original_images)])
    print(masked_images.shape)
    save_modified_mnist(masked_images, '../data/masked/masked-train-images-idx3-ubyte')

