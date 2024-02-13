# simple viewer to inspect the masked images side by side with the original images

import numpy as np
import matplotlib.pyplot as plt


def load_mnist(filename):
    with open(filename, 'rb') as file:
        file.read(16)
        data = np.fromfile(file, dtype=np.uint8).reshape(-1, 28, 28)
    return data

if __name__ == '__main__':
    original_images = load_mnist('../../data/train-images-idx3-ubyte')
    masked_images = load_mnist('../../data/masked/masked-train-images-idx3-ubyte')

    for i in range(0, 10):
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(original_images[i], cmap='gray')
        ax2.imshow(masked_images[i], cmap='gray')
        plt.show()

