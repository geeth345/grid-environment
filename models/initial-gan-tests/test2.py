# I have adapted this code from a tutorial published by Udacity as part of their "Deep Learining Nanodegree" course.
# The original code can be found here:
# https://github.com/udacity/deep-learning-v2-pytorch/blob/master/gan-mnist/MNIST_GAN_Solution.ipynb
# Available under the MIT license.

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

plt.ion()  # interactive mode


# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 64


# subset mnist
class MNIST_subset(Dataset):
    def __init__(self, mnist_dataset, classes):
        self.data = []
        self.targets = []

        for i in range(len(mnist_dataset)):
            image, label = mnist_dataset[i]
            if label in classes:  # Filter for digit '1'
                self.data.append(image)
                self.targets.append(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


# convert data to torch.FloatTensor
transform = transforms.ToTensor()
# full mnist
full_mnist_data = datasets.MNIST(root='data', train=True,
                                    download=True, transform=transform)

# the actual training data, for intial testing, is a subset of the full mnist data (defined in classes)
train_data = MNIST_subset(full_mnist_data, [8])

# prepare data loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           num_workers=num_workers)


# obtain one batch of training images, creates an iterator from the data loaded
dataiter = iter(train_loader)
images, labels = next(dataiter)
images = images.numpy()

# plot the first image for reference
img = np.squeeze(images[0])
fig = plt.figure(figsize = (3,3))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
plt.show()


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is 1 x 28 x 28
            nn.Conv2d(1, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # state size. 128 x 14 x 14
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            # state size. 256 x 7 x 7
            nn.Conv2d(256, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. 1
        )


    def forward(self, input):
        return self.main(input)


class Generator(nn.Module):

    def __init__(self, input_dim):
        super(Generator, self).__init__()

        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(input_dim, 256, 7, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # state size. 256 x 7 x 7
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # state size. 128 x 14 x 14
            nn.ConvTranspose2d(128, 1, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. output_dim x 28 x 28
        )

    def forward(self, input):
        return self.main(input)



# Discriminator hyperparameters
d_input_dim = 1

# Generator hyperparameters
g_input_dim = 1
g_output_dim = 1

# instantiate discriminator and generator
D = Discriminator(d_input_dim)
G = Generator(g_input_dim, g_output_dim)


# check that they are as you expect
print(D)
print()
print(G)


# Calculate losses
def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # label smoothing
    if smooth:
        # smooth, real labels = 0.9
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)  # real labels = 1

    # numerically stable loss
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss


def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)  # fake labels = 0
    criterion = nn.BCEWithLogitsLoss()
    # calculate loss
    loss = criterion(D_out.squeeze(), labels)
    return loss



import torch.optim as optim

# Optimizers
lr = 0.002

# Create optimizers for the discriminator and generator
d_optimizer = optim.Adam(D.parameters(), lr)
g_optimizer = optim.Adam(G.parameters(), lr)

import pickle as pkl

# training hyperparams
num_epochs = 100

# keep track of loss and generated, "fake" samples
samples = []
losses = []

print_every = 400

# Get some fixed data for sampling. These are images that are held
# constant throughout training, and allow us to inspect the model's performance
sample_size = 16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, g_input_size))
fixed_z = torch.from_numpy(fixed_z).float()

# train the network
D.train()
G.train()
for epoch in range(num_epochs):

    for batch_i, (real_images, _) in enumerate(train_loader):

        batch_size = real_images.size(0)

        ## Important rescaling step ##
        real_images = real_images * 2 - 1  # rescale input images from [0,1) to [-1, 1)

        # ============================================
        #            TRAIN THE DISCRIMINATOR
        # ============================================

        d_optimizer.zero_grad()

        # 1. Train with real images

        # Compute the discriminator losses on real images
        # smooth the real labels
        D_real = D(real_images)
        d_real_loss = real_loss(D_real, smooth=True)

        # 2. Train with fake images

        # Generate fake images
        # gradients don't have to flow during this step
        with torch.no_grad():
            z = np.random.uniform(-1, 1, size=(batch_size, g_input_size))
            z = torch.from_numpy(z).float()
            fake_images = G(z)

        # Compute the discriminator losses on fake images
        D_fake = D(fake_images)
        d_fake_loss = fake_loss(D_fake)

        # add up loss and perform backprop
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # =========================================
        #            TRAIN THE GENERATOR
        # =========================================
        g_optimizer.zero_grad()

        # 1. Train with fake images and flipped labels

        # Generate fake images
        z = np.random.uniform(-1, 1, size=(batch_size, g_input_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)

        # Compute the discriminator losses on fake images
        # using flipped labels!
        D_fake = D(fake_images)
        g_loss = real_loss(D_fake)  # use real loss to flip labels

        # perform backprop
        g_loss.backward()
        g_optimizer.step()

        # Print some loss stats
        if batch_i % print_every == 0:
            # print discriminator and generator loss
            print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                epoch + 1, num_epochs, d_loss.item(), g_loss.item()))

    ## AFTER EACH EPOCH##
    # append discriminator loss and generator loss
    losses.append((d_loss.item(), g_loss.item()))

    # generate and save sample, fake images
    G.eval()  # eval mode for generating samples
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train()  # back to train mode

# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)


fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()
plt.show()


# helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7,7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28,28)), cmap='Greys_r')


# Load samples from generator, taken while training
with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

# -1 indicates final epoch's samples (the last in the list)
view_samples(-1, samples)
plt.show()

rows = 10 # split epochs into 10, so 100/10 = every 10 epochs
cols = 6
fig, axes = plt.subplots(figsize=(7,12), nrows=rows, ncols=cols, sharex=True, sharey=True)

for sample, ax_row in zip(samples[::int(len(samples)/rows)], axes):
    for img, ax in zip(sample[::int(len(sample)/cols)], ax_row):
        img = img.detach()
        ax.imshow(img.reshape((28,28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
plt.show()

# Save the Generator
torch.save(G.state_dict(), 'generator.pth')

# Save the Discriminator
torch.save(D.state_dict(), 'discriminator.pth')
#
# # Create the model instances
# D_loaded = Discriminator(input_size, d_hidden_size, d_output_size)
# G_loaded = Generator(z_size, g_hidden_size, g_output_size)
#
# # Load the state dictionaries
# D_loaded.load_state_dict(torch.load('discriminator.pth'))
# G_loaded.load_state_dict(torch.load('generator.pth'))
#
# # Make sure to call eval() to set dropout and batch normalization layers to evaluation mode
# D_loaded.eval()
# G_loaded.eval()

