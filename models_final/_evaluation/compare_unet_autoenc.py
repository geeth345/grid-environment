# draw a plot of the training progress of the autoencoder and the unet

import pandas as pd
import matplotlib.pyplot as plt

# load the data
autoenc = pd.read_csv('../autoencoder_new/metrics.csv')
unet = pd.read_csv('../unet_mse/metrics.csv')

# rolling averages
autoenc['loss_rolling'] = autoenc['GeneratorLoss']#.rolling(window=1).mean()
unet['loss_rolling'] = unet['GeneratorLoss']#.rolling(window=1).mean()

# plot the data for the first 1000 epochs
plt.figure(figsize=(10, 5))
plt.plot(autoenc['loss_rolling'][:500], label='Autoencoder')
plt.plot(unet['loss_rolling'][:500], label='U-Net')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.title('Training Progress of Autoencoder and U-Net')
plt.savefig('eval_charts/autoenc_unet_comparison.png')
plt.close()