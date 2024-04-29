import pandas as pd
import matplotlib.pyplot as plt

# load the metrics file
metrics = pd.read_csv('metrics.csv')

# calculate rolling averages of the metrics
for col in metrics.columns:
    metrics[col + '_rolling'] = metrics[col].rolling(window=50).mean()

# metrics['cnn_accuracy'] = metrics['cnn_accuracy'].rolling(window=50).mean()

# plot the metrics
fig, ax = plt.subplots(1, 2, figsize=(15, 5))


ax[1].plot(metrics['MSE_rolling'], label='MSE')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('MSE')
ax[1].set_title('Mean Squared Error (MSE) vs Epoch')
ax[1].legend()


ax[0].plot(metrics['PSNR_rolling'], label='PSNR')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('PSNR')
ax[0].set_title('Peak Signal-to-Noise Ratio (PSNR) vs Epoch')
ax[0].legend()




plt.savefig('eval_images/accuracy.png')
plt.close()




