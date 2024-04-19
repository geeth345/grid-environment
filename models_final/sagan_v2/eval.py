import pandas as pd
import matplotlib.pyplot as plt

# load the metrics file
metrics = pd.read_csv('metrics.csv')

# calculate rolling averages of the metrics
for col in metrics.columns:
    metrics[col + '_rolling'] = metrics[col].rolling(window=20).mean()

metrics['cnn_accuracy'] = metrics['cnn_accuracy'].rolling(window=50).mean()

# plot the metrics
fig, ax = plt.subplots(2, 2, figsize=(15, 10))

ax[0, 0].plot(metrics['d_real_auth_acc_rolling'], label='real')
ax[0, 0].plot(metrics['d_fake_auth_acc_rolling'], label='fake')
ax[0, 0].set_xlabel('Epoch')
ax[0, 0].set_ylabel('Accuracy')
ax[0, 0].set_title('Discriminator Accuracy (Real / Fake)')
ax[0, 0].legend()

ax[0, 1].plot(metrics['d_real_auth_acc_rolling'], label='real')
ax[0, 1].plot(metrics['d_fake_auth_acc_rolling'], label='fake')
ax[0, 1].plot(metrics['g_auth_acc_rolling'], label='generator')
ax[0, 1].set_xlabel('Epoch')
ax[0, 1].set_ylabel('Accuracy')
ax[0, 1].set_title('Discriminator Accuracy and Generator Inverse Accuracy')
ax[0, 1].legend()

ax[1, 0].plot(metrics['cnn_accuracy'], label='generator')
ax[1, 0].set_xlabel('Epoch')
ax[1, 0].set_ylabel('Accuracy')
ax[1, 0].set_title('Accuracy of Pre-Trained CNN on Generated Images')
ax[1, 0].legend()

ax[1, 1].plot(metrics['d_real_class_acc_rolling'], label='real')
ax[1, 1].plot(metrics['d_fake_class_acc_rolling'], label='fake')
ax[1, 1].plot(metrics['g_class_acc_rolling'], label='generator')
ax[1, 1].set_xlabel('Epoch')
ax[1, 1].set_ylabel('Accuracy')
ax[1, 1].set_title('Discriminator Accuracy and Generator Accuracy (Image Class)')
ax[1, 1].legend()



plt.savefig('eval_images/accuracy.png')
plt.close()






