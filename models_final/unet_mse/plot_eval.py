import pandas as pd
import matplotlib.pyplot as plt

# load the metrics file
metrics = pd.read_csv('metrics.csv')

# create a new column for PSNR moving average
metrics['PSNR_MA'] = metrics['PSNR'].rolling(window=50).mean()

# plot PSNR
plt.plot(metrics['PSNR_MA'])
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.title('PSNR vs Epoch')
plt.savefig('metrics.png')
plt.close()

# plot moving average for cnn accuracy
metrics['CNN_Accuracy_MA'] = metrics['CNN_Acc'].rolling(window=50).mean()
plt.plot(metrics['CNN_Accuracy_MA'])
plt.xlabel('Epoch')
plt.ylabel('CNN Accuracy')
plt.title('CNN Accuracy vs Epoch')
plt.savefig('cnn_accuracy.png')
plt.close()

# plot cross entropy loss for cnn
metrics['CNN_CE_MA'] = metrics['CNN_CE'].rolling(window=50).mean()
plt.plot(metrics['CNN_CE_MA'])
plt.xlabel('Epoch')
plt.ylabel('CNN Cross Entropy Loss')
plt.title('CNN Cross Entropy Loss vs Epoch')
plt.savefig('cnn_ce.png')

print("Done generating metrics plot")
