import pandas as pd
import matplotlib.pyplot as plt

# load the metrics file
metrics = pd.read_csv('metrics.csv')

# create a new column for PSNR moving average
metrics['PSNR_MA'] = metrics['PSNR'].rolling(window=10).mean()

# plot PSNR
plt.plot(metrics['PSNR_MA'])
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.title('PSNR vs Epoch')
plt.show()

