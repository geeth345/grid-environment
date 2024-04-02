import pandas as pd
import matplotlib.pyplot as plt

# load the metrics file
metrics = pd.read_csv('metrics.csv')

# plot psnr
metrics['PSNR_MA'] = metrics['psnr'].rolling(window=25).mean()
plt.plot(metrics['PSNR_MA'])
plt.xlabel('Epoch')
plt.ylabel('PSNR')
plt.title('PSNR vs Epoch')
plt.savefig('eval_images/psnr.png')
plt.close()

# plot d_real_acc and d_fake_acc
metrics['D_REAL_ACC_MA'] = metrics['d_real_acc'].rolling(window=25).mean()
metrics['D_FAKE_ACC_MA'] = metrics['d_fake_acc'].rolling(window=25).mean()
plt.plot(metrics['D_REAL_ACC_MA'], label='D Real Acc')
plt.plot(metrics['D_FAKE_ACC_MA'], label='D Fake Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Discriminator Accuracy vs Epoch')
plt.legend()
plt.savefig('eval_images/disc_accuracy.png')
plt.close()

# plot d_loss and g_loss
metrics['D_AVG_LOSS'] = (metrics['d_fake_loss'] + metrics['d_real_loss']) / 2
metrics['D_LOSS_MA'] = metrics['D_AVG_LOSS'].rolling(window=25).mean()
metrics['G_LOSS_MA'] = metrics['g_loss'].rolling(window=25).mean()
plt.plot(metrics['D_LOSS_MA'], label='D Loss')
plt.plot(metrics['G_LOSS_MA'], label='G Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Loss vs Epoch')
plt.legend()
plt.savefig('eval_images/loss.png')
plt.close()

# plot average of d_real_acc and d_fake_acc along with g_acc
metrics['D_AVG_ACC'] = (metrics['d_real_acc'] + metrics['d_fake_acc']) / 2
metrics['D_AVG_ACC_MA'] = metrics['D_AVG_ACC'].rolling(window=25).mean()
metrics['G_ACC_MA'] = metrics['g_acc'].rolling(window=25).mean()
plt.plot(metrics['D_AVG_ACC_MA'], label='D Avg Acc')
plt.plot(metrics['G_ACC_MA'], label='G Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Discriminator Accuracy and Generator Accuracy vs Epoch')
plt.legend()
plt.savefig('eval_images/avg_disc_gen_accuracy.png')
plt.close()

# plot all the accuracies
plt.plot(metrics['D_REAL_ACC_MA'], label='D Real Acc')
plt.plot(metrics['D_FAKE_ACC_MA'], label='D Fake Acc')
plt.plot(metrics['D_AVG_ACC_MA'], label='D Avg Acc')
plt.plot(metrics['G_ACC_MA'], label='G Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('All Accuracies vs Epoch')
plt.legend()
plt.savefig('eval_images/all_accuracies.png')
plt.close()







