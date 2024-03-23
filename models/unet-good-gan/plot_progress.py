import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
# epoch,d_loss,d_acc,c_loss,c_acc,g_loss_disc,g_acc_disc,g_loss_clas,g_acc_clas
df = pd.read_csv('progress.csv')

# Plot the loss
# plt.plot(df['d_loss'], label='Discriminator Loss')
# plt.plot(df['c_loss'], label='Classifier Loss')
# plt.plot(df['g_loss_disc'], label='Generator Loss (Discriminator)')
# plt.plot(df['g_loss_clas'], label='Generator Loss (Classifier)')
# plt.legend()
# plt.show()

# Plot the accuracy
plt.plot(df['d_acc'], label='Discriminator Accuracy')
plt.plot(df['c_acc'], label='Classifier Accuracy')
plt.plot(df['g_acc_disc'], label='Generator Accuracy (Discriminator)')
plt.plot(df['g_acc_clas'], label='Generator Accuracy (Classifier)')
plt.legend()
plt.show()
