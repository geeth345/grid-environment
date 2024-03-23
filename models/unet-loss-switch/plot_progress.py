import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
# epoch,d_loss,d_acc,c_loss,c_acc,g_loss_disc,g_acc_disc,g_loss_clas,g_acc_clas
df = pd.read_csv('progress.csv')

# #Plot the loss
# plt.plot(df['d_loss'], label='Discriminator Loss')
# plt.plot(df['c_loss'], label='Classifier Loss')
# plt.plot(df['g_loss_disc'], label='Generator Loss (Discriminator)')
# plt.plot(df['g_loss_clas'], label='Generator Loss (Classifier)')
# plt.legend()
# plt.show()
#
# # Plot the accuracy
# plt.plot(df['d_acc'], label='Discriminator Accuracy')
# plt.plot(df['c_acc'], label='Classifier Accuracy')
# plt.plot(df['g_acc_disc'], label='Generator Accuracy (Discriminator)')
# plt.plot(df['g_acc_clas'], label='Generator Accuracy (Classifier)')
# plt.legend()
# plt.show()


# Function to calculate the moving average
def moving_average(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()

# Apply moving average with a window size of N (e.g., 5)
window_size = 5
df_smoothed = df.copy()
df_smoothed['d_loss'] = moving_average(df['d_loss'], window_size)
df_smoothed['c_loss'] = moving_average(df['c_loss'], window_size)
df_smoothed['g_loss_disc'] = moving_average(df['g_loss_disc'], window_size)
df_smoothed['g_loss_clas'] = moving_average(df['g_loss_clas'], window_size)
df_smoothed['d_acc'] = moving_average(df['d_acc'], window_size)
df_smoothed['c_acc'] = moving_average(df['c_acc'], window_size)
df_smoothed['g_acc_disc'] = moving_average(df['g_acc_disc'], window_size)
df_smoothed['g_acc_clas'] = moving_average(df['g_acc_clas'], window_size)

# Plot the smoothed loss
plt.plot(df_smoothed['d_loss'], label='Discriminator Loss')
plt.plot(df_smoothed['c_loss'], label='Classifier Loss')
plt.plot(df_smoothed['g_loss_disc'], label='Generator Loss (Discriminator)')
plt.plot(df_smoothed['g_loss_clas'], label='Generator Loss (Classifier)')
plt.legend()
plt.show()

# Plot the smoothed accuracy
plt.plot(df_smoothed['d_acc'], label='Discriminator Accuracy')
plt.plot(df_smoothed['c_acc'], label='Classifier Accuracy')
plt.plot(df_smoothed['g_acc_disc'], label='Generator Accuracy (Discriminator)')
plt.plot(df_smoothed['g_acc_clas'], label='Generator Accuracy (Classifier)')
plt.legend()
plt.show()
