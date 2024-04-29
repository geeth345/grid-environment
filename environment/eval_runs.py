import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode
import numpy as np

models = {
    'u-net': 'run_data_dynamic/u-net.csv',
    'acgan': 'run_data_dynamic/acgan.csv',
   # 'u-net-dynamic': 'run_data_dynamic/u-net_dynamic.csv',
   # 'acgan-dynamic': 'run_data_dynamic/acgan_dynamic.csv',
}

# models = {
#     'u-net': 'run_data/u-net.csv',
#     'acgan' : 'run_data/acgan.csv',
# }

plot_steps = 200


# Choose one of the models to extract the baseline data
baseline_df = pd.read_csv(models['u-net'])  # Using u-net data for the baseline
baseline_df['naive_correct'] = baseline_df['naive_prediction'] == baseline_df['label']
baseline_consensus = baseline_df.groupby(['run', 'iter']).naive_correct.mean().reset_index()
average_baseline_consensus = baseline_consensus.groupby('iter').naive_correct.mean().reset_index()
average_baseline_consensus = average_baseline_consensus[average_baseline_consensus['iter'] <= plot_steps]

plt.figure(figsize=(10, 6))  # Initialize the plot outside the loop

# Plot the baseline performance
plt.plot(average_baseline_consensus['iter'], average_baseline_consensus['naive_correct'], linestyle='--',
         color='grey', label='baseline')

for model in models:
    df = pd.read_csv(models[model])
    df['correct_prediction'] = df['prediction'] == df['label']
    consensus_df = df.groupby(['run', 'iter']).correct_prediction.mean().reset_index()
    average_consensus_df = consensus_df.groupby('iter').correct_prediction.mean().reset_index()
    average_consensus_df = average_consensus_df[average_consensus_df['iter'] <= plot_steps]

    # Plotting model performance within the loop
    plt.plot(average_consensus_df['iter'], average_consensus_df['correct_prediction'], linestyle='-',
             label=model)

#Final plot settings and display
plt.title('Average Consensus % Over Iterations (Including Baseline)')
plt.xlabel('Time Steps')
plt.ylabel('Consensus %')
plt.legend(title='Model')  # Adding a legend to distinguish the models and baseline
plt.grid(True)
#plt.axvline(x=200, color='r', linestyle='--', label='Image Reset', linewidth=4)
#plt.show()
#plt.savefig('run_data_dynamic/consensus.png')
plt.savefig('run_data/consensus.png')
plt.close()

plt.figure(figsize=(10, 6))  # Initialize the plot outside the loop

for model in models:
    df = pd.read_csv(models[model])


    # Function to calculate the mode and consensus for majority label
    def majority_consensus(group):
        mode_result = mode(group['prediction'])
        if isinstance(mode_result.mode, np.ndarray):
            most_common = mode_result.mode[0]  # mode_result.mode is an array, access the first element
        else:
            most_common = mode_result.mode  # mode_result.mode is a scalar value
        consensus = (group['prediction'] == most_common).mean()
        return pd.Series({'majority_consensus': consensus})


    # Group by 'run' and 'iter', and apply the custom function
    majority_df = df.groupby(['run', 'iter']).apply(majority_consensus).reset_index()

    # Average the results over all runs for each iteration
    average_majority_df = majority_df.groupby('iter').majority_consensus.mean().reset_index()
    average_majority_df = average_majority_df[average_majority_df['iter'] <= plot_steps]

    # Plotting the result for the current model
    plt.plot(average_majority_df['iter'], average_majority_df['majority_consensus'], linestyle='-',
             label=model)

# Final plot settings and display
plt.title('Average Relative Majority Opinion % Over Iterations')
plt.xlabel('Time Steps')
plt.ylabel('Relative Majority Opinion %')
plt.legend(title='Model')  # Adding a legend to distinguish the models
#plt.axvline(x=200, color='r', linestyle='--', label='Image Reset', linewidth=4)
plt.grid(True)
#plt.savefig('run_data_dynamic/majority.png')
plt.savefig('run_data/majority.png')
#plt.show()
plt.close()