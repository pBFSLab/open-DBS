# %%
import os
import numpy as np
import pandas as pd
import ants
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns

# %%
### Read data for plotting
dice_data_file = './data/DBS_2024/Parcellation152_test_Retest/demotion_DICE/DICE_test_retest_long.csv'  # Relative path
dice_data = pd.read_csv(dice_data_file)

print(dice_data.head())

# %%
# Create the plot with 'subject' on the x-axis
fig, ax = plt.subplots(figsize=(54, 20))  # size in inches (width x height)

types = dice_data['type'].unique()  # Get the unique values of 'type'
colors = {'indi': (1, 0.4, 0.4) , 'other': 'grey', 'other_average': (0.4, 0.4, 1)}  # Define a color for each 'type'
markers = {'indi': 'o', 'other': 'o', 'other_average': 'o'}  # Define a marker for each 'type'
sizes = {'indi': 50, 'other': 24, 'other_average': 50}  # Define a marker size for each 'type'
transparencies = {'indi': 1, 'other': 0.5, 'other_average': 1}  # Define transparency for each 'type'

# Iterate over the 'types' to plot
for subject_type in types:
    subset = dice_data[dice_data['type'] == subject_type]
    sns.stripplot(x='subject', y='DICE', data=subset, color=colors[subject_type], marker=markers[subject_type], 
                  label=subject_type, ax=ax, size=sizes[subject_type], alpha=transparencies[subject_type],
                  jitter=0.3)  # Add jitter to better visualize the data points

# Set axis properties
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.5, 1])  # Set y-axis ticks
ax.set_yticklabels([0, 0.5, 1])  # Custom labels
ax.set_xlabel('')
ax.set_ylabel('')
# Hide x and y axis tick labels
ax.set_xticklabels([])
ax.set_yticklabels([])

# If you want to keep the ticks, you can adjust tick parameters
ax.tick_params(axis='both', which='major', length=50, width=12)

### Keep only the left and bottom borders of the plot, set color to black
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# Set the width of the axes
ax.spines['left'].set_linewidth(10)
ax.spines['bottom'].set_linewidth(10)

# Hide the legend
ax.get_legend().remove()

# Show the plot
plt.show()

# %%
### Extract data for 'indi' and 'other_average'
indi_data = dice_data[dice_data['type'] == 'indi']
other_data = dice_data[dice_data['type'] == 'other_average']

### Calculate the mean and standard deviation for 'indi' and 'other_average'
indi_mean = indi_data['DICE'].mean()
indi_std = indi_data['DICE'].std()

other_mean = other_data['DICE'].mean()
other_std = other_data['DICE'].std()

print(f'indi_mean: {indi_mean}, indi_std: {indi_std}')
print(f'other_mean: {other_mean}, other_std: {other_std}')

# %%
### Perform paired t-test on 'indi' and 'other_average'
from scipy.stats import ttest_rel
t_statistic, p_value = ttest_rel(indi_data['DICE'], other_data['DICE'])
print(f't_statistic: {t_statistic}, p_value: {p_value}')

# %%
### Read data for plotting
dice_data_file = './data/DBS_2024/test_Retest/FC_corr/Parc152_Results/Parc_FC_demotion/Intra_Simi_rh_108_matrix_long.csv'  # Relative path
dice_data = pd.read_csv(dice_data_file)

print(dice_data.head())

# %%
# Create the plot with 'subject' on the x-axis
fig, ax = plt.subplots(figsize=(54, 20))  # size in inches (width x height)

types = dice_data['type'].unique()  # Get the unique values of 'type'
colors = {'indi': (1, 0.4, 0.4) , 'other': 'grey', 'other_average': (0.4, 0.4, 1)}  # Define a color for each 'type'
markers = {'indi': 'o', 'other': 'o', 'other_average': 'o'}  # Define a marker for each 'type'
sizes = {'indi': 50, 'other': 24, 'other_average': 50}  # Define a marker size for each 'type'
transparencies = {'indi': 1, 'other': 0.5, 'other_average': 1}  # Define transparency for each 'type'

# Iterate over the 'types' to plot
for subject_type in types:
    subset = dice_data[dice_data['type'] == subject_type]
    sns.swarmplot(x='subject', y='FC_corr', data=subset, color=colors[subject_type], marker=markers[subject_type], 
                  label=subject_type, ax=ax, size=sizes[subject_type], alpha=transparencies[subject_type])  # Add jitter to better visualize the data points

# Set axis properties
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.5, 1])  # Set y-axis ticks
ax.set_yticklabels([0, 0.5, 1])  # Custom labels
ax.set_xlabel('')
ax.set_ylabel('')
# Hide x and y axis tick labels
ax.set_xticklabels([])
ax.set_yticklabels([])

# If you want to keep the ticks, you can adjust tick parameters
ax.tick_params(axis='both', which='major', length=50, width=12)

### Keep only the left and bottom borders of the plot, set color to black
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('black')
ax.spines['bottom'].set_color('black')

# Set the width of the axes
ax.spines['left'].set_linewidth(10)
ax.spines['bottom'].set_linewidth(10)

# Hide the legend
ax.get_legend().remove()

# Show the plot
plt.show()

# %%
### Extract data for 'indi' and 'other_average'
indi_data = dice_data[dice_data['type'] == 'indi']
other_data = dice_data[dice_data['type'] == 'other_average']

### Calculate the mean and standard deviation for 'indi' and 'other_average'
indi_mean = indi_data['FC_corr'].mean()
indi_std = indi_data['FC_corr'].std()

other_mean = other_data['FC_corr'].mean()
other_std = other_data['FC_corr'].std()

print(f'indi_mean: {indi_mean}, indi_std: {indi_std}')
print(f'other_mean: {other_mean}, other_std: {other_std}')

# %%
### Perform paired t-test on 'indi' and 'other_average'
from scipy.stats import ttest_rel
t_statistic, p_value = ttest_rel(indi_data['FC_corr'], other_data['FC_corr'])
print(f't_statistic: {t_statistic}, p_value: {p_value}')
