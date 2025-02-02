import os
import numpy as np
import pandas as pd
import ants
import nibabel as nib

# Set data path
data_dir = './DBS_OPEN/results/ParFC'  # Relative path
subject_list = [f'DBS{i:02d}' for i in range(1, 15)]

# Set save directory
save_dir = './data/DBS_2024/test_Retest/FC_corr/Parc152_Results/Parc_FC/'  # Relative path
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Read the test-retest data and organize it into a suitable format for plotting
# The csv file does not contain row and column names
correlation_2 = pd.read_csv(os.path.join(save_dir, 'Intra_Simi_152matrix.csv'), header=None)

# For plotting, data needs to be in long format: 
# The first column is subject, the second is 'indi', 'other', or 'other_average'; 
# the third column is DICE; the fourth column is the subject's ID
FC_corr_df_long = []
for iter, subject in enumerate(subject_list):
    other_sum = 0
    for iter2, subject_2 in enumerate(subject_list):
        dice = correlation_2.iloc[iter, iter2]
        if subject == subject_2:
            FC_corr_df_long.append([subject, 'indi', dice, iter + 1])
        else:
            FC_corr_df_long.append([subject, 'other', dice, iter + 1])
            other_sum += dice
    FC_corr_df_long.append([subject, 'other_average', other_sum / (len(subject_list) - 1), iter + 1])

FC_corr_df_long = pd.DataFrame(FC_corr_df_long, columns=['subject', 'type', 'FC_corr', 'subject_id'])

# Save data
FC_corr_df_long.to_csv(os.path.join(save_dir, 'Intra_Simi_rh_108_matrix_long.csv'))

# Calculate the mean of 'indi' and 'other' in FC_corr_df_long
FC_corr_df_long_mean = FC_corr_df_long[FC_corr_df_long['type'] == 'indi']['FC_corr'].mean()
print(FC_corr_df_long_mean)

FC_corr_df_long_mean = FC_corr_df_long[FC_corr_df_long['type'] == 'other']['FC_corr'].mean()
print(FC_corr_df_long_mean)
