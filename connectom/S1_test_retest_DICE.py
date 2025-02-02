# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import ants
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import S0_read_save_mgh as rsm

def calculate_DICE(data_1, data_2):
    '''
    data_1: np.ndarray
    data_2: np.ndarray

    description: calculate the average DICE coefficient between two data sets
    '''
    ### Remove zero values
    data_1 = data_1[data_1 != 0]
    data_2 = data_2[data_2 != 0]

    equal_data = data_1 == data_2
    dice = 2 * np.sum(equal_data) / (len(data_1) + len(data_2))

    return dice

## Set data paths (relative paths), which contain the test and retest parcellation results
data_dir_lh = './data/open_DBS/Parcellation152/WB_lh'
data_dir_rh = './data/open_DBS/Parcellation152/WB_rh'

## Set subject list
subject_list = os.listdir('./data/Recon_all_Subjects')
### Keep only directories
subject_list = [subject for subject in subject_list if os.path.isdir(os.path.join('./data/Recon_all_Subjects', subject))]
### Remove 'fsaverage6'
subject_list = [subject for subject in subject_list if subject != 'fsaverage6']

## Set save directory
save_dir = './data/open_DBS/Parcellation152_test_Retest/demotion_DICE'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

## Record non-mean DICE values
dice_df = pd.DataFrame(columns=subject_list, index=subject_list)

## Iterate over each subject
for subject in subject_list:
    
    ## Get test and retest files
    test_file_lh = glob(os.path.join(data_dir_lh, f'{subject}_Test/Combine', f'*{subject}*fs6.mgh'))
    if len(test_file_lh) == 0:
        print(f'{subject} does not have test file')
        continue
    test_file_lh = test_file_lh[0]

    test_file_rh = glob(os.path.join(data_dir_rh, f'{subject}_Test/Combine', f'*{subject}*fs6.mgh'))
    if len(test_file_rh) == 0:
        print(f'{subject} does not have test file')
        continue
    test_file_rh = test_file_rh[0]

    for subject_2 in subject_list:
        another_file_lh = glob(os.path.join(data_dir_lh, f'{subject_2}_Retest/Combine', f'*{subject_2}*fs6.mgh'))
        another_file_rh = glob(os.path.join(data_dir_rh, f'{subject_2}_Retest/Combine', f'*{subject_2}*fs6.mgh'))

        if len(another_file_lh) == 0:
            print(f'{subject_2} does not have test or retest file')
            continue

        another_file_lh = another_file_lh[0]

        if len(another_file_rh) == 0:
            print(f'{subject_2} does not have test or retest file')
            continue

        another_file_rh = another_file_rh[0]

        test_data_lh = rsm.read_mgh(test_file_lh)
        test_data_rh = rsm.read_mgh(test_file_rh)
        test_data = np.concatenate((test_data_lh, test_data_rh), axis=0)

        retest_data_lh = rsm.read_mgh(another_file_lh)
        retest_data_rh = rsm.read_mgh(another_file_rh)
        retest_data = np.concatenate((retest_data_lh, retest_data_rh), axis=0)

        dice = calculate_DICE(test_data, retest_data)
        dice_df.loc[subject, subject_2] = dice

dice_df.to_csv(os.path.join(save_dir, 'DICE_test_retest.csv'))

## Read the test_retest2 data and reshape it for plotting
dice_df = pd.read_csv(os.path.join(save_dir, 'DICE_test_retest.csv'), index_col=0)
### Reshape data to long format suitable for plotting, with columns: subject, type, DICE, subject_id
dice_df_long = []
for iter, subject in enumerate(subject_list):
    other_sum = 0
    for subject_2 in subject_list:
        dice = dice_df.loc[subject, subject_2]
        if subject == subject_2:
            dice_df_long.append([subject, 'indi', dice, iter + 1])
        else:
            dice_df_long.append([subject, 'other', dice, iter + 1])
            other_sum += dice
    dice_df_long.append([subject, 'other_average', other_sum / (len(subject_list) - 1), iter + 1])

dice_df_long = pd.DataFrame(dice_df_long, columns=['subject', 'type', 'DICE', 'subject_id'])
## Save the long format data
dice_df_long.to_csv(os.path.join(save_dir, 'DICE_test_retest_long.csv'))
