# %%
import os

import numpy as np
import pandas as pd
import S0_read_save_nii as rsn

# %%
## Set data path
data_dir = './data/GRA0.1dict500test_retest/'  # Relative path

### Get subject list
subject_list = os.listdir('./Recon_all_Subjects/')  # Relative path
### Only keep folders
subject_list = [subj for subj in subject_list if os.path.isdir(os.path.join(data_dir, subj))]
### Remove 'fsaverage6'
subject_list = [subj for subj in subject_list if subj != 'fsaverage6']

# %%
### Set ROI mask files
ROI_name = ['Left_GPi', 'Right_GPi', 'M1']

ROI_mask_file = ['./mask/Left_GPi_MNI2mm_91_mask.nii.gz',  # Relative path
                 './mask/Right_GPi_MNI2mm_91_mask.nii.gz',  # Relative path
                 './mask/M1_mask_MNI91_2mm.nii.gz']  # Relative path

### Read mask files
ROI_mask = [rsn.read_nii(i) for i in ROI_mask_file]

### Combine GPi masks
GPi_mask = ROI_mask[0] + ROI_mask[1]
GPi_mask = GPi_mask > 0

### Set M1 mask
M1_mask = ROI_mask[2]
M1_mask = M1_mask > 0

### Reset mask names and corresponding data
ROI_name = ['GPi', 'M1']
ROI_mask = [GPi_mask, M1_mask]

# %%
### Define two dataframes to store results
cbeta_mean_GPi = pd.DataFrame(columns=['subject', 'condition', 'test', 'retest'])
cbeta_mean_M1 = pd.DataFrame(columns=['subject', 'condition', 'test', 'retest'])

for subj in subject_list:
    subj_dir = os.path.join(data_dir, subj)
    conditions = os.listdir(subj_dir)
    
    ### Only keep folders
    conditions = [cond for cond in conditions if os.path.isdir(os.path.join(subj_dir, cond))]
    
    for cond in conditions:
        cond_dir = os.path.join(subj_dir, cond)
        test_file = os.path.join(cond_dir, 'test_cbeta_snm_rep.nii.gz')
        retest_file = os.path.join(cond_dir, 'retest_cbeta_snm_rep.nii.gz')
        
        if not os.path.exists(test_file) or not os.path.exists(retest_file):
            print(f'{subj} {cond} does not have test or retest file')
            continue

        ### Read data
        test_data = rsn.read_nii(test_file)
        retest_data = rsn.read_nii(retest_file)

        ### Calculate mean for ROI
        for i, roi in enumerate(ROI_name):
            test_mean = np.mean(test_data[ROI_mask[i]])
            retest_mean = np.mean(retest_data[ROI_mask[i]])
            temp_df = pd.DataFrame({'subject': subj, 'condition': cond, 'test': test_mean, 'retest': retest_mean}, index=[0])
            
            if roi == 'GPi':
                cbeta_mean_GPi = pd.concat([cbeta_mean_GPi, temp_df], ignore_index=True)
            elif roi == 'M1':
                cbeta_mean_M1 = pd.concat([cbeta_mean_M1, temp_df], ignore_index=True)

### Save results
cbeta_mean_GPi.to_csv('./new_data_1210/cbeta_mean_GPi.csv', index=False)  # Relative path
cbeta_mean_M1.to_csv('./new_data_1210/cbeta_mean_M1.csv', index=False)  # Relative path
