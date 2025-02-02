# %%
import os
import numpy as np
import pandas as pd
import ants
import nibabel as nib
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import S0_read_save_nii as rsn

# %%
### Set the data path
data_dir = './FC_change_task_ROI_volume/FC/FC_Indi_GPi_M1/'

out_dir = os.path.join(data_dir, 'compare_FC')
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

mni128_template = './MNI2mm_template.nii.gz'
mni128_template = ants.image_read(mni128_template)
mni128_template_data = mni128_template.numpy()

#### 1. DBS (Deep Brain Stimulation)
DBS_data_dir = os.path.join(data_dir, 'DBS/PreRest')
DBS_subject_list = os.listdir(DBS_data_dir)

out_dir_dbs = os.path.join(out_dir, 'DBS')
if not os.path.exists(out_dir_dbs):
    os.makedirs(out_dir_dbs)

#### Read the GPi and M1 data separately
ROI = 'GPi'
ROI2 = 'M1'

DBS_data = []
DBS_data2 = []
for sub in DBS_subject_list:
    sub_dir = os.path.join(DBS_data_dir, sub)
    run_list = os.listdir(sub_dir)
    temp_data = []
    temp_data2 = []
    for run in run_list:
        run_dir = os.path.join(sub_dir, run)
        run_file = (glob(os.path.join(run_dir, f'{ROI}.nii.gz'))[0])
        run_file2 = (glob(os.path.join(run_dir, f'{ROI2}.nii.gz'))[0])
        temp_data.append(rsn.read_nii(run_file))
        temp_data2.append(rsn.read_nii(run_file2))
    ### Average the data across different runs
    temp_data = np.mean(temp_data, axis=0)
    temp_data2 = np.mean(temp_data2, axis=0)
    DBS_data.append(temp_data)
    DBS_data2.append(temp_data2)
    
DBS_data = np.array(DBS_data)
DBS_data2 = np.array(DBS_data2)
### Perform Fisher's z-transformation
DBS_data = np.arctanh(DBS_data)
DBS_data2 = np.arctanh(DBS_data2)
np.nan_to_num(DBS_data, copy=False)
np.nan_to_num(DBS_data2, copy=False)

### Perform non-zero testing for each voxel to get a non-zero mask
from scipy.stats import ttest_1samp

DBS_data_mask = np.zeros(DBS_data.shape[1:])
DBS_data_mask2 = np.zeros(DBS_data2.shape[1:])
for i in range(DBS_data.shape[1]):
    for j in range(DBS_data.shape[2]):
        for k in range(DBS_data.shape[3]):
            if np.sum(DBS_data[:, i, j, k] != 0) > 0:
                t, p = ttest_1samp(DBS_data[:, i, j, k], 0, alternative='greater')  # Consider only the case where values are greater than 0
                if p < 0.05:
                    DBS_data_mask[i, j, k] = 1
            if np.sum(DBS_data2[:, i, j, k] != 0) > 0:
                t, p = ttest_1samp(DBS_data2[:, i, j, k], 0, alternative='greater')
                if p < 0.05:
                    DBS_data_mask2[i, j, k] = 1

#### Save the mask
DBS_data_mask_file = os.path.join(out_dir_dbs, f'{ROI}_not0_mask.nii.gz')
DBS_data_mask_img = ants.from_numpy(DBS_data_mask, origin=mni128_template.origin, spacing=mni128_template.spacing, direction=mni128_template.direction)
ants.image_write(DBS_data_mask_img, DBS_data_mask_file)

DBS_data_mask_file2 = os.path.join(out_dir_dbs, f'{ROI2}_not0_mask.nii.gz')
DBS_data_mask_img2 = ants.from_numpy(DBS_data_mask2, origin=mni128_template.origin, spacing=mni128_template.spacing, direction=mni128_template.direction)
ants.image_write(DBS_data_mask_img2, DBS_data_mask_file2)


### Average the data across different subjects
DBS_data_mean = np.mean(DBS_data, axis=0)
### Save the data
DBS_data_mean_file = os.path.join(out_dir_dbs, f'{ROI}_mean.nii.gz')
DBS_data_mean_img = ants.from_numpy(DBS_data_mean, origin=mni128_template.origin, spacing=mni128_template.spacing, direction=mni128_template.direction)
ants.image_write(DBS_data_mean_img, DBS_data_mean_file)

DBS_data_mean2 = np.mean(DBS_data2, axis=0)
DBS_data_mean_file2 = os.path.join(out_dir_dbs, f'{ROI2}_mean.nii.gz')
DBS_data_mean_img2 = ants.from_numpy(DBS_data_mean2, origin=mni128_template.origin, spacing=mni128_template.spacing, direction=mni128_template.direction)
ants.image_write(DBS_data_mean_img2, DBS_data_mean_file2)

### Calculate the difference of the means
DBS_data_diff = DBS_data_mean - DBS_data_mean2
DBS_data_diff_file = os.path.join(out_dir_dbs, f'{ROI}_{ROI2}_diff.nii.gz')
DBS_data_diff_img = ants.from_numpy(DBS_data_diff, origin=mni128_template.origin, spacing=mni128_template.spacing, direction=mni128_template.direction)
ants.image_write(DBS_data_diff_img, DBS_data_diff_file)

#### Compare the GPi and M1 data, perform t-tests for each voxel
from scipy.stats import ttest_ind

DBS_data_diff = np.zeros(DBS_data.shape[1:])
DBS_data_diff_p = np.zeros(DBS_data.shape[1:])
for i in range(DBS_data.shape[1]):
    for j in range(DBS_data.shape[2]):
        for k in range(DBS_data.shape[3]):
            t, p = ttest_ind(DBS_data[:, i, j, k], DBS_data2[:, i, j, k])
            DBS_data_diff[i, j, k] = t
            DBS_data_diff_p[i, j, k] = p

DBS_data_diff_file = os.path.join(out_dir_dbs, f'{ROI}_{ROI2}_diff_t.nii.gz')
DBS_data_diff_img = ants.from_numpy(DBS_data_diff, origin=mni128_template.origin, spacing=mni128_template.spacing, direction=mni128_template.direction)
ants.image_write(DBS_data_diff_img, DBS_data_diff_file)

DBS_data_diff_p_file = os.path.join(out_dir_dbs, f'{ROI}_{ROI2}_diff_p.nii.gz')
DBS_data_diff_p_img = ants.from_numpy(DBS_data_diff_p, origin=mni128_template.origin, spacing=mni128_template.spacing, direction=mni128_template.direction)
ants.image_write(DBS_data_diff_p_img, DBS_data_diff_p_file)

# %%
### Process the results, retain only significant non-zero regions
DBS_data_diff = ants.image_read(DBS_data_diff_file)
DBS_data_diff_data = DBS_data_diff.numpy()

DBS_data_diff_p = ants.image_read(DBS_data_diff_p_file)
DBS_data_diff_p_data = DBS_data_diff_p.numpy()

DBS_data_mask = ants.image_read(DBS_data_mask_file)
DBS_data_mask_data = DBS_data_mask.numpy()

DBS_data_mask2 = ants.image_read(DBS_data_mask_file2)
DBS_data_mask_data2 = DBS_data_mask2.numpy()
### Combine masks
DBS_data_mask_data = DBS_data_mask_data + DBS_data_mask_data2
DBS_data_mask_data = DBS_data_mask_data > 0

DBS_data_diff_data = DBS_data_diff_data * DBS_data_mask_data
DBS_data_diff_p_data = DBS_data_diff_p_data * DBS_data_mask_data

DBS_data_diff_file = os.path.join(out_dir_dbs, f'{ROI}_{ROI2}_diff_t_not0_masked.nii.gz')
DBS_data_diff_img = ants.from_numpy(DBS_data_diff_data, origin=mni128_template.origin, spacing=mni128_template.spacing, direction=mni128_template.direction)
ants.image_write(DBS_data_diff_img, DBS_data_diff_file)

DBS_data_diff_p_file = os.path.join(out_dir_dbs, f'{ROI}_{ROI2}_diff_p_not0_masked.nii.gz')
DBS_data_diff_p_img = ants.from_numpy(DBS_data_diff_p_data, origin=mni128_template.origin, spacing=mni128_template.spacing, direction=mni128_template.direction)
ants.image_write(DBS_data_diff_p_img, DBS_data_diff_p_file)
