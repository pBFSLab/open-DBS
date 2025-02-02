import os
import pandas as pd
import numpy as np

### set the path
data_dir = './task/GRA0.1dict500test_retest/GRA0.1dict500test_retest'  # Relative path
sub_list = os.listdir(data_dir)

### sort the subject list
sub_list.sort()

### Keep only directories
sub_list = [sub for sub in sub_list if os.path.isdir(os.path.join(data_dir, sub))]

# %%
### Read data for each subject separately
sub_data = []
for iter, sub in enumerate(sub_list):
    temp_data = []
    data1 = pd.read_csv(os.path.join(data_dir, sub, 'corrTestRetest.txt'), sep=',')
    temp_data.append(data1.iloc[0, 0])
    data2 = pd.read_csv(os.path.join(data_dir, sub, 'corrCombo.txt'), sep=',')
    temp_data = np.concatenate([temp_data, data2.iloc[0, :].values])
    
    ### Remove 1 from temp_data
    temp_data = np.array(temp_data)
    temp_data = temp_data[temp_data != 1]
    print(temp_data.shape)
    sub_data.append(temp_data)

### Convert the data to a DataFrame
sub_data = np.array(sub_data)

# %%
### Convert the data to a DataFrame
sub_data = pd.DataFrame(sub_data, index=sub_list)

### Add a 'mean' column
sub_data['mean'] = sub_data.iloc[:, 1:].mean(axis=1)

### Convert to long format
sub_data = sub_data.stack().reset_index()
sub_data.columns = ['sub', 'type', 'corr']

#### The first column corresponds to the 'other' data for subjects, 
#### and the second column onwards corresponds to inter-subject data
sub_data['type'] = sub_data['type'].replace({0: 'indi', 1: 'other', 2: 'other', 3: 'other', 4: 'other', 5: 'other', 6: 'other', 7: 'other', 8: 'other', 
                                             9: 'other', 10: 'other', 11: 'other', 12: 'other', 13: 'other', 'mean': 'other_average'})

# %%
### Sort sub_data by 'sub' and 'type'
sub_data = sub_data.sort_values(by=['sub', 'type'])

### Reset index
sub_data = sub_data.reset_index(drop=True)

### Save the data
sub_data.to_csv('./task/GRA0.1dict500test_retest/GRA0.1dict500test_retest.csv', index=False)  # Relative path

# %%
## Set the data path
data_csv = './task/GRA0.1dict500test_retest/GRA0.1dict500test_retest.csv'  # Relative path
## Read the data
data = pd.read_csv(data_csv)

# %%
### Extract data where type is 'indi' and 'other_average'
data_indi = data[data['type'] == 'indi']
data_other = data[data['type'] == 'other_average']

### Calculate mean and standard deviation for each group
data_indi_mean = data_indi["corr"].mean()
data_other_mean = data_other["corr"].mean()

data_indi_std = data_indi["corr"].std()
data_other_std = data_other["corr"].std()

print(f'indi_mean: {data_indi_mean}, indi_std: {data_indi_std}')
print(f'other_mean: {data_other_mean}, other_std: {data_other_std}')

# %%
### paired t-test
from scipy.stats import ttest_rel

t_stat, p_val = ttest_rel(data_indi["corr"], data_other["corr"])
print(f't_stat: {t_stat}, p_val: {p_val}')
