from utils.utils import *


if __name__ == '__main__':
    # Load clinical information for DBS patients
    DBS_Patient_clinical_info = pd.read_csv('/path/to/your/data/DBS_Pre_14.csv')
    DBS_Patient_subject_list = DBS_Patient_clinical_info['Subject_ID'].tolist()
    result_base_path = '/path/to/your/FC_DBSPatient_VTA'

    work_dir = '/path/to/your/workspace/DBSPatient'
    os.makedirs(work_dir, exist_ok=True)

    # 1. Build group-level results and visualize individual results
    fs6_n_vertex = 40962
    subject_list = DBS_Patient_subject_list
    fc_map_group_rh = np.zeros((fs6_n_vertex, len(subject_list)))
    fc_map_group_lh = np.zeros((fs6_n_vertex, len(subject_list)))
    group_size = 0
    for indx, sub in enumerate(subject_list):
        sub = sub.split('_')[0]
        fc_map_path_rh = f'{result_base_path}/rh_Net_fcmap_{sub}_STN_VTA.mgh'
        fc_map_path_lh = f'{result_base_path}/lh_Net_fcmap_{sub}_STN_VTA.mgh'
        fc_map_rh = nib.load(fc_map_path_rh).get_fdata().flatten()
        fc_map_lh = nib.load(fc_map_path_lh).get_fdata().flatten()
        fc_map_group_rh[:, indx] = fc_map_rh
        fc_map_group_lh[:, indx] = fc_map_lh
        group_size += 1
        print(group_size)

        """
        Visualize individual results
        """
        # (1). Mask medial wall
        lh_net1_path = '/path/to/your/7networks/lh_network_1_fs6.mgh'
        rh_net1_path = '/path/to/your/7networks/rh_network_1_fs6.mgh'
        lh_net1 = nib.load(lh_net1_path).get_fdata().flatten()
        rh_net1 = nib.load(rh_net1_path).get_fdata().flatten()
        fc_map_rh[rh_net1 != 0] = 0
        fc_map_lh[lh_net1 != 0] = 0

        # (2). Save mgh
        rh_file_path = f'{work_dir}/rh_Net_{sub}_VTA_fcmap.mgh'
        lh_file_path = f'{work_dir}/lh_Net_{sub}_VTA_fcmap.mgh'
        save_mgh(fc_map_rh, rh_file_path)
        save_mgh(fc_map_lh, lh_file_path)



    # 2. Save matrices
    print(group_size, fc_map_group_rh.shape, fc_map_group_lh.shape)
    rh_mat_path = f'{work_dir}/rh_Net_VTA_fcmap_group.mat'
    lh_mat_path = f'{work_dir}/lh_Net_VTA_fcmap_group.mat'
    save_array_to_mat(fc_map_group_rh, 'fc', rh_mat_path)
    save_array_to_mat(fc_map_group_lh, 'fc', lh_mat_path)

    # 3. Calculate group mean results
    fc_map_group_rh_mean = np.mean(fc_map_group_rh, axis=-1)
    fc_map_group_lh_mean = np.mean(fc_map_group_lh, axis=-1)
    print(group_size, fc_map_group_rh_mean.shape, fc_map_group_lh_mean.shape)

    # 4. Mask medial wall
    lh_net1_path = '/path/to/your/7networks/lh_network_1_fs6.mgh'
    rh_net1_path = '/path/to/your/7networks/rh_network_1_fs6.mgh'
    lh_net1 = nib.load(lh_net1_path).get_fdata().flatten()
    rh_net1 = nib.load(rh_net1_path).get_fdata().flatten()
    fc_map_group_rh_mean[rh_net1 != 0] = 0
    fc_map_group_lh_mean[lh_net1 != 0] = 0

    # 5. Save mgh
    rh_file_path = f'{work_dir}/rh_Net_VTA_fcmap_group.mgh'
    lh_file_path = f'{work_dir}/lh_Net_VTA_fcmap_group.mgh'
    save_mgh(fc_map_group_rh_mean, rh_file_path)
    save_mgh(fc_map_group_lh_mean, lh_file_path)
