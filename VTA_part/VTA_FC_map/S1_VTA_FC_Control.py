from utils.utils import *

if __name__ == '__main__':
    # Load clinical information for controls and patients
    DBS_control_clinical_info = pd.read_csv('/path/to/your/data/DBS_controls.csv')
    DBS_control_subject_list = DBS_control_clinical_info['Subject_ID'].tolist()
    DBS_Patient_clinical_info = pd.read_csv('/path/to/your/data/DBS_Patient.csv')
    DBS_Patient_subject_list = DBS_Patient_clinical_info['Subject_ID'].tolist()
    fc_base_path = '/path/to/your/FC_DBSControl_VTA'

    work_dir = '/path/to/your/workspace/Control'
    os.makedirs(work_dir, exist_ok=True)

    # 1. Visualize group-level results for each VTA
    fs6_n_vertex = 40962
    for VTA_sub in DBS_Patient_subject_list:
        VTA_sub = VTA_sub.split('_')[0]
        print(VTA_sub)
        subject_list = DBS_control_subject_list
        fc_map_group_rh = np.zeros((fs6_n_vertex, len(subject_list)))
        fc_map_group_lh = np.zeros((fs6_n_vertex, len(subject_list)))
        group_size = 0
        for indx, sub in enumerate(subject_list):
            fc_map_path_rh = f'{fc_base_path}/rh_Net_fcmap_{sub}_{VTA_sub}.mgh'
            fc_map_path_lh = f'{fc_base_path}/lh_Net_fcmap_{sub}_{VTA_sub}.mgh'
            fc_map_rh = nib.load(fc_map_path_rh).get_fdata().flatten()
            fc_map_lh = nib.load(fc_map_path_lh).get_fdata().flatten()
            fc_map_group_rh[:, indx] = fc_map_rh
            fc_map_group_lh[:, indx] = fc_map_lh
            group_size += 1
            print(group_size)

        # 2. Save matrices
        print(group_size, fc_map_group_rh.shape, fc_map_group_lh.shape)
        rh_mat_path = f'{work_dir}/rh_Net_{VTA_sub}_fcmap_group.mat'
        lh_mat_path = f'{work_dir}/lh_Net_{VTA_sub}_fcmap_group.mat'
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

        # 5. Save mgh files
        rh_file_path = f'{work_dir}/rh_Net_{VTA_sub}_fcmap_group.mgh'
        lh_file_path = f'{work_dir}/lh_Net_{VTA_sub}_fcmap_group.mgh'
        save_mgh(fc_map_group_rh_mean, rh_file_path)
        save_mgh(fc_map_group_lh_mean, lh_file_path)

    # 6. Get group-level visualization results
    subject_list = DBS_Patient_subject_list
    fc_map_group_rh = np.zeros((fs6_n_vertex, len(subject_list)))
    fc_map_group_lh = np.zeros((fs6_n_vertex, len(subject_list)))
    group_size = 0
    for indx, sub in enumerate(subject_list):
        sub = sub.split('_')[0]
        fc_map_path_rh = f'{work_dir}/rh_Net_{sub}_fcmap_group.mgh'
        fc_map_path_lh = f'{work_dir}/lh_Net_{sub}_fcmap_group.mgh'
        fc_map_rh = nib.load(fc_map_path_rh).get_fdata().flatten()
        fc_map_lh = nib.load(fc_map_path_lh).get_fdata().flatten()
        fc_map_group_rh[:, indx] = fc_map_rh
        fc_map_group_lh[:, indx] = fc_map_lh
        group_size += 1
        print(group_size)

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

    # 5. Save mgh files
    rh_file_path = f'{work_dir}/rh_Net_VTA_fcmap_group.mgh'
    lh_file_path = f'{work_dir}/lh_Net_VTA_fcmap_group.mgh'
    save_mgh(fc_map_group_rh_mean, rh_file_path)
    save_mgh(fc_map_group_lh_mean, lh_file_path)