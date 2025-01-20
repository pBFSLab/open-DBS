from utils.utils import *

def weight_map(sub_name, work_dir):
    set_environ()
    work_dir = f"{work_dir}/weight_map"
    os.makedirs(work_dir, exist_ok=True)

    """
    1. Prepare score information
    """
    clinical_info = pd.read_csv('/path/to/your/workspace/clinical_info/DBS_UPDRSIII_paired.csv')
    print(clinical_info.head(), clinical_info.columns.tolist())

    """
    2. Weight by score change
    """
    FC_lh_path = glob('/path/to/your/workspace/VTA_FCs/lh_Net_fcmap_*_*.mgh')
    print(len(FC_lh_path))  # N=350
    FC_lh_path = [FC_lh_p for FC_lh_p in FC_lh_path if os.path.basename(FC_lh_p).split('_')[-1].strip('.mgh') != sub_name]
    print(len(FC_lh_path))  # N=350 - sub

    lh_fc_weighted_all = np.zeros((40962, len(FC_lh_path)))
    rh_fc_weighted_all = np.zeros((40962, len(FC_lh_path)))
    weights_all = np.zeros((len(FC_lh_path),))
    for i, lh_path in enumerate(FC_lh_path):
        rh_path = lh_path.replace('lh', 'rh')
        sub = os.path.basename(lh_path).split('_')[-2]
        parm = os.path.basename(lh_path).split('_')[-1].strip('.mgh')
        print(f"sub:{sub}, parm:{parm}")

        # (1) Load FC
        lh_fc = nib.load(lh_path).get_fdata().reshape(-1, order='F')
        rh_fc = nib.load(rh_path).get_fdata().reshape(-1, order='F')
        lh_fc_weighted_all[:, i] = lh_fc
        rh_fc_weighted_all[:, i] = rh_fc

        # (2) Find weight
        sub_id = f"{parm}"
        rows_with_keyword = clinical_info[
            clinical_info.apply(lambda row: any(sub_id in str(item) for item in row), axis=1)
        ]
        print(rows_with_keyword.shape)

        weight_score_CH = rows_with_keyword['UPDRSIII_CH_Change_rate'].values.mean()
        weight_score_CL = rows_with_keyword['UPDRSIII_CL_Change_rate'].values.mean()
        weight_score_CV = rows_with_keyword['UPDRSIII_CV_Change_rate'].values.mean()

        weight_score_CH_1m = rows_with_keyword[rows_with_keyword['Subject_ID_CH'].str.contains('1m')][
            'UPDRSIII_CH_Change_rate'].values.mean()
        weight_score_CL_1m = rows_with_keyword[rows_with_keyword['Subject_ID_CL'].str.contains('1m')][
            'UPDRSIII_CL_Change_rate'].values.mean()
        weight_score_CV_1m = rows_with_keyword[rows_with_keyword['Subject_ID_CV'].str.contains('1m')][
            'UPDRSIII_CV_Change_rate'].values.mean()
        if np.isnan(weight_score_CH_1m) or np.isnan(weight_score_CL_1m) or np.isnan(weight_score_CV_1m):
            weight_score_CH_1m = weight_score_CH
            weight_score_CL_1m = weight_score_CL
            weight_score_CV_1m = weight_score_CV
        weight_score = np.nanmean([weight_score_CH_1m, weight_score_CL_1m, weight_score_CV_1m])

        # (3) Weight
        weights_all[i] = weight_score

    # (4) Average
    lh_fc_weighted_average = np.average(lh_fc_weighted_all, axis=1, weights=weights_all)
    rh_fc_weighted_average = np.average(rh_fc_weighted_all, axis=1, weights=weights_all)
    print(lh_fc_weighted_average.shape, rh_fc_weighted_average.shape)

    # 4. Mask medial wall
    lh_net1_path = '/path/to/your/7networks/lh_network_1_fs6.mgh'
    rh_net1_path = '/path/to/your/7networks/rh_network_1_fs6.mgh'
    lh_net1 = nib.load(lh_net1_path).get_fdata().reshape(-1, order='F')
    rh_net1 = nib.load(rh_net1_path).get_fdata().reshape(-1, order='F')
    lh_fc_weighted_average[lh_net1 != 0] = 0
    rh_fc_weighted_average[rh_net1 != 0] = 0

    # 5. Save mgh
    lh_file_path = f'{work_dir}/lh_weight_map.mgh'
    rh_file_path = f'{work_dir}/rh_weight_map.mgh'
    save_mgh(lh_fc_weighted_average, lh_file_path)
    save_mgh(rh_fc_weighted_average, rh_file_path)


def R_map(sub_name, work_dir):
    set_environ()
    work_dir = f"{work_dir}/R_map"
    os.makedirs(work_dir, exist_ok=True)

    """
    1. Prepare score information
    """
    clinical_info = pd.read_csv('/path/to/your/workspace/clinical_info/DBS_UPDRSIII_CH_OFF_paired.csv')
    print(clinical_info.head(), clinical_info.columns.tolist())

    """
    2. Correlation between FC and score
    """
    FC_lh_path = glob('/path/to/your/workspace/FC_DBSControl_STN_Andy/lh_Net_fcmap_*_*.mgh')
    print(len(FC_lh_path))  # N=350
    FC_lh_path = [FC_lh_p for FC_lh_p in FC_lh_path if os.path.basename(FC_lh_p).split('_')[-1].strip('.mgh') != sub_name]
    print(len(FC_lh_path))  # N=350 - sub

    lh_fc_all = np.zeros((40962, len(FC_lh_path)))
    rh_fc_all = np.zeros((40962, len(FC_lh_path)))
    score_all = []
    for i, lh_path in enumerate(FC_lh_path):
        rh_path = lh_path.replace('lh', 'rh')
        sub = os.path.basename(lh_path).split('_')[-2]
        parm = os.path.basename(lh_path).split('_')[-1].strip('.mgh')
        print(f"sub:{sub}, parm:{parm}")

        # (1) Load FC
        lh_fc = nib.load(lh_path).get_fdata().reshape(-1, order='F')
        rh_fc = nib.load(rh_path).get_fdata().reshape(-1, order='F')

        # (2) Find score
        sub_id = f"{parm}"
        rows_with_keyword = clinical_info[
            clinical_info.apply(lambda row: any(sub_id in str(item) for item in row), axis=1)
        ]

        score_CH = rows_with_keyword['UPDRSIII_CH_Change_rate'].values.mean()
        score_CL = rows_with_keyword['UPDRSIII_CL_Change_rate'].values.mean()
        score_CV = rows_with_keyword['UPDRSIII_CV_Change_rate'].values.mean()

        score_CH_1m = rows_with_keyword[rows_with_keyword['Subject_ID_CH'].str.contains('1m')][
            'UPDRSIII_CH_Change_rate'].values.mean()
        score_CL_1m = rows_with_keyword[rows_with_keyword['Subject_ID_CL'].str.contains('1m')][
            'UPDRSIII_CL_Change_rate'].values.mean()
        score_CV_1m = rows_with_keyword[rows_with_keyword['Subject_ID_CV'].str.contains('1m')][
            'UPDRSIII_CV_Change_rate'].values.mean()
        if np.isnan(score_CH_1m) or np.isnan(score_CL_1m) or np.isnan(score_CV_1m):
            score_CH_1m = score_CH
            score_CL_1m = score_CL
            score_CV_1m = score_CV

        score = np.mean([score_CH_1m, score_CL_1m, score_CV_1m])
        print(f"search keywords:{sub_id}, score:{score}")

        # (3) FC and score
        lh_fc_all[:, i] = lh_fc
        rh_fc_all[:, i] = rh_fc
        score_all.append(score)

    # (4) Calculate the correlation for each voxel to get the R-map
    r_map_lh = np.zeros((40962,))
    r_map_rh = np.zeros((40962,))
    for vol_i in range(40962):
        lh_fc_temp = lh_fc_all[vol_i, :]
        rh_fc_temp = rh_fc_all[vol_i, :]
        lh_r, _ = calculate_corr_pearson(lh_fc_temp, score_all)
        rh_r, _ = calculate_corr_pearson(rh_fc_temp, score_all)
        r_map_lh[vol_i] = lh_r
        r_map_rh[vol_i] = rh_r
    print(r_map_lh.shape, r_map_rh.shape)

    # 4. Mask medial wall
    lh_net1_path = '/path/to/your/7networks/lh_network_1_fs6.mgh'
    rh_net1_path = '/path/to/your/7networks/rh_network_1_fs6.mgh'
    lh_net1 = nib.load(lh_net1_path).get_fdata().flatten()
    rh_net1 = nib.load(rh_net1_path).get_fdata().flatten()
    r_map_rh[rh_net1 != 0] = 0
    r_map_lh[lh_net1 != 0] = 0

    # 5. Save mgh
    rh_file_path = f'{work_dir}/rh_R_map.mgh'
    lh_file_path = f'{work_dir}/lh_R_map.mgh'
    save_mgh(r_map_rh, rh_file_path)
    save_mgh(r_map_lh, lh_file_path)


def ideal_map(base_dir, sub):
    set_environ()
    work_dir = f'{base_dir}/ideal_map'
    os.makedirs(work_dir, exist_ok=True)

    lh_weight_map_path = f'{base_dir}/weight_map/lh_weight_map.mgh'
    rh_weight_map_path = f'{base_dir}/weight_map/rh_weight_map.mgh'

    lh_R_map_path = f'{base_dir}/R_map/lh_R_map.mgh'
    rh_R_map_path = f'{base_dir}/R_map/rh_R_map.mgh'

    lh_weight_map = nib.load(lh_weight_map_path).get_fdata().reshape(-1, order='F')
    rh_weight_map = nib.load(rh_weight_map_path).get_fdata().reshape(-1, order='F')
    lh_R_map = nib.load(lh_R_map_path).get_fdata().reshape(-1, order='F')
    rh_R_map = nib.load(rh_R_map_path).get_fdata().reshape(-1, order='F')

    """
    This third map was computed by masking the weighted average maps by the R maps (R > 0 for positive values and R < 0 for negative values).
    """
    product_lh = lh_weight_map * lh_R_map
    product_rh = rh_weight_map * rh_R_map

    lh_ideal_map = lh_weight_map.copy()
    rh_ideal_map = rh_weight_map.copy()

    lh_ideal_map[product_lh <= 0] = np.nan
    rh_ideal_map[product_rh <= 0] = np.nan

    # 4. Mask medial wall
    lh_net1_path = '/path/to/your/7networks/lh_network_1_fs6.mgh'
    rh_net1_path = '/path/to/your/7networks/rh_network_1_fs6.mgh'
    lh_net1 = nib.load(lh_net1_path).get_fdata().flatten()
    rh_net1 = nib.load(rh_net1_path).get_fdata().flatten()
    rh_ideal_map[rh_net1 != 0] = 0
    lh_ideal_map[lh_net1 != 0] = 0

    # 5. Save mgh
    rh_file_path = f'{work_dir}/rh_ideal_map.mgh'
    lh_file_path = f'{work_dir}/lh_ideal_map.mgh'
    save_mgh(rh_ideal_map, rh_file_path)
    save_mgh(lh_ideal_map, lh_file_path)


def similarity_info(clinical_info, base_dir):
    set_environ()
    work_dir = f'{base_dir}/predict'
    os.makedirs(work_dir, exist_ok=True)

    """
    2. Calculate spatial similarity
    """
    # (1) Normalize ideal map
    norm_idealmap_lh_path = f'{base_dir}/ideal_map/lh_ideal_map.mgh'
    norm_idealmap_rh_path = f'{base_dir}/ideal_map/rh_ideal_map.mgh'
    norm_idealmap_lh = nib.load(norm_idealmap_lh_path).get_fdata().reshape(-1, order='F')
    norm_idealmap_rh = nib.load(norm_idealmap_rh_path).get_fdata().reshape(-1, order='F')
    norm_ideamap = np.hstack([norm_idealmap_lh, norm_idealmap_rh])

    # Medial wall
    lh_net1_path = '/path/to/your/7networks/lh_network_1_fs6.mgh'
    rh_net1_path = '/path/to/your/7networks/rh_network_1_fs6.mgh'
    lh_net1 = nib.load(lh_net1_path).get_fdata().flatten()
    rh_net1 = nib.load(rh_net1_path).get_fdata().flatten()

    # (3) Calculate similarity
    for idx, row in clinical_info.iterrows():
        sub = row['Subject']

        """
        Find the corresponding VTA map for the subject in the healthy controls
        """
        FC_lh_path = glob('/path/to/your/workspace/FC_DBSControl_STN_Andy/lh_Net_fcmap_*_*.mgh')
        print(len(FC_lh_path))  # N=350
        FC_lh_path = [FC_lh_p for FC_lh_p in FC_lh_path if sub in FC_lh_p]
        print(len(FC_lh_path))  # N=25 sub

        lh_fc_all = np.zeros((40962, len(FC_lh_path)))
        rh_fc_all = np.zeros((40962, len(FC_lh_path)))
        for i, lh_path in enumerate(FC_lh_path):
            rh_path = lh_path.replace('lh', 'rh')
            sub = os.path.basename(lh_path).split('_')[-2]
            parm = os.path.basename(lh_path).split('_')[-1].strip('.mgh')
            print(f"sub:{sub}, parm:{parm}")

            # (1) Load FC
            lh_fc = nib.load(lh_path).get_fdata().reshape(-1, order='F')
            rh_fc = nib.load(rh_path).get_fdata().reshape(-1, order='F')
            lh_fc_all[:, i] = lh_fc
            rh_fc_all[:, i] = rh_fc
        indi_map_lh = lh_fc_all.mean(axis=-1)
        indi_map_rh = rh_fc_all.mean(axis=-1)

        indi_map = np.hstack([indi_map_lh, indi_map_rh])

        indi_map = indi_map[~np.isnan(norm_ideamap)]
        norm_ideamap_temp = norm_ideamap[~np.isnan(norm_ideamap)]

        # Similarity
        norm_corr, _ = calculate_corr_spearman(indi_map, norm_ideamap_temp)
        clinical_info.loc[idx, 'norm_similarity'] = norm_corr

    print(clinical_info)
    clinical_info.to_csv(f"{work_dir}/DBS_with_similarity.csv", index=False)

    return clinical_info