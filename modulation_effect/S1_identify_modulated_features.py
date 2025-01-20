from utils.utils import *

if __name__ == '__main__':
	work_dir = '/path/to/your/workspace/'
	if not os.path.exists(work_dir):
		os.makedirs(work_dir)

	name_mapping_dict = {'DBS_01': 1, 'DBS_02': 2, 'DBS_03': 3, 'DBS_04': 4, 'DBS_05': 5,
	                     'DBS_06': 6, 'DBS_07': 7, 'DBS_08': 8, 'DBS_09': 9,
	                     'DBS_10': 10,
	                     'DBS_11': 11, 'DBS_12': 12, 'DBS_13': 13, 'DBS_14': 14}

	"""
	1. Data Preparation: DBS dataset
	"""
	data_flag = 'DBS'
	HC_pFC_indi = pd.read_csv('/path/to/your/data/HC_pFC_indi_info_152.csv')
	HC_fc = np.arctanh(HC_pFC_indi)
	HC_fc['Subject_ID'] = [f'HC_{i + 1}' for i in range(len(HC_fc))]
	HC_fc['Subject'] = [f'HC_{i + 1}' for i in range(len(HC_fc))]
	HC_fc['Subject_num'] = [f'{i + 1}' for i in range(len(HC_fc))]
	print(f'DBS HC_fc: {HC_fc.shape}')

	PD_fc_off = pd.read_csv('/path/to/your/data/PD_fc_off_info_152.csv')
	PD_fc_off = np.arctanh(PD_fc_off)
	off_df = pd.read_csv('/path/to/your/data/PD_fc_off_UPDRSIII.csv')
	PD_fc_off['Subject_ID'] = off_df['Subject_ID']
	PD_fc_off['Subject'] = PD_fc_off['Subject_ID'].map(lambda x: x.split('_')[0])
	PD_fc_off['Subject_num'] = PD_fc_off['Subject'].map(name_mapping_dict)
	print(f'DBS PD_fc_off: {PD_fc_off.shape}')

	PD_fc_pre = pd.read_csv('/path/to/your/data/PD_fc_pre_info_152.csv')
	PD_fc_pre = np.arctanh(PD_fc_pre)
	pre_df = pd.read_csv('/path/to/your/data/PD_fc_pre.csv')
	PD_fc_pre['Subject_ID'] = pre_df['Subject_ID']
	PD_fc_pre['Subject'] = PD_fc_pre['Subject_ID'].map(lambda x: x.split('_')[0])
	PD_fc_pre['Subject_num'] = PD_fc_pre['Subject'].map(name_mapping_dict)
	print(f'DBS PD_fc_pre: {PD_fc_pre.shape}')

	PD_fc_on = pd.read_csv('/path/to/your/data/PD_fc_on_info_152.csv')
	PD_fc_on = np.arctanh(PD_fc_on)
	on_df = pd.read_csv('/path/to/your/data/PD_fc_on_UPDRSIII.csv')
	PD_fc_on['Subject_ID'] = on_df['Subject_ID']
	PD_fc_on['Subject'] = PD_fc_on['Subject_ID'].map(lambda x: x.split('_')[0])
	PD_fc_on['Subject_num'] = PD_fc_on['Subject'].map(name_mapping_dict)
	print(f'DBS PD_fc_on: {PD_fc_on.shape}')

	"""
	2. Result Saving Path
	"""
	save_dir = f"{work_dir}/{data_flag}"
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	"""
	3. Restrict Feature Range
	"""
	# only Mot and Vis
	FC_columns = HC_pFC_indi.columns.to_list()

	"""
	4. Filter edges that meet the criteria and save corresponding results
	"""
	norm_cols = []
	abnorm_cols = []
	for fc in FC_columns:
		t_pre, p_pre = calculate_ttest_indipendent(HC_fc[fc], PD_fc_pre[fc])
		t_on, p_on = calculate_ttest_indipendent(HC_fc[fc], PD_fc_on[fc])
		t_off, p_off = calculate_ttest_indipendent(HC_fc[fc], PD_fc_off[fc])
		t_norm, p_norm = calculate_ttest_indipendent(PD_fc_pre[fc], PD_fc_on[fc])

		p_threshold_level0 = 0.05
		# p_threshold_level1 = 0.01
		# p_threshold_level2 = 0.0001
		if (p_norm < p_threshold_level0) and (p_on > p_off) and (p_on > p_pre) and (p_pre < p_threshold_level0):
			norm_cols.append(fc)
		if (p_norm < p_threshold_level0) and (p_on < p_off) and (p_on < p_pre) and (p_on < p_threshold_level0):
			abnorm_cols.append(fc)

	print(f"norm_cols: {len(norm_cols)}, {norm_cols}")
	print(f"abnorm_cols: {len(abnorm_cols)}, {abnorm_cols}")

	"""
	5. Classify the found edges into hyper and hypo
		hyper = PD > HC
		hypo = PD < HC
	"""
	# Classify normally regulated edges
	norm_cols_hyper = []
	norm_cols_hypo = []
	norm_hc_fc_hyper = []
	norm_pre_fc_hyper = []
	norm_off_fc_hyper = []
	norm_on_fc_hyper = []

	norm_hc_fc_hypo = []
	norm_pre_fc_hypo = []
	norm_off_fc_hypo = []
	norm_on_fc_hypo = []
	for col in norm_cols:
		hc_mean = HC_fc[col].mean()
		pre_mean = PD_fc_pre[col].mean()
		off_mean = PD_fc_off[col].mean()
		on_mean = PD_fc_on[col].mean()

		if pre_mean > hc_mean:
			norm_cols_hyper.append(col)
			norm_hc_fc_hyper.append(hc_mean)
			norm_pre_fc_hyper.append(pre_mean)
			norm_off_fc_hyper.append(off_mean)
			norm_on_fc_hyper.append(on_mean)
		else:
			norm_cols_hypo.append(col)
			norm_hc_fc_hypo.append(hc_mean)
			norm_pre_fc_hypo.append(pre_mean)
			norm_off_fc_hypo.append(off_mean)
			norm_on_fc_hypo.append(on_mean)

	# Classify abnormally regulated edges
	abnorm_cols_hyper = []
	abnorm_cols_hypo = []
	abnorm_hc_fc_hyper = []
	abnorm_pre_fc_hyper = []
	abnorm_off_fc_hyper = []
	abnorm_on_fc_hyper = []

	abnorm_hc_fc_hypo = []
	abnorm_pre_fc_hypo = []
	abnorm_off_fc_hypo = []
	abnorm_on_fc_hypo = []
	for col in abnorm_cols:
		hc_mean = HC_fc[col].mean()
		pre_mean = PD_fc_pre[col].mean()
		off_mean = PD_fc_off[col].mean()
		on_mean = PD_fc_on[col].mean()

		if pre_mean > hc_mean:
			abnorm_cols_hyper.append(col)
			abnorm_hc_fc_hyper.append(hc_mean)
			abnorm_pre_fc_hyper.append(pre_mean)
			abnorm_off_fc_hyper.append(off_mean)
			abnorm_on_fc_hyper.append(on_mean)
		else:
			abnorm_cols_hypo.append(col)
			abnorm_hc_fc_hypo.append(hc_mean)
			abnorm_pre_fc_hypo.append(pre_mean)
			abnorm_off_fc_hypo.append(off_mean)
			abnorm_on_fc_hypo.append(on_mean)

	print(f"norm_cols_hyper: {len(norm_cols_hyper)}, {norm_cols_hyper}")
	print(f"norm_cols_hypo: {len(norm_cols_hypo)}, {norm_cols_hypo}")
	print(f"abnorm_cols_hyper: {len(abnorm_cols_hyper)}, {abnorm_cols_hyper}")
	print(f"abnorm_cols_hypo: {len(abnorm_cols_hypo)}, {abnorm_cols_hypo}")

	"""
	6. Visualize the results for the two categories of edges
	"""
	flags = ['hyper', 'hypo']
	for flag in flags:
		if flag == 'hyper':
			thr_norm = thr_abnorm = flag
			norm_cols = norm_cols_hyper
			abnorm_cols = abnorm_cols_hyper

			norm_hc_fc = norm_hc_fc_hyper
			norm_pre_fc = norm_pre_fc_hyper
			norm_off_fc = norm_off_fc_hyper
			norm_on_fc = norm_on_fc_hyper

			abnorm_hc_fc = abnorm_hc_fc_hyper
			abnorm_pre_fc = abnorm_pre_fc_hyper
			abnorm_off_fc = abnorm_off_fc_hyper
			abnorm_on_fc = abnorm_on_fc_hyper

		if flag == 'hypo':
			thr_norm = thr_abnorm = flag
			norm_cols = norm_cols_hypo
			abnorm_cols = abnorm_cols_hypo

			norm_hc_fc = norm_hc_fc_hypo
			norm_pre_fc = norm_pre_fc_hypo
			norm_off_fc = norm_off_fc_hypo
			norm_on_fc = norm_on_fc_hypo

			abnorm_hc_fc = abnorm_hc_fc_hypo
			abnorm_pre_fc = abnorm_pre_fc_hypo
			abnorm_off_fc = abnorm_off_fc_hypo
			abnorm_on_fc = abnorm_on_fc_hypo

		"""
		Bar plot
		"""
		norm_cols_cortex = []
		for feature in norm_cols:
			if ('hippo' not in feature) and ('basalG' not in feature) and ('thala' not in feature) and (
					'cereb' not in feature) and ('gp' not in feature):
				print(f"{feature} is cortex-cortex feature!")
				norm_cols_cortex.append(feature)
		print(f"norm_cols_cortex: {len(norm_cols_cortex)}, {norm_cols_cortex}")
		if len(norm_cols_cortex) != 0:
			HC_norm_fcs = HC_fc[norm_cols_cortex]
			HC_norm_fcs['dataset'] = ['HC'] * len(HC_norm_fcs)
			HC_norm_fcs['FCmean'] = HC_norm_fcs[norm_cols_cortex].mean(axis=1)
			HC_norm_fcs['Subject_num'] = HC_fc['Subject_num']

			Pre_norm_fcs = PD_fc_pre[norm_cols_cortex]
			Pre_norm_fcs['dataset'] = ['Pre'] * len(Pre_norm_fcs)
			Pre_norm_fcs['FCmean'] = Pre_norm_fcs[norm_cols_cortex].mean(axis=1)
			Pre_norm_fcs['Subject_num'] = PD_fc_pre['Subject_num']

			OFF_norm_fcs = PD_fc_off[norm_cols_cortex]
			OFF_norm_fcs['dataset'] = ['OFF'] * len(OFF_norm_fcs)
			OFF_norm_fcs['FCmean'] = OFF_norm_fcs[norm_cols_cortex].mean(axis=1)
			OFF_norm_fcs['Subject_num'] = PD_fc_off['Subject_num']

			ON_norm_fcs = PD_fc_on[norm_cols_cortex]
			ON_norm_fcs['dataset'] = ['ON'] * len(ON_norm_fcs)
			ON_norm_fcs['FCmean'] = ON_norm_fcs[norm_cols_cortex].mean(axis=1)
			ON_norm_fcs['Subject_num'] = PD_fc_on['Subject_num']

			# Additional datasets can be loaded and processed similarly as above

			"""HC Pre OFF ON"""
			all_norm_fc = pd.concat([HC_norm_fcs, Pre_norm_fcs, OFF_norm_fcs, ON_norm_fcs], ignore_index=True)
			print(all_norm_fc.head())

			# Group by 'dataset' and 'Subject_num' and calculate the mean
			all_norm_fc_subject = all_norm_fc.groupby(['dataset', 'Subject_num']).mean().reset_index()
			bar_plot(all_norm_fc_subject, 'dataset', 'FCmean', -0.25, 0.5, (3, 3),
			         0.25, save_dir, f'mean_norm_cols_{thr_norm}', order_list=['HC', 'Pre', 'OFF', 'ON'])
			all_norm_fc_subject.to_csv(f"{save_dir}/mean_norm_cols_{thr_norm}.csv", index=False)

			# Additional visualizations can be performed similarly for other datasets

		abnorm_cols_cortex = []
		for feature in abnorm_cols:
			if ('hippo' not in feature) and ('basalG' not in feature) and ('thala' not in feature) and (
					'cereb' not in feature) and ('gp' not in feature):
				print(f"{feature} is cortex-cortex feature!")
				abnorm_cols_cortex.append(feature)
		print(f"abnorm_cols_cortex: {len(abnorm_cols_cortex)}, {abnorm_cols_cortex}")
		if len(abnorm_cols_cortex) != 0:
			HC_abnorm_fcs = HC_fc[abnorm_cols_cortex]
			HC_abnorm_fcs['dataset'] = ['HC'] * len(HC_abnorm_fcs)
			HC_abnorm_fcs['FCmean'] = HC_abnorm_fcs[abnorm_cols_cortex].mean(axis=1)
			HC_abnorm_fcs['Subject_num'] = HC_fc['Subject_num']

			Pre_abnorm_fcs = PD_fc_pre[abnorm_cols_cortex]
			Pre_abnorm_fcs['dataset'] = ['Pre'] * len(Pre_abnorm_fcs)
			Pre_abnorm_fcs['FCmean'] = Pre_abnorm_fcs[abnorm_cols_cortex].mean(axis=1)
			Pre_abnorm_fcs['Subject_num'] = PD_fc_pre['Subject_num']

			OFF_abnorm_fcs = PD_fc_off[abnorm_cols_cortex]
			OFF_abnorm_fcs['dataset'] = ['OFF'] * len(OFF_abnorm_fcs)
			OFF_abnorm_fcs['FCmean'] = OFF_abnorm_fcs[abnorm_cols_cortex].mean(axis=1)
			OFF_abnorm_fcs['Subject_num'] = PD_fc_off['Subject_num']

			ON_abnorm_fcs = PD_fc_on[abnorm_cols_cortex]
			ON_abnorm_fcs['dataset'] = ['ON'] * len(ON_abnorm_fcs)
			ON_abnorm_fcs['FCmean'] = ON_abnorm_fcs[abnorm_cols_cortex].mean(axis=1)
			ON_abnorm_fcs['Subject_num'] = PD_fc_on['Subject_num']

			# Additional datasets can be loaded and processed similarly as above

			"""HC Pre OFF ON"""
			all_abnorm_fc = pd.concat([HC_abnorm_fcs, Pre_abnorm_fcs, OFF_abnorm_fcs, ON_abnorm_fcs], ignore_index=True)
			print(all_abnorm_fc.head())

			# Group by 'dataset' and 'Subject_num' and calculate the mean
			all_abnorm_fc_subject = all_abnorm_fc.groupby(['dataset', 'Subject_num']).mean().reset_index()
			bar_plot(all_abnorm_fc_subject, 'dataset', 'FCmean', -0.25, 0.5, (3, 3),
			         0.25, save_dir, f'mean_abnorm_cols_{thr_abnorm}', order_list=['HC', 'Pre', 'OFF', 'ON'])
			all_abnorm_fc_subject.to_csv(f"{save_dir}/mean_abnorm_cols_{thr_abnorm}.csv", index=False)

			# Additional visualizations can be performed similarly for other datasets
