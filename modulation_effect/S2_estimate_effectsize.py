from utils.utils import *


def fc2matrix_with_effectsize(FC_columns, effect_size_dict):
	"""
	Convert FC columns to a matrix with effect sizes.
	"""
	fc_matrix_flatten = []
	for fc in FC_columns:
		if fc in effect_size_dict.keys():
			value = effect_size_dict[fc]
			fc_matrix_flatten.append(value)
		else:
			fc_matrix_flatten.append(0)
	feature_idx = np.tril_indices(211, k=-1)
	fc_matrix = np.zeros((211, 211))
	for i in range(len(FC_columns)):
		fc_matrix[feature_idx[0][i], feature_idx[1][i]] = np.round(fc_matrix_flatten[i], 2)
		fc_matrix[feature_idx[1][i], feature_idx[0][i]] = np.round(fc_matrix_flatten[i], 2)
	return fc_matrix


def cohens_d(group1, group2):
	"""
	Calculate Cohen's d effect size.
	"""
	mean1, mean2 = np.mean(group1), np.mean(group2)
	std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
	n1, n2 = len(group1), len(group2)
	pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))
	d = (mean1 - mean2) / pooled_std
	return d


def fc2matrix(FC_columns, select_fc):
	"""
	Convert FC columns to a binary matrix (1 for selected, 0 for unselected).
	"""
	fc_matrix_flatten = []
	for fc in FC_columns:
		if fc in select_fc:
			fc_matrix_flatten.append(1)
		else:
			fc_matrix_flatten.append(0)
	feature_idx = np.tril_indices(211, k=-1)
	fc_matrix = np.zeros((211, 211))
	for i in range(len(FC_columns)):
		fc_matrix[feature_idx[0][i], feature_idx[1][i]] = np.round(fc_matrix_flatten[i], 2)
		fc_matrix[feature_idx[1][i], feature_idx[0][i]] = np.round(fc_matrix_flatten[i], 2)
	return fc_matrix


if __name__ == '__main__':
	work_dir = '/path/to/your/workspace/effectsize'
	if not os.path.exists(work_dir):
		os.makedirs(work_dir)

	"""
	1. Data Preparation: DBS dataset
	"""
	# same as S1, pass

	"""
	2. Load edges that meet the criteria and save corresponding results
	"""
	# load data
	norm_cols = []
	abnorm_cols = []
	print(f"norm_cols: {len(norm_cols)}, {norm_cols}")
	print(f"abnorm_cols: {len(abnorm_cols)}, {abnorm_cols}")

	"""
	3. Calculate effect size for the selected edges
	"""
	all_cols = norm_cols + abnorm_cols
	all_effec_size_dict = {}
	all_effec_size = []
	for fc in all_cols:
		"""
		Calculate effect size using Cohen's d
		"""
		# ON
		d = cohens_d(PD_fc_pre[fc], PD_fc_on[fc])
		all_effec_size.append(np.abs(d))
		all_effec_size_dict[fc] = d

	print(f"all_cols: {len(all_cols)}, {all_cols}")
	print(f"all_effec_size: {len(all_effec_size)}, {all_effec_size}")

	"""
	4. Plot matrix
	"""
	# Reorder matrix
	mot_index = 'Mot parcel index in Parc152'
	vis_index = 'Vis parcel index in Parc152'
	# Visualize the matrix for the selected edges, with the order [Motor, Visual]
	temp_dict = all_effec_size_dict
	name = 'ON'
	temp_fc_mat = fc2matrix_with_effectsize(FC_columns, temp_dict)
	temp_mot_vis_mat = temp_fc_mat[mot_index + vis_index, :][:, mot_index + vis_index]

	# Save matrix
	save_path = f'{work_dir}/sig_mot_vis_mat.npy'
	np.save(save_path, temp_mot_vis_mat)

	# Create a figure and axis object
	fig, ax = plt.subplots()
	fig.patch.set_facecolor('none')  # Set the figure background to transparent
	ax.set_facecolor('none')  # Set the axis background to transparent
	ax = sns.heatmap(temp_mot_vis_mat, vmin=-1.5, vmax=1.5, cmap="coolwarm", annot=False, linewidths=0.5,
	                 linecolor='gray')
	plt.title(f"{name}")

	# Add horizontal and vertical lines to separate motor and visual regions
	ax.axhline(y=len(mot_index), color='black', linestyle='-')
	ax.axvline(x=len(mot_index), color='black', linestyle='-')

	# Remove x and y axis ticks and labels
	ax.set_xticks([])
	ax.set_xticklabels([])
	ax.set_yticks([])
	ax.set_yticklabels([])

	# Add small rectangles around non-zero values
	for i in range(temp_mot_vis_mat.shape[0]):
		for j in range(temp_mot_vis_mat.shape[1]):
			if temp_mot_vis_mat[i, j] != 0:  # If the value is not zero
				xlim = [j, j + 1]
				ylim = [i, i + 1]
				ax.add_patch(plt.Rectangle((xlim[0], ylim[0]), xlim[1] - xlim[0], ylim[1] - ylim[0], fill=False,
				                           edgecolor='black', lw=1))

	# Save the plot
	plt.savefig(f"{work_dir}/{name}_matrix.png", transparent=True)
	plt.close()
