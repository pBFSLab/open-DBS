from utils.utils import *


def dice_similarity_coefficient(arr1, arr2):
	"""
    Calculate the Dice similarity coefficient between two binary arrays.

    Parameters:
    arr1: The first binary array.
    arr2: The second binary array.

    Returns:
    dice_coeff: The Dice similarity coefficient.
    """
	# Ensure arrays are binary
	arr1 = np.asarray(arr1 > 0).astype(bool)
	arr2 = np.asarray(arr2 > 0).astype(bool)

	# Calculate intersection
	intersection = np.sum(arr1 & arr2)

	# Calculate union
	sum_elements = np.sum(arr1) + np.sum(arr2)

	# Calculate Dice similarity coefficient
	dice_coeff = (2. * intersection) / sum_elements if sum_elements != 0 else 0

	return dice_coeff


if __name__ == '__main__':
	set_environ()

	work_dir = '/path/to/your/workspace/similarity'
	os.makedirs(work_dir, exist_ok=True)

	# 4. Mask medial wall
	lh_net1_path = '/path/to/your/7networks/lh_network_1_fs6.mgh'
	rh_net1_path = '/path/to/your/7networks/rh_network_1_fs6.mgh'
	lh_net1 = nib.load(lh_net1_path).get_fdata().flatten()
	rh_net1 = nib.load(rh_net1_path).get_fdata().flatten()

	pre_op_VTA_path_lh = '/path/to/your/DBSPatient/rh_Net_VTA_fcmap_group.mgh'
	pre_op_VTA_path_rh = '/path/to/your/DBSPatient/rh_Net_VTA_fcmap_group.mgh'

	after_op_VTA_path_lh = '/path/to/your/DBSPatient_afterDBS/lh_Net_VTA_fcmap_group.mgh'
	after_op_VTA_path_rh = '/path/to/your/DBSPatient_afterDBS/rh_Net_VTA_fcmap_group.mgh'

	pre_op_VTA_lh = nib.load(pre_op_VTA_path_lh).get_fdata().flatten()
	pre_op_VTA_rh = nib.load(pre_op_VTA_path_rh).get_fdata().flatten()
	pre_op_VTA_lh[lh_net1 != 0] = 0
	pre_op_VTA_rh[rh_net1 != 0] = 0
	pre_op_VTA = np.concatenate((pre_op_VTA_lh, pre_op_VTA_rh), axis=0)

	after_op_VTA_lh = nib.load(after_op_VTA_path_lh).get_fdata().flatten()
	after_op_VTA_rh = nib.load(after_op_VTA_path_rh).get_fdata().flatten()
	after_op_VTA_lh[lh_net1 != 0] = 0
	after_op_VTA_rh[rh_net1 != 0] = 0
	after_op_VTA = np.concatenate((after_op_VTA_lh, after_op_VTA_rh), axis=0)

	r, p = calculate_corr_pearson(pre_op_VTA, after_op_VTA)
	print(f"Pearson correlation: r={r:.4f}, p={p:.4f}")
	r, p = calculate_corr_spearman(pre_op_VTA, after_op_VTA)
	print(f"Spearman correlation: r={r:.4f}, p={p:.4f}")

	dice_coeff = dice_similarity_coefficient(pre_op_VTA, after_op_VTA)
	print(f'Dice similarity coefficient: {dice_coeff:.4f}')