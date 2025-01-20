import nibabel as nib # version 5.2.1
from sklearn.decomposition import PCA # version 1.4.2
from filters.filters import bandpass_nifti


def regressor_PCA_singlebold(pca_data, n):
    pca = PCA(n_components=n, random_state=False)
    pca_regressor = pca.fit_transform(pca_data.T)
    return pca_regressor

def regressors_PCA(data, maskpath):
    '''
    Generate PCA regressor from outer points of brain.
        bold_path - path. Path of bold.
        maskpath - Path to file containing mask.
        outpath  - Path to file to place the output.
    '''
    # PCA parameter.
    n = 10

    # Open mask.
    mask_img = nib.load(maskpath)
    mask = mask_img.get_fdata().swapaxes(0, 1)
    mask = mask.flatten(order='F') == 0
    nvox = float(mask.sum())
    assert nvox > 0, 'Null mask found in %s' % maskpath

    data = data.swapaxes(0, 1)
    vol_data = data.reshape((data.shape[0] * data.shape[1] * data.shape[2], data.shape[3]), order='F')
    pca_data = vol_data[mask]
    pca_regressor = regressor_PCA_singlebold(pca_data, n)

    return pca_regressor


if __name__ == '__main__':
    aseg_brainmask_bin = 'label-brain_probseg.nii.gz' # binarized brainmask
    bold_path = 'sub-01_ses-01_task-rest_run-01_space-T1w_desc-preproc_bold.nii.gz' # preprocessed BOLD in T1w space

    img = nib.load(bold_path)
    data = img.get_fdata()
    nskip = 0
    tr = 2
    data_bandpass = bandpass_nifti(data, nskip=nskip, order=2, band=[0.01, 0.08], tr=tr)
    b_comp_cor = regressors_PCA(data_bandpass, str(aseg_brainmask_bin))
    print(b_comp_cor)
