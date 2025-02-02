# -*- coding: utf-8 -*-
import numpy as np
import nibabel as nib

def read_mgh(file):
    '''
    file: str, the path to the mgh file

    Return the data in the mgh file
    '''
    img = nib.load(file).get_fdata()
    if len(img.shape) == 4:
        img = img.reshape(-1, img.shape[-1], order='F')
    elif len(img.shape) == 3:
        img = img.reshape(-1, order='F')
    return img

def save_mgh(data, file):
    '''
    data: np.array, the data to be saved
    file: str, the path to save the data

    Save the data as a mgh file
    '''
    if data.dtype == np.float64 or data.dtype == bool or data.dtype == np.int64:
        data = data.astype('float32')
    ## check if there is nan in the data
    if np.isnan(data).any():
        print('There are nan in the data')
    img = nib.MGHImage(data, np.eye(4))
    nib.save(img, file)
    return file