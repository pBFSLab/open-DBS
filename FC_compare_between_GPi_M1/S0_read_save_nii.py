import numpy as np
import pandas as pd
import ants

def read_nii(file):
    '''
    file: str, the path to the nii file, the nii file should be a 3D or 4D file

    Return the data in the nii file, if the nii file is a 4D file, the data will be reshaped to (N1 \* N2\* N3, N4)
    '''
    img = ants.image_read(file)
    data = img.numpy()
    if len(data.shape) == 4:
        data = data.reshape(data.shape[0]*data.shape[1]*data.shape[2], data.shape[3], order='F')
    elif len(data.shape) == 3:
        data = data
    else:
        raise ValueError('The nii file should be a 3D or 4D file')
    return data
