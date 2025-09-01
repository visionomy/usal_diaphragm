import os 

import numpy as np
from scipy import ndimage 

from usal_diaphragm.vol import read_4d_vol


def dummy_volumes_list(full_source):
    """-"""
    _, filename = os.path.split(full_source)

    if not filename == "dummy_data":
        return None 
    
    nt = 137  # matches real data
    volumes_list = [
        _dummy_volume(t)
        for t in range(nt)
    ]    
    
    vol_props = None

    return volumes_list, vol_props


def _dummy_volume(t):
    """-"""
    nk, nj, ni, nt = 43, 112, 752, 137  # matches real data

    vol = np.zeros((nk, nj, ni), dtype="uint8")

    kernel = np.array([0.25, 0.5, 0.25])
    for k in range(nk):
        for j in range(nj):
            mk = -2 # -2
            mj = -2 # 2.5
            i = int(600 + mk*k + mj*j + 100*np.sin(2*np.pi*t/nt))
            vol[k, j, i-1:i+2] += 200
            vol[k, nj-j-1, i-1:i+2] += 200

    for ax in range(3):
        vol = ndimage.convolve1d(vol, kernel, axis=ax)

    vol += np.random.randint(0, 30, size=vol.shape, dtype="uint8")

    return vol 


read_4d_vol.READ_FUNCS.append(dummy_volumes_list)