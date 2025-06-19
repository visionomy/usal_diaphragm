"""
Module to facilitate reading a 4D volume either from a single file or from a folder.
"""
import os

import numpy as np 

from usal_diaphragm.vol.fio import vol_reader


def read_4d_vol_from(full_source):
    """-"""
    res = None 

    for read_func in READ_FUNCS:
        res = read_func(full_source)
        if res is not None:
            break
    
    if res is None:
        print(
            "Input " + full_source + " should be either a 4D .vol file"
            " or a folder of sequentially numbered 3D .vol files"
        )
        vol_arrays_list = None
    else:
        vol_arrays_list, vol_props = res

    if vol_arrays_list is None:
        print("Cannot load file " + full_source)
        vol_props = None
      
    return vol_arrays_list, vol_props
    

def _frames_from_3d_vols(full_source):
    """Get the sequence data from a list of 3D volumes"""
    if not os.path.isdir(full_source):
        return None

    vol_arrays_list = []

    # This is used for volumes that have been split already.
    vol_files = os.listdir(full_source)
    frame_files = [
        f for f in vol_files
        if "-" in f and f.endswith(".vol")
    ]

    vol_props = {}
    n = len(frame_files)
    for filename in frame_files[:n]:
        full_filename = os.path.join(full_source, filename)
        rdr = vol_reader.VolReader(full_filename)
        vol_3d = rdr.read()

        v = bytearray(vol_3d.frame_data())
        nk, nj, ni = vol_3d.frame_dimensions()
        v = np.array(v).reshape((nk, nj, ni))

        vol_arrays_list.append(v)

        vol_props = vol_3d.volume_properties()

    return vol_arrays_list, vol_props


def _frames_from_4d_vol(full_source):
    """Get the sequence data from a single 4D volume"""
    if not full_source.endswith(".vol"):
        return None

    OneHundredMb = 100 * 1024**2
    if os.path.getsize(full_source) > OneHundredMb:
        vol_arrays_list = []

        rdr = vol_reader.VolReader(full_source)
        vol_4d = rdr.read()

        vol_props = vol_4d.volume_properties()
        nk, nj, ni = vol_4d.frame_dimensions()
        for i in range(vol_4d.n_frames()):
            v = bytearray(vol_4d.frame_data(i))
            v = np.array(v).reshape((nk, nj, ni))
            vol_arrays_list.append(v)

        print("{:d} frames found".format(len(vol_arrays_list)))
    else:
        print("Ignoring " + full_source + " because FileSize < 100Mb")
        vol_arrays_list = vol_props = None

    return vol_arrays_list, vol_props


READ_FUNCS = [
    _frames_from_3d_vols,
    _frames_from_4d_vol,
]
