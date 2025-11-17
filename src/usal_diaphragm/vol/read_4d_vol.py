"""
Module to facilitate reading a 4D ultrasound volume from either a single file or a folder.
"""
import os
import numpy as np

from usal_diaphragm.vol.fio import vol_reader

# Minimum file size threshold for 4D volumes (in bytes)
MIN_4D_VOLUME_SIZE_MB = 100
MIN_4D_VOLUME_SIZE_BYTES = MIN_4D_VOLUME_SIZE_MB * 1024 ** 2


def read_4d_vol_from(full_source):
    """
    Read a 4D ultrasound volume from a file or directory.

    Args:
        full_source: Path to either:
                    - A single 4D .vol file (must be >100MB)
                    - A directory containing sequentially numbered 3D .vol files

    Returns:
        Tuple of (vol_arrays_list, vol_props) where:
            - vol_arrays_list: List of 3D numpy arrays, one per frame
            - vol_props: Dictionary of volume properties
        Returns (None, None) if the source cannot be loaded
    """
    vol_arrays_list = None
    vol_props = None

    # Try each reader in sequence
    for read_func in READ_FUNCS:
        result = read_func(full_source)
        if result is not None:
            vol_arrays_list, vol_props = result
            break

    # Handle unsuccessful reads
    if vol_arrays_list is None:
        print(
            f"Input {full_source} should be either a 4D .vol file"
            " or a folder of sequentially numbered 3D .vol files"
        )
        return None, None

    if vol_props is None:
        print(f"Cannot load file {full_source}")
        return None, None

    return vol_arrays_list, vol_props


def _frames_from_3d_vols(full_source):
    """
    Get the sequence data from a directory of 3D volumes.

    Args:
        full_source: Path to directory containing .vol files

    Returns:
        Tuple of (vol_arrays_list, vol_props) or None if not a valid directory
    """
    if not os.path.isdir(full_source):
        return None

    vol_arrays_list = []

    # Find all .vol files with hyphen in name (indicating frame number)
    vol_files = os.listdir(full_source)
    frame_files = [
        f for f in vol_files
        if "-" in f and f.endswith(".vol")
    ]

    if not frame_files:
        return None

    vol_props = {}
    n_frames = len(frame_files)

    for filename in frame_files[:n_frames]:
        full_filename = os.path.join(full_source, filename)
        reader = vol_reader.VolReader(full_filename)
        vol_3d = reader.read()

        # Extract frame data and reshape
        frame_data_bytes = bytearray(vol_3d.frame_data())
        nk, nj, ni = vol_3d.frame_dimensions()
        frame_array = np.array(frame_data_bytes).reshape((nk, nj, ni))

        vol_arrays_list.append(frame_array)
        vol_props = vol_3d.volume_properties()

    return vol_arrays_list, vol_props


def _frames_from_4d_vol(full_source):
    """
    Get the sequence data from a single 4D volume file.

    Args:
        full_source: Path to a .vol file

    Returns:
        Tuple of (vol_arrays_list, vol_props) or None if file doesn't meet requirements
    """
    if not full_source.endswith(".vol"):
        return None

    # Check file size threshold
    file_size = os.path.getsize(full_source)
    if file_size <= MIN_4D_VOLUME_SIZE_BYTES:
        print(f"Ignoring {full_source} because FileSize < {MIN_4D_VOLUME_SIZE_MB}MB")
        return None

    # Read the volume
    reader = vol_reader.VolReader(full_source)
    vol_4d = reader.read()

    vol_props = vol_4d.volume_properties()
    nk, nj, ni = vol_4d.frame_dimensions()
    n_frames = vol_4d.n_frames()

    vol_arrays_list = []
    for frame_idx in range(n_frames):
        frame_data_bytes = bytearray(vol_4d.frame_data(frame_idx))
        frame_array = np.array(frame_data_bytes).reshape((nk, nj, ni))
        vol_arrays_list.append(frame_array)

    print(f"{len(vol_arrays_list)} frames found")

    return vol_arrays_list, vol_props


# List of reader functions to try in order
READ_FUNCS = [
    _frames_from_3d_vols,
    _frames_from_4d_vol,
]
