'''
Created on 25 Feb 2022

@author: ptresadern
'''
import os

from usal_diaphragm.vol.vol_obj import VolObj

_NAME_TO_HEX = {
    "dim_i": "00C00100",
    "dim_j": "00C00200",
    "dim_k": "00C00300",
    "rad_res": "00C10100",
    "theta_angles": "00C30200",
    "offset1": "00C20100",
    "offset2": "00C20200",
    "phi_angles": "00C30100",
    "cartesian_spacing": "10002200",
    "frame_data": "00D00100",
    "sequence_data": "00D60100",
}
_HEX_TO_NAME = {h: n for n, h in _NAME_TO_HEX.items()}
_HEADER = "KRETZFILE 1.0   ".encode("utf-8")


class VolWriter(object):
    '''
    classdocs
    '''

    def __init__(self, filename, destdir=None):
        '''
        Constructor
        '''
        assert isinstance(filename, str)

        self._filename = filename
        self._destdir = destdir

    def write_all(self, obj, n_frames=None):
        """-"""
        assert isinstance(obj, VolObj)

        if n_frames is None:
            n_frames = obj.n_frames()

        for f in range(n_frames):
            self.write_one(obj, f)

    def write_one(self, obj, frame=None):
        """Write .vol file"""
        assert isinstance(obj, VolObj)

        destdir, basename = os.path.split(self._filename)
        if self._destdir is not None:
            destdir = self._destdir

        basename, ext = os.path.splitext(basename)
        if frame is not None:
            filename_i = os.path.join(
                destdir,
                "{:s}-{:03d}{:s}".format(basename, frame, ext)
            )

        with open(filename_i, "wb") as fid:
            fid.write(_HEADER)

            for k, v in obj.items():
                h = _NAME_TO_HEX.get(k)

                if k == "sequence_data":
                    pass
                # These two export the file in polar coordinates
                # which loads faster and might actually be better
                # for processing (the results can be transformed
                # to cartesian space at the end).
                elif k == "rad_res":
                    pass
                elif k == "theta_angles":
                    pass
                elif k == "frame_data":
                    fid.write(bytes.fromhex(h))
                    fid.write(obj.frame_size())
                    fid.write(obj.frame_data(frame))
                elif h is not None:
                    fid.write(bytes.fromhex(h))
                    fid.write(v[0])
                    fid.write(v[1])

        return
