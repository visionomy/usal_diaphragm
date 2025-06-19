'''
Created on 25 Feb 2022

@author: ptresadern
'''
import struct


class VolObj(object):

    '''
    Class that stores information about an ultrasound volume, 
    either 3D or 4D.
    '''

    def __init__(self, data_dict):
        '''
        Constructor
        '''
        self._datadict = data_dict

        self._frame_size, self._frame_data = self._datadict["frame_data"]
        seq_tuple = self._datadict.get("sequence_data")
        if seq_tuple is not None:
            self._seq_size, self._seq_data = seq_tuple
        else:
            self._seq_size = self._frame_size
            self._seq_data = self._frame_data

    def frame_data(self, ind=-1):
        """-"""
        _frame_data = None

        if ind + 1 > self.n_frames():
            msg = "Frame index out of bounds"
            raise IndexError(msg)
        elif ind > -1:
            frame_size_as_int = self._int_from(self.frame_size())
            i0 = ind * frame_size_as_int
            i1 = i0 + frame_size_as_int
            _frame_data = self._seq_data[i0:i1]
        else:
            _frame_data = self._frame_data

        return _frame_data

    def volume_properties(self):
        """-"""
        vol_props = {}

        vol_props.update({
            k: self._int_from(self._datadict[k][1])
            for k in [
                "dim_i", "dim_j", "dim_k",
            ]
        })
        vol_props.update({
            k: self._double_from(self._datadict[k][1])
            for k in [
                "offset1", "offset2",
                "rad_res",
                "cartesian_spacing",
            ]
        })
        vol_props.update({
            k: self._doubles_list_from(self._datadict[k][1])
            for k in [
                "theta_angles", "phi_angles",
            ]
        })

        return vol_props

    def frame_size(self):
        """-"""
        return self._frame_size

    def frame_dimensions(self):
        """Return the dimensions of the frame data as (nk,nj,ni)"""
        return (
            self._int_from(self._datadict["dim_k"][1]),
            self._int_from(self._datadict["dim_j"][1]),
            self._int_from(self._datadict["dim_i"][1]),
        )

    def sequence_size(self):
        """-"""
        return self._seq_size

    def n_frames(self):
        """-"""
        n_frames = (
            self._int_from(self.sequence_size()) /
            self._int_from(self.frame_size())
        )
        assert n_frames % 1 == 0

        return int(n_frames)

    def theta_angles(self):
        """-"""
        return self._datadict["theta_angles"]

    def phi_angles(self):
        """-"""
        return self._datadict["phi_angles"]

    def items(self):
        """-"""
        return self._datadict.items()

    def _doubles_list_from(self, b):
        """-"""
        n = len(b) / 8
        assert n % 1 < 1e-6

        n = int(n)

        vals = [
            self._double_from(b[i * 8:(i + 1) * 8])
            for i in range(n)
        ]

        return vals

    @staticmethod
    def _int_from(b, order="little"):
        """-"""
        return int.from_bytes(b, order)

    @staticmethod
    def _double_from(b):
        """-"""
        return struct.unpack("d", b)[0]
