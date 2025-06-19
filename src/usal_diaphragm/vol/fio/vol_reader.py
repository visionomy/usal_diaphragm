'''
Created on 25 Feb 2022

@author: ptresadern
'''
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


class VolReader(object):
    '''
    classdocs
    '''

    def __init__(self, filename):
        '''
        Constructor
        '''
        self._filename = filename

    def read(self):
        """Read .vol file"""
        datadict = dict.fromkeys(_NAME_TO_HEX.keys(), None)

        group_element = "hello"
        with open(self._filename, "rb") as fid:
            header = fid.read(16)
            assert header == _HEADER

            while group_element != b'':
                group_element = fid.read(4)
                itemsize = fid.read(4)
                itemsize_as_int = int.from_bytes(itemsize, "little")
                itemdata = fid.read(itemsize_as_int)

                hexstr = bytes.hex(group_element).upper()
                namestr = _HEX_TO_NAME.get(hexstr)
                if namestr in datadict:
                    datadict[namestr] = (itemsize, itemdata)

        return VolObj(datadict)
