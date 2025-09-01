import os 
import yaml
import mock
import numpy as np 
import unittest 
import itertools 

from usal_diaphragm.vol import read_4d_vol
from usal_diaphragm.find_peaks import peak_finders


class PeakFindersTest(unittest.TestCase):

    def test_gradprime(self):
        filter = peak_finders.create("gradprime2d", self._finder_args()) 

    def test_peak_finders(self):
        with open("tests/config.yaml", "r") as fid:
            config = yaml.load(fid, yaml.Loader)

        data_root = config.get("test_data_folder", ".")

        vol_files = [
            f
            for f in os.listdir(data_root)
            if f.endswith(".vol")
        ]
        args = self._finder_args()

        for vol_file in vol_files:
            print(vol_file)

            volumes_list, _ = read_4d_vol.read_4d_vol_from(
                os.path.join(data_root, vol_file)
            )

            combs = itertools.product(
                peak_finders.peak_finder_names(),
                (False,),
                (0, 1, 2),
            )

            for comb in combs:
                kwargs = dict(zip(["name", "filter", "peaks_axis"], comb))
                finder_name, args.filter, args.peak_axis = comb
                with self.subTest(**kwargs):
                    try:
                        target = _TARGETS[vol_file][finder_name][args.filter][args.peaks_axis]
                    except KeyError:
                        pass 
                    else:
                        finder = peak_finders.create(finder_name, args)
                        ii, jj, kk = finder.find_peaks_in(volumes_list[0])
                        res = np.concatenate((ii, jj, kk))
                        
                        self.assertEqual((len(res), res.sum()), target)                           
                        print(comb)

        return

    @staticmethod
    def _finder_args():
        """-"""
        args = mock.Mock(["ztrim", "ytrim", "filter", "peaks_axis"])
        args.ztrim = 200
        args.ytrim = 0
        args.filter = False
        args.peaks_axis = 0
        args.intensity_threshold = 8
        args.grad_threshold = 20

        return args 


_TARGETS = {
    "Pract03_1.vol": {
        "max": {
            False: {
                0: (507, 92788),
                1: (71208, 12851113),
                2: (185316, 34268751),
            },
            True: {
                0: (10173, 1551523),
                1: (71208, 12851113),
                2: (185316, 34268751),
            },
        },
        "grad": {
            False: {
                0: (390, 55662.0),
                1: (35208, 6113870.0),
                2: (96534, 15917864.0),
            },
            True: {
                0: (14448, 2247318),
                1: (71208, 12851113),
                2: (185316, 34268751),
            },
        },
        "dog": {
            False: {
                0: (0, 0.0),
                1: (54558, 10049188.0),
                2: (76041, 12478397.0),
            },
            True: {
                0: (14448, 2247318),
                1: (71208, 12851113),
                2: (185316, 34268751),
            },
        },
    }
}


if __name__ == "__main__":
    unittest.main()
