import numpy as np
from scipy import signal

from usal_diaphragm.find_peaks import filter_point_cloud


def peak_finder_names():
    """-"""
    return list(_PEAK_FINDERS_REGISTRY.keys())


def create(pf_type, args):
    """-"""
    cls = _PEAK_FINDERS_REGISTRY.get(pf_type)
    if cls:
        inst = cls(args)
    else:
        inst = None

    return inst


class PeakFinderBase:

    """Base class for finding interesting points in a volume"""

    def __init__(self, args):
        """-"""
        self._ztrim = args.ztrim
        self._ytrim = args.ytrim
        self._use_filter = args.filter
        self._n_transpose = args.peaks_axis

    def find_peaks_in(self, volume):
        for _ in range(self._n_transpose):
            volume = np.transpose(volume, axes=[2, 0, 1])
        
        ii, jj, kk = self._find_peaks_in(volume)

        for _ in range(self._n_transpose):
            ii, jj, kk = kk, ii, jj

        if self._use_filter:
            ii, jj, kk = filter_point_cloud.filter_isolated_points_from(
                ii, jj, kk,
                threshold=3.0
            )

        keep = (
            np.array(ii > self._ztrim) &
            np.array(jj > self._ytrim)
        )

        return ii[keep], jj[keep], kk[keep]

    def _find_peaks_in(self, volume):
        """-"""
        raise NotImplementedError


# Implementation

class _MaxPeakFinder(PeakFinderBase):

    """Peak finder that looks for the maximum along the i axis in the volume"""

    def __init__(self, args):
        """-"""
        super().__init__(args)

        self._threshold = args.intensity_threshold

    def _find_peaks_in(self, volume):
        """-"""
        nk, nj = volume.shape[:2]

        # ii is a 2D array...
        ii = np.argmax(volume, axis=2)
        # .. as are jj and kk.
        jj, kk = np.meshgrid(range(nj), range(nk))

        if self._threshold > 0:
            # If the user specifies a threshold between 0 and 1, it
            # is treated as a fraction of the maximum value.
            # Otherwise the threshold is treated as an absolute value.
            if 0 < self._threshold < 1.0:
                threshold = self._threshold * volume.max()
            else:
                threshold = self._threshold

            # Filter out points that are below the threshold.
            vv = np.max(volume, axis=2)
            good_peaks = vv > threshold
            ii = ii[good_peaks]
            jj = jj[good_peaks]
            kk = kk[good_peaks]
        else:
            ii = ii.flatten()
            jj = jj.flatten()
            kk = kk.flatten()

        # Return the arrays as a list for each dimension.
        return ii, jj, kk


class _1DFilteredPeakFinder(PeakFinderBase):

    """Peak finder that looks for high gradients along the i axis 
    in the volume
    """

    def __init__(self, args):
        """-"""
        super().__init__(args)

        self._filter = None
        self._threshold = None

    def _find_peaks_in(self, volume):
        """-"""
        nk, nj = volume.shape[:2]

        # Convolve the volume with an edge filter along the i axis
        fltr = self._filter
        fltr /= abs(fltr).sum()

        ii = np.zeros((nk, nj))
        good_peaks = np.zeros((nk, nj))
        for j in range(nj):
            for k in range(nk):
                z = np.convolve(
                    np.float64(volume[k, j, :]), fltr, 
                    mode="same"
                )

                # Find the point furthest from the probe that has a
                # gradient bigger than the specified threshold
                # and flag that one has been found in good_peaks.
                for i in reversed(range(len(z))):
                    if 0 < i < len(z)-1 and z[i] > self._threshold:
                        ii[k, j] = i
                        good_peaks[k, j] = 1.0
                        break

        jj, kk = np.meshgrid(range(nj), range(nk))

        inds_to_keep = good_peaks > 0
        ii = ii[inds_to_keep]
        jj = jj[inds_to_keep]
        kk = kk[inds_to_keep]

        return ii, jj, kk


class _GradientPeakFinder(_1DFilteredPeakFinder):

    """Peak finder that looks for high gradients along the i axis 
    in the volume
    """

    def __init__(self, args):
        """-"""
        super().__init__(args)

        self._filter = np.array([-1.0, 0, 1.0])
        self._threshold = args.grad_threshold


class _DoGPeakFinder(_1DFilteredPeakFinder):

    """Peak finder that looks for high responses to a 
    difference of gaussians filter along the i axis
    in the volume
    """

    def __init__(self, args):
        """-"""
        super().__init__(args)

        self._filter = np.array([-1.0, 2.0, -1.0])
        self._threshold = args.grad_threshold


class _2DFilteredPeakFinder(PeakFinderBase):

    """Peak finder that looks for high gradients along the i axis 
    in the volume
    """

    def __init__(self, args):
        """-"""
        super().__init__(args)

        self._filter = None
        self._threshold = None

    def _find_peaks_in(self, volume):
        """-"""
        nk, nj, ni = volume.shape[:3]

        # Convolve the volume with an edge filter along the i axis
        fltr = self._filter[::-1,::-1]
        fltr /= abs(fltr).sum()

        kk, jj, ii = np.meshgrid(range(nk), range(nj), range(ni), indexing="ij")
        good_peaks = np.zeros((nk, nj, ni), dtype=np.bool_)

        for k in range(nk):
            z = signal.convolve2d(
                np.float64(volume[k, :, :].squeeze()), fltr, 
                mode="same"
            )

            good_peaks[k, 1:-2, 1:-2] = z[1:-2, 1:-2] > self._threshold

        ii = ii[good_peaks]
        jj = jj[good_peaks]
        kk = kk[good_peaks]

        return ii, jj, kk


class _Wrong2DGradientPeakFinder(_2DFilteredPeakFinder):

    def __init__(self, args):
        """-"""
        super().__init__(args)

        sz = 3
        self._filter = np.zeros((sz, sz))
        for i in range(sz):
            for j in range(sz):
                self._filter[i,j] = j-i

        self._threshold = args.grad_threshold


class _2DGradientPeakFinder(_Wrong2DGradientPeakFinder):

    def __init__(self, args):
        """-"""
        super().__init__(args)
        self._filter = self._filter[::-1,:]


class _2DGradGradientPeakFinder(_2DFilteredPeakFinder):

    def __init__(self, args):
        """-"""
        super().__init__(args)

        half_sz = 3
        sz = 1+2*half_sz

        theta = np.pi/4
        sig = half_sz/3
        self._filter = np.zeros((sz, sz))
        for fi, i in enumerate(range(-half_sz, half_sz+1)):
            for fj, j in enumerate(range(-half_sz, half_sz+1)):
                i0 = np.cos(theta)*i - np.sin(theta)*j
                j0 = np.sin(theta)*i + np.cos(theta)*j

                gi = np.exp(-0.5*(i0/sig)**2)
                dgj = -(j0/sig)*np.exp(-0.5*(j0/sig)**2)

                self._filter[fi,fj] = gi*dgj
        
        self._threshold = args.grad_threshold


class _2DDoGPeakFinder(_2DFilteredPeakFinder):

    """Peak finder that looks for high responses to a 
    difference of gaussians filter along the i axis
    in the volume
    """

    def __init__(self, args):
        """-"""
        super().__init__(args)

        self._filter = np.array([
            [ 0.0, -1.0,  0.0],
            [-1.0,  4.0, -1.0],
            [ 0.0, -1.0,  0.0],
        ])
        self._threshold = args.grad_threshold


_PEAK_FINDERS_REGISTRY = {
    "grad2d-wrong": _Wrong2DGradientPeakFinder,
    "grad2d": _2DGradientPeakFinder,
    "gradprime2d": _2DGradGradientPeakFinder,
    "max": _MaxPeakFinder,
    "grad": _GradientPeakFinder,
    "dog": _DoGPeakFinder,
    "dog2d": _2DDoGPeakFinder,
    "none": None,
}
