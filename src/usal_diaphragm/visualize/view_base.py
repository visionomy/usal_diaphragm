"""
Class for displaying a grid of slices from an ultrasound volume
"""
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import animation as anim


class ViewBase:

    def __init__(self, fig, filename, frames_dict, vol_props, args):
        """-"""
        if fig is None:
            fig = plt.figure()
            
        self._fig = fig
        self._filename = filename
        self._frames_dict = frames_dict
        self._n_frames = len(frames_dict["raw"])

        self._n_slices = args.n_slices
        self._ztrim = args.ztrim
        self._ytrim = args.ytrim
        self._show_mask = args.show_mask
        self._show_peaks = args.show_peaks
        self._frame_rate = args.rate
        self._vol_props = vol_props

        self._surface = args.surface

        self._axes_list = None 
        self.projection = None

    def _create_axes(self):
        """Set up the axes"""
        raise NotImplementedError

    def _plot_frame(self, fi, ax=None):
        """-"""
        raise NotImplementedError

    def _masked_volume(self, fi):
        """-"""
        vol = np.array([self._frames_dict["raw"][fi]]*3, dtype=np.float64)

        if self._show_mask:
            # set channels 1(G) and 2(B) to zero, leaving
            # only the red (channel 0) component.
            vol[1:,:,:,:self._ztrim] = 0
            vol[1:,:,:self._ytrim,:] = 0

        vol /= 255.0

        # out_dim = (n_phi, n_theta, n_rad, n_channels)
        return np.transpose(vol, axes=[1,2,3,0])
    
    def animate(self):
        """-"""
        self._axes_list = self._create_axes()
        
        ani = anim.FuncAnimation(
            self._fig, self._plot_frame,
            frames=self._n_frames,
            init_func=None,
            interval=1000.0 / (2*self._frame_rate),
        )
        plt.show()
        del ani
