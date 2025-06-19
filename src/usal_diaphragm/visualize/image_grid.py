"""
Class for displaying a grid of slices from an ultrasound volume
"""
import numpy as np
from matplotlib import pyplot as plt

from usal_diaphragm.visualize import view_base


class ImageGrid(view_base.ViewBase):

    def _create_axes(self):
        """Set up the axes"""
        # Create a grid of images at most 5 rows deep
        n = self._n_slices
        if n < 5:
            k = n
        else:
            k = 5

        return [
            self._fig.add_subplot(k, int((n - 1) / 5) + 1, i + 1)
            for i in range(n)
        ]

    def _plot_frame(self, fi, ax=None):
        """-"""
        vol = self._masked_volume(fi)

        n_phi, n_theta, n_rad, _ = vol.shape
        slice_indices = [
            int(phi_ind)
            for phi_ind in np.linspace(0, n_phi, self._n_slices + 2)
        ]
        slice_indices = slice_indices[1:-1]

        plt.figure(self._fig)
        for ax, phi_ind in zip(self._axes_list, slice_indices):
            plt.axes(ax)
            ax.cla()

            ax.axis("off")
            ax.axis("equal")
            ax.set_xlim([0, n_rad])
            ax.set_ylim([0, n_theta])

            plt.imshow(
                vol[phi_ind],
                aspect="auto",
                origin="lower",
            )
            ax.set_xlabel("radius")

            nk = len(vol)
            ax.set_ylabel(f"theta (slice {phi_ind}/{nk})")

            if self._show_peaks:
                ii, jj, kk = self._frames_dict["peaks"][fi]
                ii = ii[kk == phi_ind]
                jj = jj[kk == phi_ind]

                plt.scatter(ii, jj, s=1, c="m")

            if self._surface != "none":
                surf_ii, surf_jj, surf_kk = self._frames_dict["surface"][fi]
                this_slice = surf_kk == phi_ind
                plt.scatter(surf_ii[this_slice], surf_jj[this_slice], s=1, c="c")

        self._fig.suptitle(self._filename + "\n" + "Frame {:03d}".format(fi))

    def _movie_subdir(self):
        """-"""
        return "image_grid_movie"
