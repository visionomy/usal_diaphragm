"""
Class for displaying a 3D scatter plot of points
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as anim

from usal_diaphragm.visualize import view_base
from usal_diaphragm.geom import convert


class PointsPlot(view_base.ViewBase):

    def __init__(self, fig, filename, frames_dict, vol_props, args):
        """-"""
        super().__init__(fig, filename, frames_dict, vol_props, args)

        self._bounding_box = args.box
        self._cartesian = args.cartesian

        self._yoffset = float(args.slice_offsets[0]) / 1000.0  # mm -> m
        self._dy_thresh = args.dy_thresh / 1000.0  # mm -> m

        self.projection = "3d"

    def _create_axes(self):
        """Set up the axes"""
        return [
            self._fig.add_subplot(projection='3d')
        ]

    def _plot_frame(self, fi, ax=None):
        """-"""
        if ax is None:
            ax = self._axes_list[0]

        ax.cla()
        ax.set_title("Peaks and fitted surface")

        # Call the appropriate transform when calling f(...)
        if self._cartesian:
            f = convert.to_cartesian
        else:
            f = self._null_conv

        for key, val in self._frames_dict.items():
            if key in ("peaks", "surface"):
                rads, thetas, phis = val[fi]
                ii, jj, kk = f(rads, thetas, phis, self._vol_props)
                ax.scatter(ii, jj, kk, marker=".")
            elif key in ("surface_ref",):
                ii, jj, kk = val[:,0], val[:,1], val[:,2]
                ax.plot(ii, jj, kk, "g-")
            elif key in ("sphere_params",):
                cx, cy, cz, r = val[fi][:4]
                ax.scatter(cx, cy, cz, marker="^")

                npts = 100
                theta = np.linspace(0, 2*np.pi, npts)

                rprime = np.sqrt(r**2 - (self._yoffset - cy)**2)
                for off_mult in [-1, 1]:
                    ax.plot(
                        cx + rprime*np.sin(theta), 
                        (self._yoffset+off_mult*self._dy_thresh)*np.ones(npts),
                        cz + rprime*np.cos(theta), 
                        "-", color="grey",
                    )

        nk, nj, ni = self._frames_dict["raw"][0].shape

        if self._bounding_box:
            self._draw_bounding_box_on(ax, ni, nj, nk)

        self._set_ax_limits_to_bounding_box(ax, ni, nj, nk)
        ax.set_aspect("equal", adjustable="datalim")

        if self._cartesian:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        else:
            ax.set_xlabel("radius")
            ax.set_ylabel("theta")
            ax.set_zlabel("phi")

        return ii, jj, kk

    def _movie_subdir(self):
        """-"""
        return "points_plot_movie"

    def _draw_bounding_box_on(self, ax, ni, nj, nk):
        """-"""
        plt.sca(ax)

        if self._cartesian:
            f = convert.to_cartesian
        else:
            f = self._null_conv
        vp = self._vol_props

        # *f() unpacks the list of returned values so:
        #    plot(*f(...))
        # is the same as:
        #    a, b, c = f(...)
        #    plot(a, b, c)

        def _draw_the_box(imin, jmin, linestyle):
            imax = ni - 1
            jmax = nj - 1
            kmin = 0
            kmax = nk - 1

            i_line = range(imin, imax + 1)
            n = len(i_line)
            plt.plot(*f(i_line, jmin * np.ones(n), kmin * np.ones(n), vp), linestyle)
            plt.plot(*f(i_line, jmax * np.ones(n), kmin * np.ones(n), vp), linestyle)
            plt.plot(*f(i_line, jmin * np.ones(n), kmax * np.ones(n), vp), linestyle)
            plt.plot(*f(i_line, jmax * np.ones(n), kmax * np.ones(n), vp), linestyle)

            j_line = range(jmin, jmax + 1)
            n = len(j_line)
            plt.plot(*f(imin * np.ones(n), j_line, kmin * np.ones(n), vp), linestyle)
            plt.plot(*f(imax * np.ones(n), j_line, kmin * np.ones(n), vp), linestyle)
            plt.plot(*f(imin * np.ones(n), j_line, kmax * np.ones(n), vp), linestyle)
            plt.plot(*f(imax * np.ones(n), j_line, kmax * np.ones(n), vp), linestyle)

            k_line = range(kmin, kmax + 1)
            n = len(k_line)
            plt.plot(*f(imin * np.ones(n), jmin * np.ones(n), k_line, vp), linestyle)
            plt.plot(*f(imax * np.ones(n), jmin * np.ones(n), k_line, vp), linestyle)
            plt.plot(*f(imin * np.ones(n), jmax * np.ones(n), k_line, vp), linestyle)
            plt.plot(*f(imax * np.ones(n), jmax * np.ones(n), k_line, vp), linestyle)

        _draw_the_box(0, 0, "r:")
        _draw_the_box(self._ztrim, self._ytrim, "r-")

        # See above
        boxi, boxj, boxk = f([0], [nj // 2], [nk // 2], self._vol_props)
        plt.plot(boxi, boxj, boxk, "r*")

    def _set_ax_limits_to_bounding_box(self, ax, ni, nj, nk):
        """Set the axis limits to fit everything in nicely"""
        if self._cartesian:
            f = convert.to_cartesian
        else:
            f = self._null_conv

        ii, jj, kk = f(
            [0, ni, ni, ni, ni, ni],
            [nj / 2, nj / 2, nj / 2, nj / 2, 0, nj - 1],
            [nk / 2, nk / 2, 0, nk - 1, nk / 2, nk / 2],
            self._vol_props,
        )

        ax.set_xlim((min(ii), max(ii)))
        ax.set_ylim((min(jj), max(jj)))
        ax.set_zlim((min(kk), max(kk)))

    @staticmethod
    def _null_conv(ii, jj, kk, vol_props):
        """Null conversion function (does nothing)"""
        del vol_props

        return ii, jj, kk
