"""
Class for displaying a 3D scatter plot of points
"""
import numpy as np

from usal_diaphragm.visualize import view_base


class RectParamsPlot(view_base.ViewBase):

    def __init__(self, fig, filename, frames_dict, vol_props, args):
        """-"""
        super().__init__(fig, filename, frames_dict, vol_props, args)

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
        ax.set_title("Plane params trajectory")

        ps, n_to_keep = self._frames_dict["params"]
        all_ps = np.concatenate(ps)
        ax.scatter(
            all_ps[:, 0], all_ps[:, 1], all_ps[:, 2],
            marker=".", color="orange", s=1,
        )
        
        i0 = 0
        i1 = n_to_keep
        ps_fi = np.vstack(ps[fi])
        ax.scatter(
            ps_fi[i0:i1, 0], ps_fi[i0:i1, 1], ps_fi[i0:i1, 2],
            marker="o", color="blue"
        )
        
        all_meanp = self._frames_dict["best_params"]
        param_limiter = all_meanp

        ax.set_xlim(param_limiter[:,0].min(), param_limiter[:,0].max())
        ax.set_ylim(param_limiter[:,1].min(), param_limiter[:,1].max())
        ax.set_zlim(param_limiter[:,2].min(), param_limiter[:,2].max())

        meanp = self._frames_dict["best_params"][:fi]
        ax.plot3D(
            meanp[:, 0], meanp[:, 1], meanp[:, 2],
            marker="o", color="green"
        )

        ax.set_xlabel("a")
        ax.set_ylabel("b")
        ax.set_zlabel("c")

    def _movie_subdir(self):
        """-"""
        return "rect_params_movie"


class SphereParamsPlot(view_base.ViewBase):

    def __init__(self, fig, filename, frames_dict, vol_props, args):
        """-"""
        super().__init__(fig, filename, frames_dict, vol_props, args)

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
        ax.set_title("Sphere centre trajectory")

        v = np.array(self._frames_dict["sphere_params"])
        c = v[:, :3]
        ax.plot3D(c[:, 0], c[:, 1], c[:, 2], 'g-')
        ax.scatter(c[fi, 0], c[fi, 1], c[fi, 2], 'yo')

        ax.set_xlabel("cx")
        ax.set_ylabel("cy")
        ax.set_zlabel("cz")

    def _movie_subdir(self):
        """-"""
        return "sphere_params_movie"
