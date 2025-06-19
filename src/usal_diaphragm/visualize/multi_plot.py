"""
Class for displaying a 3D scatter plot of points
"""
import os 

from matplotlib import pyplot as plt
from matplotlib import animation as anim


class MultiPlot:

    def __init__(self, fig, filename, ax_plotters_list):
        """-"""
        self._fig = fig
        self._ax_plotters_list = ax_plotters_list
        self._filename = filename

        self._n_frames = ax_plotters_list[0]._n_frames
        self._frame_rate = ax_plotters_list[0]._frame_rate
        self._axes_list = None

    def _create_axes(self):
        """Set up the axes"""
        n = len(self._ax_plotters_list)

        axes_list = []
        for i, axp in enumerate(self._ax_plotters_list):
            axes_list.append(
                self._fig.add_subplot(1, n, i+1, projection=axp.projection)
            )

        return axes_list

    def _plot_frame(self, fi):
        """-"""
        for i, axp in enumerate(self._ax_plotters_list):
            axp._plot_frame(fi, self._axes_list[i])

        _, filename = os.path.split(self._filename)
        self._fig.suptitle(filename + ": " + "Frame {:03d}".format(fi))

    def _movie_subdir(self):
        """-"""
        return "multiplot"

    def animate(self):
        """-"""
        self._axes_list = self._create_axes()
        ani = anim.FuncAnimation(
            self._fig, self._plot_frame,
            frames=self._n_frames,
            init_func=None,
            interval=1000.0 / self._frame_rate,
        )
        plt.show()
        del ani
