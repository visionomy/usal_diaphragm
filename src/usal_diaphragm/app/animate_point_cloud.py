# -*- coding: utf-8 -*-
"""
Created on Sat Feb 05 16:25:47 2022

@author: ptresadern
"""
import os
import sys

from matplotlib import pyplot as plt

from usal_diaphragm.fit_surface import fit_surface
from usal_diaphragm.visualize import (
    image_grid, points_plot, params_plot, volume_slice,
    multi_plot,
    csv_writer,
)
from usal_diaphragm.app import _app_args
from usal_diaphragm.app import app_base


def main(app_args=sys.argv):
    """-"""
    the_app = ShowAnimApp(
        "Animate 3D reconstruction of diaphragm"
        " from a 4D ultrasound volume",
        app_args,
    )

    return the_app.main()


class ShowAnimApp(app_base.DiaphragmApp):

    def __init__(self, description, app_args):
        """-"""
        super().__init__(description, app_args)

        self._rect_fitter = None
        self._cart_fitter = fit_surface.create_fitter("sphere", self._args)

    # Nonexported methods 

    def _process_source(self, full_source):
        """-"""
        frames_dict = self._crunch_the_numbers(full_source)

        # Create the figure and set up the axes according to the options
        # specified by the user.
        fig = plt.figure()

        if self._args.output_csv or self._args.action == "output_csv":
            wtr = csv_writer.CsvWriter(frames_dict, full_source, self._args)
            wtr.save_csv_files()

        if self._args.action == "show_3d_points":
            # The animation must be assigned to a variable.
            # If not, it is garbage collected before the animation
            # is seen.

            _, filename = os.path.split(full_source)
            plots = []
            plots.append(
                points_plot.PointsPlot(
                    fig, filename, frames_dict, self._vol_props, self._args
                )
            )
            _, filename = os.path.split(full_source)
            multiplt = multi_plot.MultiPlot(fig, filename, plots)
            multiplt.animate()

        elif self._args.action == "show_2d_movie":
            _, filename = os.path.split(full_source)
            if self._args.cartesian:
                plots = []
                for yoff in self._args.slice_offsets:
                    self._args.yoffset = yoff
                    plots.append(
                        volume_slice.CartesianVolumeSlice(
                            fig, filename, frames_dict, self._vol_props, self._args
                        )
                    )
                multiplt = multi_plot.MultiPlot(fig, filename, plots)
                multiplt.animate()
            else:
                img_grid = image_grid.ImageGrid(
                    fig, filename, frames_dict, self._vol_props, self._args
                )
                img_grid.animate()

        print("Finished!")

    def _arg_parser(self):
        """Define all the arguments that the user can pass to the program"""
        parser = _app_args.new_parser(
            "Script to process a 4D volume and output graphical"
            " feedback."
        )

        parser.add_argument(
            "--action",
            help="What output to generate.",
            choices=(
                "none",
                "output_csv",
                "show_3d_points",
                "show_2d_movie",
            ),
            default="show_anim",
        )

        _app_args.add_input_args_to(parser)
        _app_args.add_trim_args_to(parser)
        _app_args.add_peaks_args_to(parser)
        _app_args.add_surface_args_to(parser)
        _app_args.add_plot_args_to(parser)

        return parser



if __name__ == "__main__":
    main()
