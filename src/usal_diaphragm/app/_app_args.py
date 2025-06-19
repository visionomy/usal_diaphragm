import argparse

from usal_diaphragm.find_peaks import peak_finders


def new_parser(description_str):
    """-"""
    parser = argparse.ArgumentParser(
        description=description_str,
    )
    
    parser.add_argument(
        "--dev",
        help="Development mode",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--randseed",
        help="Seed for random number generator",
        type=int,
        default=-1,
    )

    return parser


def add_input_args_to(parser):
    """-"""
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument(
        "--config",
        help="Path to config file",
        default="",
    )

    parser.add_argument(
        "--input_root",
        help="Root folder of input files",
        default="",
    )
    parser.add_argument(
        "--input_vol",
        help="Input .vol file to process",
        type=str,
        default="",
    )
    parser.add_argument(
        "input",
        help="Input .vol or .yaml file to process",
    )


def add_trim_args_to(parser):
    """-"""
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument(
        "-z", "--ztrim",
        help="Number of pixels to trim from near edge of Z axis.",
        type=int,
        default=350,
    )
    parser.add_argument(
        "--tstart",
        help="First frame to use",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-y", "--ytrim",
        help="Number of pixels to trim from near edge of Y axis.",
        type=int,
        default=0,
    )
    parser.add_argument(
        "-n", "--n_frames_max",
        help="Maximum number of frames to visualise.",
        type=int,
        default=10000,
    )
    

def add_peaks_args_to(parser):
    """-"""
    assert isinstance(parser, argparse.ArgumentParser)

    peak_finder_types = peak_finders.peak_finder_names()
    def_peak_finder = peak_finder_types[0]
    parser.add_argument(
        "--peaks",
        help="Peak finder type (default = " + def_peak_finder + ")",
        choices=peak_finder_types,
        default=def_peak_finder,
    )
    parser.add_argument(
        "-p", "--show_peaks",
        help="Overlay peaks in ultrasound movies",
        action="store_true",
    )
    parser.add_argument(
        "--peaks_axis",
        help="Axis along which to search for peaks",
        type=int,
        choices=[0,1,2],
        default=0,
    )
    parser.add_argument(
        "-i", "--intensity_threshold",
        help="Threshold below which we discard peaks.",
        type=float,
        default=-1.0,
    )
    parser.add_argument(
        "--grad_threshold",
        help="Gradient threshold above which we accept a diaphragm point.",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "-f", "--filter",
        help="Filter peaks in the point cloud.",
        action="store_true",
    )


def add_surface_args_to(parser):
    """-"""
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument(
        "-s", "--surface",
        help="What surface to fit to the point cloud.",
        choices=(
            "none",
            "plane",
            "parabola",
        ),
        default="plane",
    )
    parser.add_argument(
        "--dist_threshold",
        help="Distance threshold for RanSaC surface fitting.",
        type=float,
        default=25.0,
    )
    parser.add_argument(
        "--n_surfaces_test",
        help="Number of surfaces to test per frame.",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--n_surfaces_keep",
        help="Number of tested surfaces to consider per frame.",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--dist_weight",
        help="Weight applied to distance between consecutive parameters",
        type=float,
        default=1000.0,
    )
    parser.add_argument(
        "--sphere",
        help="Fit sphere to cartesian points",
        action="store_true",
    )
    parser.add_argument(
        "--replace_outliers",
        help="Attempt to replace outliers",
        action="store_true",
    )


def add_plot_args_to(parser):
    """-"""
    assert isinstance(parser, argparse.ArgumentParser)

    parser.add_argument(
        "--movie_format",
        help="Movie format (default=mp4)",
        choices=(
            "mp4",
            "mpg",
            "wmv",
        ),
        default="mp4",
    )
    parser.add_argument(
        "-r", "--rate",
        help="Frame rate of any resulting movie.",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--n_slices",
        help="Maximum number of US slices to visualise.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--slice_offsets",
        help="Offsets (in mm) of slices from centre.",
        type=list,
        default=[0.0],
    )
    parser.add_argument(
        "--dy_thresh",
        help="Max. distance of included peaks from current slice",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--show_mask",
        help="Show region of the volume that is masked.",
        action="store_true",
    )
    parser.add_argument(
        "-u", "--ultra",
        help="Show ultrasound alongside point cloud.",
        action="store_true",
    )
    parser.add_argument(
        "-b", "--box",
        help="Show bounding box of volume.",
        action="store_true",
    )
    parser.add_argument(
        "-c", "--cartesian",
        help="Convert spherical coordinates to cartesian ones.",
        action="store_true",
    )
    parser.add_argument(
        "--flip_x",
        help="Flip the image in X",
        action="store_true",
    )    
    parser.add_argument(
        "--flip_y",
        help="Flip the image in Y",
        action="store_true",
    )    
    parser.add_argument(
        "--output_csv",
        help="Save output to a .csv table",
        action="store_true",
    )    
    parser.add_argument(
        "--csv_suffix",
        help="Add a suffix to the .csv filename",
        action="store",
        default="",
    )    
    parser.add_argument(
        "--direction_points",
        help="Endpoints of the direction of movement in the first slice",
        type=list,
        default=[],
    )