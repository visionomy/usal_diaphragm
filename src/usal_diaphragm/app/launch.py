import sys 

from usal_diaphragm.app import (
    animate_point_cloud
)

_APP_DICT = {
    "animate_point_cloud": animate_point_cloud,
}


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    args = ["animate_point_cloud"] + args
    app = _APP_DICT.get(args[0])
    app.main(args)


if __name__ == "__main__":
    main()
