import argparse
import logging
import os
import sys

from PIL import Image

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("path", metavar="PATH", help="path to image")
    parser.add_argument("--dump", "-d", dest="is_dump",
                        default=False, action="store_true", help="enable dumping")
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("--quiet", "-q", dest="is_verbose",
                           action="store_false", help="disable verbosity")
    verbosity.add_argument("--verbose", "-v", dest="is_verbose", default=True,
                           action="store_true", help="enable verbosity (default)")

    args = parser.parse_args()

    # Turn logging on / off
    if args.is_verbose:
        logging.basicConfig(format="%(message)s", level=logging.INFO)
    else:
        logging.basicConfig(format="%(message)s")

    CWD = os.getcwd()

    # Check the permission to write in CWD
    if not os.access(CWD, os.W_OK):
        logging.error("No permission to write in current directory!")
        sys.exit(1)

    # Check the path to the image
    try:
        IMG = Image.open(os.path.abspath(args.path))
    except IOError as err:
        logging.error("Cannot open image: {}".format(err))
        sys.exit(1)

    # Convert the image to grayscale and scale it - the script will run faster.
    IMG = IMG.convert("L")
    IMG.thumbnail((512, 512))
