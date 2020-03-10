"""Image reproduction using a genetic algorithm."""
import argparse
import logging
import os
import sys
import random
import glob

from PIL import Image
import numpy as np
from scipy.spatial import distance


def get_fitness(target: np.ndarray, img_vector: np.ndarray) -> float:
    """Calculate fitness of specimen.
    Args:
        target: Original image converted to NumPy Array
        img_vector: Specimen NumPy Array
    Returns:
        Fitness of specimen
    """
    return distance.euclidean(target, img_vector)


def mutate(img_array: np.ndarray, probability: float = 0.4) -> np.ndarray:
    """Mutate specimen with given probability.
    Args:
        img_array: Specimen NumPy Array
        probability: Probability of mutation
    Returns:
        Mutated specimen
    """
    if probability >= random.random():

        # Get P0 point of rectangle
        x_0 = random.randint(0, img_array.shape[1])
        y_0 = random.randint(0, img_array.shape[0])

        # Get width and height of rectangle
        width = int(random.randint(0, img_array.shape[1] - x_0) / 4)
        height = int(random.randint(0, img_array.shape[0] - y_0) / 4)

        # Get color of rectangle
        if len(img_array.shape) == 2:
            # Color of grayscale rectangle
            color = random.randint(0, 255)
        else:
            if img_array.shape[2] == 3:
                # Color of RGB rectangle
                color = [random.randint(0, 255) for _ in range(3)]
            else:
                # Color of RGBA rectangle
                color = [random.randint(0, 255) for _ in range(3)]
                color.append(random.randint(0, 101))

        # "Draw" rectangle on NumPy Array
        img_array[y_0:y_0 + height + 1, x_0: x_0 + width + 1] = color

        # Call mutate function again
        mutate(img_array, probability / 2)

    return img_array


def create_file_name(num: int, length: int) -> str:
    """Create file name.
    Args:
        num: number of iteration
        length: max string length
    Returns:
        File name string
    """
    return "dump-" + "0" * (len(str(length)) - len(str(num))) + str(num) + ".jpg"


if __name__ == "__main__":

    MAX_GENERATIONS = 1000000
    POPULATION_SIZE = 10
    CROSSOVER_POPULATION = 2
    MUTATION_PROBABILITY = 0.4
    RESULT_SIZE = (300, 300)
    DUMP_EVERY_N = 1000

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
    DUMPS = CWD + "/dumps/"

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
    IMG.thumbnail(RESULT_SIZE)

    # Convert image to NumPy Array
    IMG_DATA = np.array(IMG)
    # Convert matrix to vector
    IMG_VECTOR = IMG_DATA.flatten()

    # Populate generation 0
    bests = [np.zeros(shape=IMG_DATA.shape, dtype=IMG_DATA.dtype)
             for _ in range(CROSSOVER_POPULATION)]

    logging.info("Generation:\t{}".format(0))
    logging.info("Fitness:\t{}".format(
        get_fitness(IMG_VECTOR, bests[0].flatten())))

    # Dump best specimen
    if args.is_dump:
        # Create dumps directory if not exists
        if not os.path.exists(DUMPS):
            os.makedirs(DUMPS)
        # Clean dumps directory
        else:
            filelist = glob.glob(os.path.join(DUMPS, "*.jpg"))
            for f in filelist:
                os.remove(f)

        Image.fromarray(bests[0], mode=IMG.mode).save(
            DUMPS + create_file_name(0, MAX_GENERATIONS), "JPEG", quality=100)

    for i in range(1, MAX_GENERATIONS + 1):
        logging.info("Generation:\t{}".format(i))

        # Populate generation
        generation = [mutate(np.copy(random.choice(bests)),
                             MUTATION_PROBABILITY) for _ in range(POPULATION_SIZE)]

        # Extend generation with bests specimens from previous generation
        # It prevents from decreasing of fitness
        generation.extend(bests)

        # Sort from the best to the worst
        generation.sort(key=lambda x: get_fitness(IMG_VECTOR, x.flatten()))

        # Get bests specimens to crossover
        bests = generation[:CROSSOVER_POPULATION]
        logging.info("Fitness:\t{}".format(
            get_fitness(IMG_VECTOR, bests[0].flatten())))

        # Dump every N iterations
        if args.is_dump:
            if i % DUMP_EVERY_N == 0:
                Image.fromarray(bests[0], mode=IMG.mode).save(
                    DUMPS + create_file_name(i, MAX_GENERATIONS), "JPEG", quality=100)

    # Dump best specimen
    if args.is_dump:
        Image.fromarray(bests[0], mode=IMG.mode).save(
            DUMPS + create_file_name(MAX_GENERATIONS, MAX_GENERATIONS), "JPEG", quality=100)

    # Save best specimen
    Image.fromarray(bests[0], mode=IMG.mode).save(
        "result.jpg", "JPEG", quality=100)
