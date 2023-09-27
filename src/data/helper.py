from numpy import genfromtxt
import numpy as np
from binaryornot.check import is_binary


def get_coordinates_from_file(coordinates_path, delimiter=' '):
    """
        Returns a NumPy array of coordinates extracted from the file at `coordinates_path`.
        The file can be a binary file, or a text file.

        Args:
            coordinates_path (str): The file path to read coordinates from.
            delimiter (str, optional): The delimiter used to separate values in the coordinates file.
            Defaults to ' '.

        Returns:
            numpy.ndarray: An array of coordinates, with each row representing a single coordinate.
    """
    if is_binary(coordinates_path):
        coordinates = np.load(coordinates_path)
    else:
        coordinates = genfromtxt(coordinates_path, delimiter=delimiter)
    return coordinates
