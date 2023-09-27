from itertools import islice
from csv import reader
import random
import numpy as np

ALADIP_FILE = '../../data/raw/aladip_implicit.xyz'
N_PROTS = 1420738
ROWS_PER_PROT = 10


class Protein():
    def __init__(self, coordinates: list[list[float]]):
        """
        Initialize Protein object with the given coordinates.

        Args:
            coordinates (list[list[float]]): A list of lists containing the x, y, and z coordinates of the atoms making up the proteins.
        """
        self._coordinates = np.array(coordinates)
        self._centroid = self._calculate_centroid()
        self.center_coordinates()
        self._g_value = self._calculate_g_value()

    @property
    def g_value(self):
        return self._g_value

    def _calculate_g_value(self):
        return np.trace(np.dot(np.transpose(self._coordinates), self._coordinates))

    @property
    def centroid(self):
        return self._centroid

    def _calculate_centroid(self):
        return np.mean(self._coordinates, axis=0)

    @property
    def coordinates(self):
        return self._coordinates

    def center_coordinates(self):
        self._coordinates = self._coordinates - self._centroid


def get_n_proteins(n, seed=1234):
    """
    Return a list of n Protein objects randomly selected from the ALADIP_FILE.

    Args:
        n (int): Number of proteins to select.
        seed (int): Seed to use for the random number generator.

    Returns:
        list: List of n Protein instances.
    """
    random.seed(seed)
    prot_idxs = random.sample(range(0, N_PROTS - 1), n)
    proteins = []
    with open(ALADIP_FILE, 'r') as csvfile:
        for prot_idx in prot_idxs:
            aladip_file = reader(csvfile, delimiter=' ')
            protein = []
            for prot_lines in islice(aladip_file, prot_idx, prot_idx + ROWS_PER_PROT):
                protein.append([float(x) for x in prot_lines])
            proteins.append(Protein(protein))
            csvfile.seek(0)
    return proteins
