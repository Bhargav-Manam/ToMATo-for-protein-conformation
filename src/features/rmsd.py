import numpy as np
import math
import multiprocessing as mp
from itertools import islice
from csv import reader
import random

# LEGACY FILE: IMPLEMENTATION OF RMSD USING GRADIENT DESCENT (instead of qcp)

ALADIP_FILE = '../../data/raw/aladip_implicit.xyz'
N_PROTS = 1420738
ROWS_PER_PROT = 10

def rmsd(protein1, protein2):
    """
    Apply RMSD formula

    Args:
        protein1, protein2
    """
    distances = []
    for atom1, atom2 in zip(protein1, protein2):
        distances.append(np.linalg.norm(np.array(atom1) - np.array(atom2)))
    return math.sqrt(np.sum(np.array(distances) ** 2) / len(distances))


def assert_is_rotation_matrix(r):
    """Enforce orthogonality and determinant 1 on rotation matrix R using SVD."""
    u, s, vh = np.linalg.svd(r)
    r_new = np.dot(u, vh)
    if np.linalg.det(r_new) < 0:
        s[-1] = -s[-1]
        r_new = np.dot(u, np.dot(np.diag(s), vh))
    return r_new


def paa(protein1: np.ndarray, protein2: np.ndarray):
    """
    Principal Axes Alignment: returns the rotational matrix that aligns the x, y, and z axis of protein1 and
    protein2

    Args: protein1 (numpy.ndarray): The coordinates of the first protein structure. protein2 (numpy.ndarray): The coordinates of the second protein structure.

    Returns: numpy.ndarray: The optimal rotation matrix that minimizes the distance between the two structures.
    """

    # Center the structures at the origin
    x_centered = protein1 - np.mean(protein1, axis=0)
    y_centered = protein2 - np.mean(protein2, axis=0)

    # Compute the covariance matrix and its eigenvectors
    cov = np.dot(y_centered.T, x_centered)
    u, _, v_t = np.linalg.svd(cov)

    # Compute the optimal rotation matrix
    s = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(v_t.T) < 0:
        s[2, 2] = -1
    r = np.dot(u, np.dot(s, v_t))

    return r


def get_n_proteins(n, seed=1234):
    """
    Return a list of lists with protein coordinates randomly selected from the ALADIP_FILE.

    Args:
        n (int): Number of proteins to select.
        seed (int): Seed to use for the random number generator.

    Returns:
        list: List of lists containing the x, y, z coordinates of each atom in the protein.
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
            proteins.append(np.array(protein))
            csvfile.seek(0)
    return proteins


def gradient_descent(x: list[float], y: list[float], r_init=None, alpha=0.01, threshold=1e-5):
    """Find rotation matrix R that minimizes RMSD between x and y using gradient descent."""
    if r_init is None:
        r_init = assert_is_rotation_matrix(np.linalg.qr(np.random.randn(3, 3))[0])
    elif r_init == 'paa':
        r_init = paa(x, y)

    n = len(x)
    r = r_init
    y_rot = np.dot(y, r)
    rmsd_prev = rmsd(x, y_rot)
    while True:
        # Compute gradient of RMSD with respect to R
        diff = x - y_rot
        grad = -2 / n * np.dot(diff.T, np.dot(y, r))
        # Update R using gradient descent
        r -= alpha * grad
        r = assert_is_rotation_matrix(r)
        # Compute new RMSD and check for convergence
        y_rot = np.dot(y, r)
        rmsd_curr = rmsd(x, y_rot)
        if np.abs(rmsd_curr - rmsd_prev) < threshold:
            break
        else:
            rmsd_prev = rmsd_curr
    return r


def get_optimized_rmsd(protein1: list[float], protein2: list[float], r_init=None, alpha=0.001, threshold=1e-5):
    """First align the proteins by minimizing the RMSD over the rotation matrix, and then return the optimized RMSD"""
    init_rmsd = rmsd(protein1, protein2)
    if init_rmsd < threshold:
        return init_rmsd
    optim_r = gradient_descent(protein1, protein2, r_init, alpha, threshold)
    protein2 = np.dot(protein2, optim_r)
    return rmsd(protein1, protein2)


## old implementation without parallelization ##
# def get_rmsd_matrix(protein_list, r_init=None, alpha=0.001, threshold=1e-5):
#     n_prots = len(protein_list)
#     rmsd_matrix = np.zeros((n_prots, n_prots))
#     for i in range(n_prots):
#         for j in range(i + 1, n_prots):
#             rmsd_matrix[i][j] = get_optimized_rmsd(protein_list[i], protein_list[j], r_init, alpha, threshold)
#     rmsd_matrix += rmsd_matrix.T
#     return rmsd_matrix

def parallelize_rmsd_helper(i: int, j: int, protein_list: list[list[float]], r_init=None, alpha=0.001, threshold=1e-5):
    return i, j, get_optimized_rmsd(protein_list[i], protein_list[j], r_init, alpha, threshold)


def get_rmsd_matrix(protein_list: list[list[float]], r_init=None, alpha=0.001, threshold=1e-5):
    n_prots = len(protein_list)
    rmsd_matrix = np.zeros((n_prots, n_prots))
    with mp.Pool() as pool:
        results = [pool.apply_async(parallelize_rmsd_helper,
                                    args=(i, j, protein_list, r_init, alpha, threshold)) for i in
                   range(n_prots) for j in
                   range(i + 1, n_prots)]
        for r in results:
            i, j, rmsd_value = r.get()
            rmsd_matrix[i][j] = rmsd_value
            rmsd_matrix[j][i] = rmsd_value
    return rmsd_matrix
