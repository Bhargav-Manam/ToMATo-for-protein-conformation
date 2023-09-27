from parse_proteins import Protein
import numpy as np
from math import sqrt
import multiprocessing as mp


def calculate_m_matrix(protein1: np.ndarray, protein2: np.ndarray):
    """
    Calculates the M matrix for two given proteins.

    Args:
    - protein1, protein2: 2D Numpy arrays representing protein 1 and 2.

    Returns:
    - np.ndarray: A 2D numpy array representing the M matrix.

    Raises:
    - ValueError: If protein1 and protein2 don't have the same number of atoms.

    Example:
    >>> p1 = np.array([[1, 2], [3, 4]])
    >>> p2 = np.array([[5, 6], [7, 8]])
    >>> calculate_m_matrix(p1, p2)
    array([[19, 22],
           [43, 50]])

    Note: This function calculates the M matrix as defined in the article "Rapid calculation of RMSDs using a
    quaternion-based characteristic polynomial" by Theobald and Douglas L (2005).
    """
    return np.dot(np.transpose(protein2), protein1)


def calculate_coefficients(m_matrix: np.ndarray):
    """
        Calculates the coefficients c0, c1, and c2 of the qcp equation.

        Args:
        - m_matrix: A 3x3 numpy array representing the M matrix.

        Returns:
            A list of three floating-point numbers representing the coefficients of
            the quadratic equation.
        
        Note: This function calculates the coefficients as defined in the article "Rapid calculation of RMSDs using a
        quaternion-based characteristic polynomial" by Theobald and Douglas L (2005).

    """

    s_xx, s_xy, s_xz = m_matrix[0][0], m_matrix[0][1], m_matrix[0][2]
    s_yx, s_yy, s_yz = m_matrix[1][0], m_matrix[1][1], m_matrix[1][2]
    s_zx, s_zy, s_zz = m_matrix[2][0], m_matrix[2][1], m_matrix[2][2]
    # /**/ #
    s_xz2 = s_xz ** 2
    s_yz2 = s_yz ** 2
    s_zz2 = s_zz ** 2
    s_xy2 = s_xy ** 2
    s_yy2 = s_yy ** 2
    s_zy2 = s_zy ** 2
    s_xx2 = s_xx ** 2
    s_yx2 = s_yx ** 2
    s_zx2 = s_zx ** 2
    # /**/ #
    s_yyzz = s_yy * s_zz
    s_yzzy = s_yz * s_zy
    s_xzpzx = s_xz + s_zx
    s_yzpzy = s_yz + s_zy
    s_xypyx = s_xy + s_yx
    s_yypzz = s_yy + s_zz
    s_yzmzy = s_yz - s_zy
    s_xymyx = s_xy - s_yx
    s_xzmzx = s_xz - s_zx
    s_yymzz = s_yy - s_zz

    c2 = -2 * (s_xz2 + s_yz2 + s_zz2 + s_xy2 + s_yy2 + s_zy2 + s_xx2 + s_yx2 + s_zx2)
    c1 = 8 * (
            (s_xx * s_yzzy + s_yy * s_zx * s_xz + s_zz * s_xy * s_yx)
            - (s_xx * s_yyzz + s_yz * s_zx * s_xy + s_zy * s_yx * s_xz)
    )
    d = (s_xypyx * s_xymyx + s_xzpzx * s_xzmzx) ** 2
    e = ((-s_xx2 + s_yy2 + s_zz2 + s_yz2 + s_zy2) ** 2) - (4 * ((s_yyzz - s_yzzy) ** 2))
    f = (-(s_xzpzx * s_yzmzy) + (s_xymyx * (s_xx - s_yypzz))) * (-(s_xzmzx * s_yzpzy) + (s_xymyx * (s_xx - s_yymzz)))
    g = (-(s_xzpzx * s_yzpzy) - (s_xypyx * (s_xx + s_yymzz))) * (-(s_xzmzx * s_yzmzy) - (s_xypyx * (s_xx + s_yypzz)))
    h = ((s_xypyx * s_yzpzy) + (s_xzpzx * (s_xx - s_yymzz))) * (-(s_xymyx * s_yzmzy) + (s_xzpzx * (s_xx + s_yypzz)))
    i = ((s_xypyx * s_yzmzy) + (s_xzmzx * (s_xx - s_yypzz)) * (-(s_xymyx * s_yzpzy) + (s_xzmzx * (s_xx + s_yymzz))))
    c0 = d + e + f + g + h + i

    return [c0, c1, c2]


def p_lmd(lmd: float, c0: float, c1: float, c2: float):
    """
    Calculates P(lamda) as defined in the study by Theobald and Douglas L (2005).

    Args:
    - lmd: the lamda to evaluate the polynomial function at.
    - c0, c1, c2: coefficients 0,1,2 to the polynomial equation P.

    Returns:
    - float: the value of the polynomial function at lambda=lmd
    """
    return lmd ** 4 + c2 * lmd ** 2 + c1 * lmd + c0


def dp_lmd_dlmd(lmd: float, c1: float, c2: float):
    """
    Calculate the dP(lambda)/d(lamda) as described by Theobald and Douglas L (2005).

    Args:
    - lmd: The value of lamda to calculate the derivative at.
    -c1, c2: coefficients 1,2 to the polynomial equation P.

    Returns:
    - float: The value of the derivative of the function with respect to lmd.
    """
    return 4 * lmd ** 3 + 2 * c2 * lmd + c1


def update_lmd(lmd, c0, c1, c2):
    """
    Update the value of lambda (lmd) using the Newton-Raphson method.

    Returns:
        float: the updated value of lambda.
    """
    return lmd - (p_lmd(lmd, c0, c1, c2) / dp_lmd_dlmd(lmd, c1, c2))


def minimize_lmd_max(protein1: Protein, protein2: Protein, tol=10e-3, max_iter=5):
    """
    Minimizes the maximum distance between two proteins by calculating the M matrix, 
    calculating coefficients c0, c1, and c2, and updating lambda until convergence 
    or maximum iterations are reached. Returns the optimized lambda value.

    Args:
    - protein1 (Protein): first protein object
    - protein2 (Protein): second protein object
    - tol (float): tolerance for convergence
    - max_iter (int): maximum number of iterations

    Returns:
    - lmd (float): optimized lambda value
    """
    m_matrix = calculate_m_matrix(protein1.coordinates, protein2.coordinates)
    c0, c1, c2 = calculate_coefficients(m_matrix)
    lmd_init = (protein1.g_value + protein2.g_value) / 2
    lmd = lmd_init
    lmd_new = update_lmd(lmd, c0, c1, c2)
    i = 0
    while (abs(lmd_new - lmd) > tol) and (i < max_iter):
        i += 1
        lmd = lmd_new
        lmd_new = update_lmd(lmd, c0, c1, c2)
    lmd = lmd_init if lmd > lmd_init else (0 if lmd < 0 else lmd)
    return lmd


def get_rmsd(protein1: Protein, protein2: Protein, tol=10e-3, max_iter=5):
    """
    Calculates the root-mean-square deviation (RMSD) between two protein structures.

    Args:
    - protein1 (Protein): The first protein structure.
    - protein2 (Protein): The second protein structure.
    - tol (float, optional): The tolerance for the RMSD calculation. Defaults to 10e-3.
    - max_iter (int, optional): The maximum number of iterations for the RMSD calculation. Defaults to 5.

    Returns:
    - float: The RMSD between the two protein structures.

    Raises:
    - ValueError: If the proteins have a different number of atoms
    """
    n = len(protein1.coordinates)
    if n != len(protein2.coordinates):
        raise ValueError('proteins 1 and 2 should have the same length')
    lmd_max = minimize_lmd_max(protein1, protein2, tol, max_iter)
    return sqrt((protein1.g_value + protein2.g_value - 2 * lmd_max) / n)


def get_rmsd_matrix_chunk(args):
    """
    Helper function to calculate the RMSD matrix of a list of proteins in chunks.

    Args:
    - protein_list_chunk1, protein_list_chunk2: Lists of protein objects.
    - i_start, j_start: The start indexes chunks 1 and 2 respectively
    - tol: Tolerance to stop the minimization process

    Returns (tuple): start indices of both chunks and the calculated RMSD sub-matrix for the chunk.
    """
    i_start, j_start, protein_list_chunk1, protein_list_chunk2, tol = args

    i_end = i_start + len(protein_list_chunk1)
    j_end = j_start + len(protein_list_chunk2)
    rmsd_chunk = np.zeros((len(protein_list_chunk1), len(protein_list_chunk2)))
    for i in range(len(protein_list_chunk1)):
        for j in range(len(protein_list_chunk2)):
            rmsd_chunk[i, j] = get_rmsd(protein_list_chunk1[i], protein_list_chunk2[j], tol)
    rmsd_matrix_chunk = np.zeros((i_end - i_start, j_end - j_start))
    rmsd_matrix_chunk[:, :] = rmsd_chunk[:, :]
    return i_start, j_start, rmsd_matrix_chunk


def get_rmsd_matrix(protein_list, tol=10e-3, chunk_size=None):
    """
    Calculate the pairwise root-mean-square deviation (RMSD) matrix for a list of proteins. This function uses
    python's multiprocessing library to be able to run multiple processes in parallel, and speedup the pair-wise
    calculation of distances.

    Args: - protein_list (list): A list of protein structures. - tol (float): The tolerance for the RMSD calculation.
    Default is 10e-3. - chunk_size (int): The size of the chunks to split the protein list into. If None,
    all available processing units will be used.

    Returns:
    - numpy.ndarray: A square matrix of the pairwise RMSD values between all proteins in the input list.
    """
    n_prots = len(protein_list)
    rmsd_matrix = np.zeros((n_prots, n_prots))

    if chunk_size is None:
        chunk_size = int(n_prots / mp.cpu_count()) + 1

    with mp.Pool() as pool:
        chunks = []
        for i in range(0, n_prots, chunk_size):
            chunk1 = protein_list[i:i + chunk_size]
            for j in range(i + chunk_size, n_prots, chunk_size):
                chunk2 = protein_list[j:j + chunk_size]
                chunks.append((i, j, chunk1, chunk2, tol))

        results = pool.imap_unordered(get_rmsd_matrix_chunk, chunks)

        for i_start, j_start, rmsd_matrix_chunk in results:
            i_end = i_start + rmsd_matrix_chunk.shape[0]
            j_end = j_start + rmsd_matrix_chunk.shape[1]
            rmsd_matrix[i_start:i_end, j_start:j_end] = rmsd_matrix_chunk
            rmsd_matrix[j_start:j_end, i_start:i_end] = rmsd_matrix_chunk.T

    return rmsd_matrix

##  old implementation without parallelization  ##
# def get_rmsd_matrix(protein_list, tol=10e-6, max_iter=5):
#     n_prots = len(protein_list)
#     rmsd_matrix = np.zeros((n_prots, n_prots))
#     for i in range(n_prots):
#         for j in range(i + 1, n_prots):
#             rmsd_matrix[i][j] = get_rmsd(protein_list[i], protein_list[j], tol, max_iter)
#     rmsd_matrix += rmsd_matrix.T
#     return rmsd_matrix
