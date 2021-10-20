
import numpy as np

from scipy.sparse import diags



def adjacency2degree(A):
    """ Compute the degree matrix for a give adjacency matrix A"""
    return diags(np.asarray(A.sum(1)).reshape(-1), format="csc")


def adjacency2laplacian(A, D=None):
    """
    This function create a graph laplacian matrix from the adjacency matrix.
    Parameters
    ----------
    A (sparse matrix): Adjacency matrix.
    D (Degree matrix): Optional, diagonal matrix containing the sum over the adjacency row.


    Returns
    -------
    L (sparse matrix): graph Laplacian  L = D - A.
    """
    if D is None:  # compute the degree matrix
        D = adjacency2degree(A)

        return D - A.T



def adjacency2transition(A, D=None):
    """ Compute the transition matrix associated with the adjacency matrix A"""
    if D is None:
        D = adjacency2degree(A)
    return D.power(-1)*A 
