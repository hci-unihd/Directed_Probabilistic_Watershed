from scipy.sparse.linalg import bicg,bicgstab,cgs,gmres,lgmres,qmr
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix
import numpy as np
#%%


def direct_solver(A, b, max_workers=None):
    """ Simple wrapper around scipy spsolve """
    if isinstance(b, np.ndarray):
        return spsolve(A, b, use_umfpack=True)
    else:
        return spsolve(A, b, use_umfpack=True).toarray()


def solve_bicg(A, b, tol=1e-5):
    """
    Implementation follows the source code of skimage:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/random_walker_segmentation.py
    it solves the linear system of equations: Ax = b,
    by biconjugate gradient
    Parameters
    ----------
    A: Sparse csr matrix (NxN)
    b: Sparse array or array (NxM)
    tol: result tolerance

    returns x array (NxM)
    -------
    """
    pu = []
    A = csr_matrix(A)

    # The actual cast will be performed slice by slice to reduce memory footprint
    check_type = True if type(b) == np.ndarray else False



    for i in range(b.shape[-1]):
        _b = b[:, i].astype(np.float32) if check_type else b[:, i].todense().astype(np.float32)
        _pu = bicg(A, _b, tol=tol)[0].astype(np.float32)
        pu.append(_pu)

    return np.array(pu, dtype=np.float32).T


def solve_bicgstab(A, b, tol=1e-8):
    """
    Implementation follows the source code of skimage:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/random_walker_segmentation.py
    it solves the linear system of equations: Ax = b,
    by biconjugate gradient stabilized
    Parameters
    ----------
    A: Sparse csr matrix (NxN)
    b: Sparse array or array (NxM)
    tol: result tolerance

    returns x array (NxM)
    -------
    """
    pu = []
    A = csr_matrix(A)

    # The actual cast will be performed slice by slice to reduce memory footprint
    check_type = True if type(b) == np.ndarray else False


    for i in range(b.shape[-1] ):
        _b = b[:, i].astype(np.float32) if check_type else b[:, i].todense().astype(np.float32)
        _pu = bicgstab(A, _b, tol=tol)[0].astype(np.float32)
        pu.append(_pu)
    return np.array(pu, dtype=np.float32).T



def solve_gmres(A, b, tol=1e-5):
    """
    Implementation follows the source code of skimage:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/random_walker_segmentation.py
    it solves the linear system of equations: Ax = b,
    by biconjugate gradient
    Parameters
    ----------
    A: Sparse csr matrix (NxN)
    b: Sparse array or array (NxM)
    tol: result tolerance

    returns x array (NxM)
    -------
    """
    pu = []
    A = csr_matrix(A)

    # The actual cast will be performed slice by slice to reduce memory footprint
    check_type = True if type(b) == np.ndarray else False



    for i in range(b.shape[-1]):
        _b = b[:, i].astype(np.float32) if check_type else b[:, i].todense().astype(np.float32)
        _pu = gmres(A, _b, tol=tol)[0].astype(np.float32)
        pu.append(_pu)

    return np.array(pu, dtype=np.float32).T


def solve_lgmres(A, b, tol=1e-5):
    """
    Implementation follows the source code of skimage:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/random_walker_segmentation.py
    it solves the linear system of equations: Ax = b,
    by biconjugate gradient
    Parameters
    ----------
    A: Sparse csr matrix (NxN)
    b: Sparse array or array (NxM)
    tol: result tolerance

    returns x array (NxM)
    -------
    """
    pu = []
    A = csr_matrix(A)

    # The actual cast will be performed slice by slice to reduce memory footprint
    check_type = True if type(b) == np.ndarray else False



    for i in range(b.shape[-1]):
        _b = b[:, i].astype(np.float32) if check_type else b[:, i].todense().astype(np.float32)
        _pu = lgmres(A, _b, tol=tol)[0].astype(np.float32)
        pu.append(_pu)

    return np.array(pu, dtype=np.float32).T


def solve_qmr(A, b, tol=1e-5):
    """
    Implementation follows the source code of skimage:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/random_walker_segmentation.py
    it solves the linear system of equations: Ax = b,
    by biconjugate gradient
    Parameters
    ----------
    A: Sparse csr matrix (NxN)
    b: Sparse array or array (NxM)
    tol: result tolerance

    returns x array (NxM)
    -------
    """
    pu = []
    A = csr_matrix(A)

    # The actual cast will be performed slice by slice to reduce memory footprint
    check_type = True if type(b) == np.ndarray else False



    for i in range(b.shape[-1]):
        _b = b[:, i].astype(np.float32) if check_type else b[:, i].todense().astype(np.float32)
        _pu = qmr(A, _b, tol=tol)[0].astype(np.float32)
        pu.append(_pu)

    return np.array(pu, dtype=np.float32).T

def solve_cgs(A, b, tol=1e-5):
    """
    Implementation follows the source code of skimage:
    https://github.com/scikit-image/scikit-image/blob/master/skimage/segmentation/random_walker_segmentation.py
    it solves the linear system of equations: Ax = b,
    by biconjugate gradient
    Parameters
    ----------
    A: Sparse csr matrix (NxN)
    b: Sparse array or array (NxM)
    tol: result tolerance

    returns x array (NxM)
    -------
    """
    pu = []
    A = csr_matrix(A)

    # The actual cast will be performed slice by slice to reduce memory footprint
    check_type = True if type(b) == np.ndarray else False



    for i in range(b.shape[-1]):
        _b = b[:, i].astype(np.float32) if check_type else b[:, i].todense().astype(np.float32)
        _pu = cgs(A, _b, tol=tol)[0].astype(np.float32)
        pu.append(_pu)

    return np.array(pu, dtype=np.float32).T
