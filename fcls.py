#
# Code taken from PySptools by Christian Therien
# https://pysptools.sourceforge.io/index.html
#
import numpy as np


def _numpy_to_cvxopt_matrix(A):
    from cvxopt import matrix
    A = np.array(A, dtype=np.float64)
    if A.ndim == 1:
        return matrix(A, (A.shape[0], 1), 'd')
    else:
        return matrix(A, A.shape, 'd')


def _numpy_None_vstack(A1, A2):
    if A1 is None:
        return A2
    else:
        return np.vstack([A1, A2])


def _numpy_None_concatenate(A1, A2):
    if A1 is None:
        return A2
    else:
        return np.concatenate([A1, A2])


def _normalize(M):
    """
    Normalizes M to be in range [0, 1].

    Parameters:
      M: `numpy array`
          1D, 2D or 3D data.

    Returns: `numpy array`
          Normalized data.
    """
    minVal = np.min(M)
    maxVal = np.max(M)

    Mn = M - minVal

    if maxVal == minVal:
        return np.zeros(M.shape)
    else:
        return Mn / (maxVal-minVal)


class FCLS(object):
    """
    Performs fully constrained least squares. Fully constrained least squares
    is least squares with the abundance sum-to-one constraint (ASC) and the
    abundance nonnegative constraint (ANC).
    """

    def __init__(self):
        self.amap = None
        self.m = None
        self.n = None
        self.q = None

    def map(self, M, U, normalize=False, mask=None):
        """
        Performs fully constrained least squares of each pixel in M
        using the endmember signatures of U.

        Parameters:
        M: `numpy array`
             A HSI cube (m x n x p).

          U: `numpy array`
             A spectral library of endmembers (q x p).

          normalize: `boolean [default False]`
             If True, M and U are normalized before doing the spectra mapping.

          mask: `numpy array [default None]`
             A binary mask, when *True* the selected pixel is unmixed.

        Returns: `numpy array`
              An abundance maps (m x n x q).
        """

        h, w, numBands = M.shape
        if normalize:
            M = _normalize(M)
            U = _normalize(U)
        Mr = np.reshape(M, (w*h, numBands))
        if isinstance(mask, np.ndarray):
            m = np.reshape(mask, (w*h))
            cMr = _compress(Mr, m)
            c_amap2D = fcls_real(cMr, U)
            amap2D = _expand(c_amap2D, m, w*h, U.shape[0])
        else:
            amap2D = fcls_real(Mr, U)
        self.amap = np.reshape(amap2D, (h, w, U.shape[0]))
        self.m = h
        self.n = w
        self.q = U.shape[0]
        return self.amap

    def __str__(self):
        return 'pysptools.abundance_maps.amaps_int.FCLS object, maps: {0}x{1}x{2}'.format(self.m, self.n, self.q)

    def plot(self, path, mask= None, interpolation='none', colorMap='jet', columns=None, suffix=None):
        _plot_abundance_map(self.amap, path, 'FCLS', mask, interpolation, colorMap, columns, suffix)

    def display(self, mask= None, interpolation='none', colorMap='jet', columns=None, suffix=None):
        _plot_abundance_map(self.amap, None, 'FCLS', mask, interpolation, colorMap, columns, suffix)


def fcls_real(M, U):
    """
    Performs fully constrained least squares of each pixel in M
    using the endmember signatures of U. Fully constrained least squares
    is least squares with the abundance sum-to-one constraint (ASC) and the
    abundance nonnegative constraint (ANC).

    Parameters:
        M: `numpy array`
            2D data matrix (N x p).

        U: `numpy array`
            2D matrix of endmembers (q x p).

    Returns: `numpy array`
        An abundance maps (N x q).

    References:
         Daniel Heinz, Chein-I Chang, and Mark L.G. Fully Constrained
         Least-Squares Based Linear Unmixing. Althouse. IEEE. 1999.

    Notes:
        Three sources have been useful to build the algorithm:
            * The function hyperFclsMatlab, part of the Matlab Hyperspectral
              Toolbox of Isaac Gerg.
            * The Matlab (tm) help on lsqlin.
            * And the Python implementation of lsqlin by Valera Vishnevskiy, click:
              http://maggotroot.blogspot.ca/2013/11/constrained-linear-least-squares-in.html
              , it's great code.
    """
    from cvxopt import solvers, matrix
    solvers.options['show_progress'] = False
    N, p1 = M.shape
    nvars, p2 = U.shape

    C = _numpy_to_cvxopt_matrix(U.T)
    Q = C.T * C

    lb_A = -np.eye(nvars)
    lb = np.repeat(0, nvars)
    A = _numpy_None_vstack(None, lb_A)
    b = _numpy_None_concatenate(None, -lb)
    A = _numpy_to_cvxopt_matrix(A)
    b = _numpy_to_cvxopt_matrix(b)

    Aeq = _numpy_to_cvxopt_matrix(np.ones((1,nvars)))
    beq = _numpy_to_cvxopt_matrix(np.ones(1))

    M = np.array(M, dtype=np.float64)
    X = np.zeros((N, nvars), dtype=np.float32)
    for n1 in range(N):
        d = matrix(M[n1], (p1, 1), 'd')
        q = - d.T * C
        sol = solvers.qp(Q, q.T, A, b, Aeq, beq, None, None)['x']
        X[n1] = np.array(sol).squeeze()
    return X
