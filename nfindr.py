#
# Code taken from PySptools by Christian Therien
# https://pysptools.sourceforge.io/index.html
#
import math
import random

import numpy as np
import scipy as sp
from sklearn.decomposition import PCA


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


def nfindr_real(data, q, transform=None, maxit=None, ATGP_init=False):
    """
    N-FINDR endmembers induction algorithm.
    Parameters:
        data: `numpy array`
            Column data matrix [nvariables x nsamples].
        q: `int`
            Number of endmembers to be induced.
        transform: `numpy array [default None]`
            The transformed 'data' matrix by MNF (N x components). In this
            case the number of components must == q-1. If None, the built-in
            call to PCA is used to transform the data.
        maxit: `int [default None]`
            Maximum number of iterations. Default = 3*q.
        ATGP_init: `boolean [default False]`
            Use ATGP to generate the first endmembers set instead
            of a random selection.
    Returns: `tuple: numpy array, numpy array, int`
        * Set of induced endmembers (N x p)
        * Set of transformed induced endmembers (N x p)
        * Array of indices into the array data corresponding to the
          induced endmembers
        * The number of iterations.
    References:
      Winter, M. E., "N-FINDR: an algorithm for fast autonomous spectral
      end-member determination in hyperspectral data", presented at the Imaging
      Spectrometry V, Denver, CO, USA, 1999, vol. 3753, pgs. 266-275.
    """
    # data size
    nsamples, nvariables = data.shape
    random.seed(123456789)
    if maxit is None:
        maxit = 3*q

    if transform is None:
        # transform as shape (N x p)
        transform = data
        transform = PCA(n_components=q-1).fit_transform(data)
    else:
        transform = transform

    # Initialization
    # TestMatrix is a square matrix, the first row is set to 1
    TestMatrix = np.zeros((q, q), dtype=np.float32, order='F')
    TestMatrix[0, :] = 1
    IDX = None
    if ATGP_init:
        induced_em, idx = eea.ATGP(transform, q)
        IDX = np.array(idx, dtype=np.int64)
        for i in range(q):
            TestMatrix[1:q, i] = induced_em[i]
    else:
        IDX = np.zeros((q), dtype=np.int64)
        for i in range(q):
            idx = int(math.floor(random.random()*nsamples))
            TestMatrix[1:q, i] = transform[idx]
            IDX[i] = idx

    actualVolume = 0
    it = 0
    v1 = -1.0
    v2 = actualVolume

    while it <= maxit and v2 > v1:
        for k in range(q):
            for i in range(nsamples):
                TestMatrix[1:q, k] = transform[i]
                volume = math.fabs(sp.linalg._flinalg.sdet_c(TestMatrix)[0])
                if volume > actualVolume:
                    actualVolume = volume
                    IDX[k] = i
            TestMatrix[1:q, k] = transform[IDX[k]]
        it = it + 1
        v1 = v2
        v2 = actualVolume

    E = np.zeros((len(IDX), nvariables), dtype=np.float32)
    Et = np.zeros((len(IDX), q-1), dtype=np.float32)
    for j in range(len(IDX)):
        E[j] = data[IDX[j]]
        Et[j] = transform[IDX[j]]

    return E, Et, IDX, it


class NFINDR(object):
    """
    N-FINDR endmembers induction algorithm.
    """

    def __init__(self):
        self.E = None
        self.Et = None
        self.w = None
        self.idx = None
        self.it = None
        self.idx3D = None
        self.is_normalized = False

    # @ExtractInputValidation2('NFINDR')
    def extract(self, M, q, transform=None, maxit=None, normalize=False, ATGP_init=False, mask=None):
        """
        Extract the endmembers.

        Parameters:
            M: `numpy array`
                A HSI cube (m x n x p).

            q: `int`
                The number of endmembers to be induced.

            transform: `numpy array [default None]`
                The transformed 'M' cube by MNF (m x n x components). In this
                case the number of components must == q-1. If None, the built-in
                call to PCA is used to transform M in q-1 components.

            maxit: `int [default None]`
                The maximum number of iterations. Default is 3*p.

            normalize: `boolean [default False]`
                If True, M is normalized before doing the endmembers induction.

            ATGP_init: `boolean [default False]`
                Use ATGP to generate the first endmembers set instead
                of a random selection.

            mask: `numpy array [default None]`
                A binary mask, when *True* the corresponding signal is part of the
                endmembers search.

        Returns: `numpy array`
            Set of induced endmembers (N x p).

        References:
            Winter, M. E., "N-FINDR: an algorithm for fast autonomous spectral
            end-member determination in hyperspectral data", presented at the Imaging
            Spectrometry V, Denver, CO, USA, 1999, vol. 3753, pgs. 266-275.

        Note:
            The division by (factorial(p-1)) is an invariant for this algorithm,
            for this reason it is skipped.
        """
        # from . import nfindr
        if normalize:
            M = _normalize(M)
            self.is_normalized = True
        h, w, numBands = M.shape
        self.h, self.w, self.numBands = M.shape
        self.q = q
        M = np.reshape(M, (self.w*h, M.shape[2]))
        if transform is not None:
            transform = np.reshape(transform, (self.w*h, transform.shape[2]))

        if isinstance(mask, np.ndarray):
            m = np.reshape(mask, (self.w*h))
            cM = _compress(M, m)
        else:
            cM = M

        self.E, self.Et, self.idx, self.it = nfindr_real(cM, q, transform, maxit, ATGP_init)
        self.idx3D = [(i % self.w, i // self.w) for i in self.idx]
        return self.E

    def __str__(self):
        return 'pysptools.eea.eea_int.NFINDR object, hcube: {0}x{1}x{2}, n endmembers: {3}'.format(self.h, self.w, self.numBands, self.q)

    def get_idx(self):
        """
        Returns : numpy array
            Array of indices into the HSI cube corresponding to the
            induced endmembers
        """
        return self.idx3D

    def get_iterations(self):
        """
        Returns : int
            The number of iterations.
        """
        return self.it

    def get_endmembers_transform(self):
        return self.Et

    # @PlotInputValidation('NFINDR')
    # def plot(self, path, axes=None, suffix=None):
    #     _plot_end_members(path, self.E, 'NFINDR', self.is_normalized, axes=axes, suffix=suffix)

    # @DisplayInputValidation('NFINDR')
    # def display(self, axes=None, suffix=None):
    #     _display_end_members(self.E, 'NFINDR', self.is_normalized, axes=axes, suffix=suffix)
