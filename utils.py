import random
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.sparse.linalg import svds


def load_data(file):
    mat = scipy.io.loadmat(file)
    data = mat["HAC_Image"]

    img_stack = data[0][0]["imageStruct"]["data"][0][0]

    hdr = []
    hdrtmp = data[0][0]["imageStruct"]["protocol"][0][0]["channel"][0][0]["excitationWavelength"]
    for i in range(70):
        hdr.append(float(hdrtmp[0][i]))

    return np.array(hdr), img_stack


def read_fluro_spec(file: str, channels: int) -> np.ndarray:
    """Read in spectrum and interpolate to get required number of channels

    Parameters
    ---------

    file: str
        File to read in.
    channels: int
        Number of channels to return

    Returns
    -------

    yp: np.ndarray, 1D
        Array containg the y values for each channel (1 - channel)

    """

    x, y = np.loadtxt(file, unpack=True, delimiter=",")
    # only want n channels
    xp = np.linspace(1, channels, channels)
    yp = np.interp(xp, x, y)

    return yp / np.amax(yp)


def affineProj(y, p):

    my = np.mean(y, 1)
    my = my[:, np.newaxis]

    L, samp_size = y.shape

    yp = y - np.tile(my, (1, samp_size))

    Up, D, vt = svds(np.matmul(yp, yp.T)/samp_size, p-1)
    # represent yp in the subspace R^(p-1)
    yp = np.matmul(Up.T, yp)
    # compute the orthogonal component of my
    my_ortho = np.matmul(np.matmul(my-Up, Up.T), my)
    # define the p-th orthonormal direction

    Up = np.c_[Up, my_ortho/np.sqrt(np.sum(my_ortho**2))]

    # compute my coordinates wrt Up
    myp = np.matmul(Up.T, my)
    yp[-1, :] = 0     # augmented dimension
    yp = np.vstack([yp, np.zeros((1, 600))])
    # # lift yp to R^p
    yp = yp + np.tile(myp, (1, samp_size))

    return yp, Up, my
