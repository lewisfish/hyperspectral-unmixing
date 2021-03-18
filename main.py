from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

import artifical
import hysime
import nfindr
import utils
import fcls
import vca


def get_abundances(data: np.ndarray, num_members: int) -> np.ndarray:
    """Summary

    Parameters
    ----------
    data : np.ndarray
        Description
    num_members : int
        Description

    Returns
    -------
    np.ndarray
        Description
    """

    from sklearn.neighbors import NearestNeighbors

    scaler = StandardScaler().fit(data)
    scaled_data = scaler.transform(data)

    pca = PCA(n_components=.95, svd_solver="full").fit(scaled_data)
    pca_data = pca.transform(scaled_data)

    # neigh = NearestNeighbors(n_neighbors=2)
    # nbrs = neigh.fit(scaled_data)
    # distances, indices = nbrs.kneighbors(scaled_data)
    # distances = np.sort(distances, axis=0)
    # distances = distances[:, 1]
    # plt.plot(distances)
    # plt.show()
    # stop

    kmeans = KMeans(n_clusters=num_members, random_state=0)
    # kmeans = DBSCAN(eps=1.5)
    # kmeans.fit(pca_data)
    labels = kmeans.fit_predict(pca_data)

    fig, ax = plt.subplots(1, 3)
    print(pca_data.shape)
    ax[0].scatter(pca_data[:, 0], pca_data[:, 1], c=labels)
    ax[1].scatter(pca_data[:, 0], pca_data[:, 2], c=labels)
    ax[2].scatter(pca_data[:, 1], pca_data[:, 2], c=labels)

    return labels, pca, kmeans


def get_spectra(data: np.ndarray, num_members: int) -> np.ndarray:
    """Summary

    Parameters
    ----------
    data : np.ndarray
        Description
    num_members : int
        Description

    Returns
    -------
    np.ndarray
        Description
    """

    new_data = np.transpose(data, axes=[1, 0])
    scaler = StandardScaler().fit(new_data)
    scaled_data = scaler.transform(new_data)

    # pca = FastICA(n_components=4).fit(scaled_data)
    pca = PCA(n_components=4, svd_solver="full").fit(scaled_data)
    pca_data = pca.transform(scaled_data)

    # for i in range(4):
    #     pca_data[:, i] = pca_data[:, i] + np.abs(np.amin(pca_data[:, i]))
    #     pca_data[:, i] /= np.amax(pca_data[:, i])

    fig, ax = plt.subplots(1, 1)
    ax.plot(np.abs(pca_data[:, 0]))
    ax.plot(np.abs(pca_data[:, 1]))
    ax.plot(np.abs(pca_data[:, 2]))
    ax.plot(np.abs(pca_data[:, 3]))
    plt.show()
    # return labels, pca, kmeans


def get_ems(data: np.ndarray, num_members: int, cube_size: Tuple[int], method: str) -> np.ndarray:
    """Summary

    Parameters
    ----------
    data : np.ndarray
        Description
    num_members : int
        Description
    cube_size : Tuple[int]
        Description

    Returns
    -------
    np.ndarray
        Description
    """

    methods = ["nfindr", "vca"]

    assert method in methods, f"{method} not in {methods}"

    if method == "nfindr":
        ee = nfindr.NFINDR()
        data = data.reshape(cube_size)
        U = ee.extract(data, num_members, maxit=10, normalize=True, ATGP_init=False)
    elif method == "vca":
        data = np.transpose(data, (1, 0))
        U, _, _ = vca.vca(data, num_members)
        U = np.transpose(U, (1, 0))

    return U


def run_artifical_analysis(img_size: Tuple[int], target_snr: float) -> None:
    """Summary

    Parameters
    ----------
    img_size : Tuple[int]
        Description
    target_snr : float
        Description
    """

    import matplotlib.gridspec as gridspec

    w, h, channels = img_size

    files = ["fad.csv", "free_nadh.csv", "cytc.csv", "porphyrin.csv"]

    nadh_spec = utils.read_fluro_spec(files[0], channels=channels)
    fad_spec = utils.read_fluro_spec(files[1], channels=channels)
    porphyrin_spec = utils.read_fluro_spec(files[2], channels=channels)
    cytc_spec = utils.read_fluro_spec(files[3], channels=channels)

    specs = [fad_spec, nadh_spec, cytc_spec, porphyrin_spec]

    abundance = artifical.create_abundance_map((w, h), "strips", ratios=[1.,1.,1.,1.])
    images, noises = artifical.create_images(abundance, specs, (w, h), target_snr=target_snr, channels=channels, show=True)

    # cov = np.cov(noises.T)
    noises, cov = hysime.est_noise(images)
    a, b = hysime.hysime(images, noises, cov)
    print(f"There are {a} end members")
    U = get_ems(images, a, img_size, method="vca")

    fig = plt.figure()
    spec = gridspec.GridSpec(ncols=4, nrows=3, figure=fig)
    ems_ax = fig.add_subplot(spec[0, :])
    gt_ax = [fig.add_subplot(spec[1, 0]), fig.add_subplot(spec[1, 1]), fig.add_subplot(spec[1, 2]), fig.add_subplot(spec[1, 3])]
    pd_ax = [fig.add_subplot(spec[2, 0]), fig.add_subplot(spec[2, 1]), fig.add_subplot(spec[2, 2]), fig.add_subplot(spec[2, 3])]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    newU = []
    for i in range(a):
        tmp_u = U[i, :] / np.amax(U[i, :])
        val = 1000000000
        idx = -1
        for j in range(a):
            chi = np.sum(np.abs(tmp_u - specs[j]))
            if chi < val:
                val = chi
                idx = j
        newU.append(idx)

    for i, j in enumerate(newU):
        ems_ax.plot(U[i, :] / np.amax(U[i, :]), color=colors[i], marker="x", ls="")
        ems_ax.plot(specs[j], color=colors[i], label=files[i], ls=":")
    ems_ax.set_xlabel("Channel")
    ems_ax.set_ylabel("Signal/arb.")
    ems_ax.set_xticks(np.arange(0, 18, 2))
    ems_ax.legend()

    obj = fcls.FCLS()
    pt_abundence = obj.map(images.reshape(img_size), U)

    for i, ax in enumerate(pd_ax):
        ax.imshow(pt_abundence[:, :, i])

    for i, ax in zip(newU, gt_ax):
        ax.imshow(abundance[:, :, i])
        ax.set_title(files[i][:-3])

    # get_spectra(images, a)
    # stop
    # labels, pca, kmeans = get_abundances(images, a)

    # scaler = StandardScaler().fit(images)
    # scaled_data = scaler.transform(images)

    # pixels = np.zeros((w*h))
    # for i in range(w*h):
    #     single_pixel = scaled_data[i, :]
    #     pcaed = pca.transform(single_pixel.reshape(1, -1))
    #     pixels[i] = kmeans.predict(pcaed)

    # pd_ax.imshow(pixels.reshape((w, h)))
    # pd_ax.set_title("Predicted")

    plt.show()


def real_data(file: str) -> None:
    """Summary

    Parameters
    ----------
    file : str
        Description
    """

    hdr, imgs = utils.load_data(file)
    imgs = imgs.reshape((1024*1024, 70))

    noise, noiseCov = hysime.est_noise(imgs)
    a, b = hysime.hysime(imgs, noise, noiseCov)

# # todo
# # look at other methods?
# # GANS? -> conv nets?
# # Flip image dims around ((w*h), c) -> (c, (w*h)) gives spectra?


# a, b, Rx = hysime(images, noises, cov)
# # proj_matrix = np.dot(images, np.dot(b, b.T))
# # tmp = np.ones((224, 5))
# new = np.matmul(np.matmul(b, b.T), images.T)

# yp, up, my = affineProj(new, 4)
# Y = np.matmul(up.T, images.T)

# E_I = np.eye(4)

# I, J, K = 0, 1, 2

# v1 = E_I[:, I]
# v2 = E_I[:, J]
# v3 = E_I[:, K]

# Y = np.matmul(np.array([v1, v2, v3]), Y)
# plt.scatter(Y[0, :], Y[1, :])
# plt.show()


if __name__ == '__main__':

    run_artifical_analysis((24, 25, 18), 5.)
    # real_data("data/D080601.mat")
