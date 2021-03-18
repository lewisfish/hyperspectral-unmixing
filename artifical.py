from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def create_images(abundance: np.ndarray, specs: np.ndarray, img_size: Tuple[int], target_snr:float, channels: int, show=True) -> np.ndarray:
    """Summary

    Parameters
    ----------
    abundance : np.ndarray
        Description
    specs : np.ndarray
        Description
    img_size : Tuple[int]
        Description
    channels : int
        Description

    Returns
    -------
    np.ndarray
        Description
    """

    imgs = []
    noises = []
    np.random.seed(123456789)
    for i in range(channels):
        img = np.zeros(img_size)
        for j, spec in enumerate(specs):
            img += abundance[:, :, j] * spec[i]

        img_avg = np.mean(img)
        img_avg_db = 10 * np.log10(img_avg)
        noise_avg_db = img_avg_db - target_snr
        noise_avg_pwr = 10**(noise_avg_db/10)

        noise = np.random.normal(0., np.sqrt(noise_avg_pwr), img_size)
        img += noise

        imgs.append(img.flatten())
        noises.append(noise.flatten())
    if show:
        data = np.array(imgs).reshape((18, img_size[0], img_size[1]))
        fig, axs = plt.subplots(4, 5)
        axs = axs.ravel()
        for i in range(18):
            img = data[i, :, :]
            axs[i].imshow(img)
        plt.show()
        stop
    return np.array(imgs).T, np.array(noises).T


def create_abundance_map(img_size: Tuple[int], map_type: str, ratios=[1., .75, .5, .25]) -> np.ndarray:
    """Setup artificial abundance map with 4 fluorophores

    Parameters
    ----------
    img_size : Tuple[int]
        Size to make each abundance map

    Returns
    -------
    abundance: np.ndarray, 3D (img_size.x, img_size.y, 4)
    """

    maps = ["strips", "squares"]

    assert map_type in maps, f"Not a valid map type, '{map_type}'' not in {maps}."

    if map_type == "strips":
        abundance = _strips(img_size)
    # elif map_type == "circles":
    #     abundance = _circles(img_size)
    elif map_type == "squares":
        abundance = _squares(img_size, ratios)

    return abundance


def _strips(img_size: Tuple[int]) -> np.ndarray:
    """Summary

    Parameters
    ----------
    img_size : Tuple[int]
        Description

    Returns
    -------
    np.ndarray
        Description
    """

    abundance = np.zeros((*img_size, 4))

    wall1 = int(img_size[1] / 4)
    wall2 = int(img_size[1] / 2)
    wall3 = int(3 * img_size[1] / 4)

    # loop over different fluros
    for i in range(4):
        for j in range(img_size[0]):
            for k in range(img_size[1]):
                # set nadh
                if i == 0:
                    if k <= wall1:
                        abundance[j, k, i] = np.random.uniform(0.8, 0.85000001)
                # set fad
                elif i == 1:
                    if k > wall1 and k <= wall2:
                        abundance[j, k, i] = np.random.uniform(0.8, 0.85000001)
                # set porphyrin
                elif i == 2:
                    if k > wall2 and k <= wall3:
                        abundance[j, k, i] = np.random.uniform(0.8, 0.85000001)
                # set cytc
                elif i == 3:
                    if k > wall3:
                        abundance[j, k, i] = np.random.uniform(0.8, 0.85000001)
                else:
                    print("fuck up")
    return abundance


def _squares(img_size: Tuple[int], ratios: List[float]) -> np.ndarray:
    """Summary

    Parameters
    ----------
    img_size : Tuple[int]
        Description
    ratios : List[float]
        Description

    Returns
    -------
    np.ndarray
        Description
    """

    abundance = np.zeros((*img_size, 4))

    square_width = int(img_size[0] / 10)
    gap_width = int((img_size[0] - 2.*square_width) / 6)

    pos_s = [square_width*i + (i+1)*gap_width for i in range(4)]
    pos = []
    for i in pos_s:
        for j in pos_s:
            pos.append((i, j))

# todo make this better
    count = 0
    i = 0
    for (j, k) in pos:
        if count == 0:  # pure
            abundance[k:k+square_width, j:j+square_width, i] = ratios[count]
        elif count == 1:
            abundance[k:k+square_width, j:j+square_width, i] = ratios[count]
            next_member = i + 1
            if next_member == 4:
                next_member = 0
            abundance[k:k+square_width, j:j+square_width, next_member] = 1. - ratios[count]
        elif count == 2:
            abundance[k:k+square_width, j:j+square_width, i] = ratios[count]
            next_member = i + 1
            if next_member == 4:
                next_member = 0
            abundance[k:k+square_width, j:j+square_width, next_member] = 1. - ratios[count]
        elif count == 3:
            abundance[k:k+square_width, j:j+square_width, i] = ratios[count]
            next_member = i + 1
            if next_member == 4:
                next_member = 0
            abundance[k:k+square_width, j:j+square_width, next_member] = 1. - ratios[count]
        count += 1

        if count == 4:
            count = 0
            i += 1

    return abundance


def _circles(img_size) -> np.ndarray:
    pass
