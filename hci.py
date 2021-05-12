from pathlib import Path
from statistics import mode  # type: ignore[import]
from typing import Optional, List, Union, Tuple

import numpy as np  # type: ignore[import]
from scipy import stats  # type: ignore[import]
import scipy.io  # type: ignore[import]
from skimage import io  # type: ignore[import]
from scipy.interpolate import interp1d  # type: ignore[import]
from skimage.restoration import denoise_wavelet  # type: ignore[import]
import torchvision.transforms.functional as F  # type: ignore[import]
import torch  # type: ignore[import]


class HCI(object):
    """docstring for HCI"""
    def __init__(self, img_file: Union[str, Path],
                 folder: Optional[Path] = Path("."),
                 scale: Optional[bool] = False,
                 bkg_img: Optional[np.ndarray] = None,
                 cal_img: Optional[np.ndarray] = None,
                 n_channels: Optional[int] = 69,
                 stride: Optional[int] = 32,
                 kernel_size: Optional[int] = 32,
                 mask: Optional[bool] = False,
                 mask_size: Optional[Tuple[int, int]] = (512, 512),
                 write: Optional[bool] = False,
                 out_type: Optional[str] = "3d-patches",
                 img_folder: Optional[Tuple[str, Path]] = Path(".")):
        """Summary

        Parameters
        ----------
        img_file : Union[str, Path]
            Data cube to open.
        folder : Optional[Path], optional
            Folder which contains mask files
        scale : Optional[bool], optional
            If true then output datacube is scaled from 0-1
        bkg_img : Optional[np.ndarray], optional
            Background image cube
        cal_img : Optional[np.ndarray], optional
            Calibration image cube
        n_channels : Optional[int], optional
            Number of channels to return
        stride : Optional[int], optional
            Stride for spike removal
        kernel_size : Optional[int], optional
            Size of window/kernel for spike removal
        mask : Optional[bool], optional
            If true then resize each image
        mask_size : Optional[Tuple[int, int]], optional
            Size to resize each image in datacube
        write : Optional[bool], optional
            Description
        out_type : Optional[str], optional
            Description
        img_folder : Optional[Tuple[str, Path]], optional
            Description
        """

        super(HCI, self).__init__()
        self.img_file = img_file
        self.cal_img = cal_img
        self.bkg_img = bkg_img
        self.n_channels = n_channels
        self.win = kernel_size
        self.stride = stride
        self.open_img()
        self.scale = scale
        self.roi_folder = folder
        self.mask_image = mask
        self.max_mask_x, self.max_mask_y = mask_size  # type: ignore[misc]
        self.out_type = out_type
        self.write = write
        self.img_folder = img_folder

    def preprocess(self, patch_size: Optional[Tuple[int, int]] = (32, 32), stride: Optional[int] = 16):
        """Preprocess image cube"""

        # normalise for electron multiplicatoin gain, exposure time and Quatum eff.
        self.img /= (self.emGain * self.exposure * self.QE)

        self.remove_spikes()
        self.smooth_image()
        if self.bkg_img is not None:
            self.remove_background()
            if self.cal_img is not None:
                self.flatten_image()

        if self.mask_image:
            self.mask()

        if self.scale:
            # rescale data in range 0-1 on a cube basis.
            global_min = np.amin(self.img)
            self.img += np.abs(global_min)
            global_max = np.amax(self.img)
            self.img /= global_max

        if self.write:
            self.write_out(patch_size, stride)

    def open_img(self):
        """Open .mat file and return pertinant image cube and metadata
        """

        mat = scipy.io.loadmat(self.img_file)

        # read image data
        HCI = mat["HAC_Image"][0][0]["imageStruct"]["data"][0][0]

        # read metadata for normalisation
        QE = mat["HAC_Image"][0][0]["projectStruct"][0][0]["hardware"]["camera"][0][0]["QuantumEffResponse"][0][0]
        emGain = mat["HAC_Image"][0][0]["imageStruct"]["protocol"][0][0]["channel"][0][0]["emGain"][0]
        exposure = mat["HAC_Image"][0][0]["imageStruct"]["protocol"][0][0]["channel"][0][0]["exposure"][0]
        # https://stackoverflow.com/a/35975664/6106938
        self.emGain = np.array(emGain, dtype=[("O", np.float)]).astype(np.float)[:self.n_channels]
        self.exposure = np.array(exposure, dtype=[("O", np.float)]).astype(np.float)[:self.n_channels]
        emission = mat["HAC_Image"][0][0]["imageStruct"]["protocol"][0][0]["channel"][0][0]["emission"][0]
        self.emission = np.array(emission, dtype=[("O", np.float)]).astype(np.float)

        self.get_QEs(QE)

        self.img = HCI[:, :, :self.n_channels]

    def get_QEs(self, QE: np.ndarray):
        """Get the correct Quatum eff by linear interpolation of emission values

        Parameters
        ----------
        QE : np.ndarray
            Array of Quatum eff for full wavelength range
        """

        x = QE[:, 0]
        y = QE[:, 1]
        f = interp1d(x, y)
        self.QE = f(self.emission)[:self.n_channels]

    def remove_spikes(self):
        """Remove cosmic spikes by sliding window over image cube

        Parameters
        ----------
        img : np.ndarray
            Input image with possible spikes to remove.

        Returns
        -------
        np.ndarray
            Image with removed spikes.
        """

        # order in which to view 8 connected pixels
        xvec = np.array([1, 0, -1, -1, 0, 0, 1, 1])
        yvec = np.array([0, -1, 0, 0, 1, 1, 0, 0])

        xsize, ysize, _ = self.img.shape

        self.img = self._spike_removal(self.img, self.n_channels, xvec, yvec, self.win, self.stride, xsize)

    def smooth_image(self, wavelet: Optional[str] = "sym2",
                     levels: Optional[int] = 3) -> np.ndarray:
        """Smooth image with multiple layers of wavelets.

        Parameters
        ----------
        wavelet : Optional[str], optional
            Wavelet type
        levels : Optional[int], optional
            Number of levels of wavelets to use

        """

        self.img = denoise_wavelet(self.img, mode="soft", wavelet=wavelet, wavelet_levels=levels, rescale_sigma=False, multichannel=True)

    def mask(self):

        roifile = self.roi_folder / self.img_file.with_suffix(".png").name
        self.image_mask = io.imread(roifile)

        [rows, cols] = np.where(self.image_mask)
        row1 = min(rows)
        row2 = max(rows)
        col1 = min(cols)
        col2 = max(cols)

        xdiff = row2 - row1
        remainder = self.max_mask_x - xdiff
        left = remainder // 2
        right = remainder - left
        ydiff = col2 - col1
        remainder = self.max_mask_y - ydiff
        top = remainder // 2
        bottom = remainder - top
        row1 -= left
        row2 += right
        col1 -= top
        col2 += bottom
        self.img = self.img[row1:row2, col1:col2, :]
        self.image_mask = self.image_mask[row1:row2, col1:col2]

    @staticmethod
    def _spike_removal(img: np.ndarray, n_channels: int, xvec: List[int],
                       yvec: List[int], win: int, stride: int,
                       xsize: int) -> np.ndarray:
        """Remove spikes above 5 stds of the mode of a sliding window width
           win and stride stride

        Parameters
        ----------
        img : np.ndarray
            Image to remove spikes from
        n_channels : int
            Number of channels to operate on
        xvec : List[int]
            list of pixels to visit, x
        yvec : List[int]
            list of pixels to vist, y
        win : int
            window size
        stride : int
            stride size
        xsize : int
            size of image. Assume square images

        Returns
        -------
        np.ndarray
            Image with smoothed spikes
        """

        for channel in range(n_channels):
            xpos = 0
            ypos = 0
            while True:
                subImg = img[xpos:xpos + win, ypos:ypos + win, channel]
                modeVal = stats.mode(subImg, axis=None)
                std = 5 * np.std(subImg)
                mask = np.where(subImg >= std + modeVal.mode[0]) or np.where(subImg <= modeVal.mode[0] - std)

                cpixels = list(zip(*mask))
                # loop over pixels in pixel array
                pixels = []
                if len(cpixels) != 0:
                    for x, y in cpixels:
                        xcur = x
                        ycur = y
                        for i in range(0, 8):
                            xcur += xvec[i]
                            ycur += yvec[i]
                            if xcur >= win or ycur >= win or xcur < 0 or ycur < 0:
                                continue
                            pixels.append(subImg[xcur, ycur])
                        mean_val = np.array(pixels).mean()
                        subImg[x, y] = mean_val
                img[xpos:xpos + win, ypos:ypos + win, channel] = subImg

                xpos += stride
                if xpos >= xsize:
                    ypos += stride
                    xpos = 0
                    if ypos >= xsize:
                        break

        return img

    def remove_background(self):
        """Open and return background HCI image"""

        self.img -= self.bkg_img

    def flatten_image(self):
        """Flatten image to compensate for uneven illumination."""

        self.img -= self.cal_img

    def write_out(self, patch_size, stride):
        if self.out_type == "2d-images":
            for i, band in enumerate(range(self.img.shape[-1])):
                with open(self.img_folder / f"expanded/{file.stem}_{i}.npy", "wb") as f:
                    np.save(f, self.img[:, :, band])
        elif self.out_type == "3d-patches":
            mask = torch.from_numpy(self.image_mask).unsqueeze(0)
            mask = F.resize(mask, (320, 320)).squeeze()

            HCI = self.img.astype(np.float32)
            HCI = torch.from_numpy(HCI)
            HCI = HCI.permute(2, 0, 1)
            HCI = F.resize(HCI, (320, 320))

            xpos = 0
            ypos = 0
            self.counter = 0
            while True:
                sample = HCI[:, ypos:ypos + patch_size[0], xpos:xpos + patch_size[1]]
                if mask[ypos + stride, xpos + stride] == 255:
                    with open(self.img_folder / f"{self.img_file.stem}_{self.counter}.npy", "wb") as f:
                        np.save(f, sample.numpy())
                self.counter += 1
                xpos += stride
                if xpos + patch_size[1] > HCI.shape[1]:
                    xpos = 0
                    ypos += stride
                if ypos + patch_size[0] > HCI.shape[2]:
                    break
