from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import scipy.io
from skimage import io
from skimage.transform import resize
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import artifical
import utils


class AusDataModule(pl.LightningDataModule):
    """docstring for AusDataModule"""
    def __init__(self, data_dir, cube=False, normalise=True, transform=None, batch_size=576):
        super(AusDataModule, self).__init__()
        self.data_dir = data_dir
        self.normalise = normalise
        self.batch_size = batch_size
        self.transform = transform
        self.cube = cube

    def setup(self, stage=None):

        if stage == "fit" or stage is None:
            if self.cube:
                self.train = AusDataCube(self.data_dir / "train", normalise=self.normalise, transform=self.transform)
                self.val = AusDataCube(self.data_dir / "val", normalise=self.normalise, transform=self.transform)
            else:
                self.train = AusDataImg(self.data_dir / "train", normalise=self.normalise, transform=self.transform)
                self.val = AusDataImg(self.data_dir / "val", normalise=self.normalise, transform=self.transform)

        if stage == "test" or stage is None:
            if self.cube:
                self.test = AusDataCube(self.data_dir / "test", normalise=self.normalise, transform=self.transform)
            else:
                self.test = AusDataImg(self.data_dir / "test", normalise=self.normalise, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=32, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=32)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=32)


class AusDataImg(Dataset):
    """docstring for AusDataImg"""
    def __init__(self, folder,  normalise=False, transform=None):
        super(AusDataImg, self).__init__()
        self.folder = folder
        self.transform = transform
        self.normalise = normalise

        self.label_dict = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.files = list(folder.glob("A*.npy"))
        self.files += list(folder.glob("B*.npy"))
        self.shape = np.load(self.files[0]).shape

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]
        target = self.label_dict[str(file.stem)[0]]
        sample = np.load(file)

        sample = resize(sample, (263, 263))

        if self.transform:
            self.transform(sample)

        if self.normalise:
            # normalise on per channel basis
            sample /= np.max(sample)

        sample = torch.from_numpy(sample)
        sample = sample.unsqueeze(0)
        return sample, target


class AusDataCube(Dataset):
    """docstring for AusDataImg"""
    def __init__(self, folder,  normalise=False, transform=None):
        super(AusDataCube, self).__init__()
        self.folder = folder
        self.transform = transform
        self.normalise = normalise

        self.label_dict = {"A": 0, "B": 1, "C": 2, "D": 3}

        self.files = list(folder.glob("A*.npy"))
        self.files += list(folder.glob("B*.npy"))
        self.shape = np.load(self.files[0]).shape

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        file = self.files[idx]
        target = self.label_dict[str(file.stem)[0]]
        sample = np.load(file)

        if self.transform:
            self.transform(sample)

        if self.normalise:
            # normalise on per channel basis
            sample /= np.max(sample)

        sample = torch.from_numpy(sample)
        sample = sample.unsqueeze(0)
        return sample, target


# (A) 0%
# (B) 100%
# (C) 50%
# (D) 75%
# (E) Control


class AusData(Dataset):
    """docstring for AusData"""
    def __init__(self, folder, pad=False, normalise=False, transform=None):
        super(AusData, self).__init__()
        self.folder = folder
        self.transform = transform
        self.HCI = None
        self.imgNum = 0
        files = folder.glob("*.mat")
        pngs = folder / "ROIs"
        pngs = pngs.glob("*.png")
        self.idx = 0

        xsize = 0
        ysize = 0
        for png in pngs:
            mask = io.imread(png)

            [rows, cols] = np.where(mask)
            row1 = min(rows)
            row2 = max(rows)
            col1 = min(cols)
            col2 = max(cols)
            rdiff = row2 - row1
            cdiff = col2 - col1
            if rdiff > xsize:
                xsize = rdiff
            if cdiff > ysize:
                ysize = cdiff

        for i, file in enumerate(files):
            mat = scipy.io.loadmat(file)
            roifile = self.folder / "ROIs" / file.with_suffix(".png").name
            mask = io.imread(roifile)

            [rows, cols] = np.where(mask)
            row1 = min(rows)
            row2 = max(rows)
            col1 = min(cols)
            col2 = max(cols)
            newMask = mask[row1:row2, col1:col2]

            if pad:
                xdiff = row2 - row1
                remainder = xsize - xdiff
                left = remainder // 2
                right = remainder - left
                ydiff = col2 - col1
                remainder = ysize - ydiff
                top = remainder // 2
                bottom = remainder - top
                row1 -= left
                row2 += right
                col1 -= top
                col2 += bottom
                # newMask = np.pad(newMask, ((left, right), (top, bottom)), "constant", constant_values=0)

            HCI = mat["HAC_Image"][0][0]["imageStruct"]["data"][0][0]
            mask = np.repeat(newMask[:, :, np.newaxis], 70, axis=-1)
            HCI = HCI[row1:row2, col1:col2, :]# * mask
            # self.HCI = np.where(self.HCI == 0, np.mean(self.HCI, axis=(1, 0)), self.HCI)
            shape = HCI.shape

            if normalise:
                # normalise on per channel basis
                maxvals = np.max(HCI, axis=(0, 1))
                HCI /= maxvals

            HCI = HCI.reshape((HCI.shape[0] * HCI.shape[1], HCI.shape[2]))
            HCI = HCI.astype(np.float32)
            if self.HCI is not None:
                self.HCI = np.dstack((self.HCI, HCI))
            else:
                self.HCI = HCI
            print(f"read {i+1} images")
        self.shape = (shape[0], shape[1], shape[2], self.HCI.shape[-1])

    def __len__(self):
        return self.HCI.shape[0] * self.HCI.shape[-1]

    def __getitem__(self, idx):

        imgNum = idx // self.HCI.shape[0]
        newIDX = idx - (imgNum*self.HCI.shape[0])

        sample = self.HCI[newIDX, :, imgNum]

        if self.transform:
            sample = self.transform(sample)

        return sample


class PaviaUDataset(Dataset):
    """docstring for PaviaUDataset"""
    def __init__(self, file, transform=None):
        super(PaviaUDataset, self).__init__()

        self.HCI = scipy.io.loadmat(file)["paviaU"]
        self.gt = scipy.io.loadmat(file[:-4] + "_gt.mat")["paviaU_gt"]
        shape = self.HCI.shape
        self.HCI = self.HCI.reshape((shape[0] * shape[1], shape[2]))
        self.HCI = self.HCI.astype(np.float32)
        self.shape = self.HCI.shape

        self.transform = transform

    def __len__(self):
        return self.HCI.shape[0]

    def __getitem__(self, idx):

        sample = self.HCI[idx, :]

        if self.transform:
            sample = self.transform(sample)

        return sample


class UrbanDataset(Dataset):
    """docstring for UrbanDataset"""
    def __init__(self, data_file, gt_file, transform=None, patches=False, patch_size=(40, 40), stride=1):
        super(UrbanDataset, self).__init__()
        self.data_file = data_file
        self.gt_file = gt_file
        self.transform = transform
        self.patches = patches
        self.patch_size = patch_size
        self.stride = stride
        self.xpos = 0
        self.ypos = 0

        mat = scipy.io.loadmat(self.data_file)
        nbands = len(mat["SlectBands"])
        if patches:
            self.shape = (mat["nCol"][0][0].astype(np.int32), mat["nRow"][0][0].astype(np.int32), nbands)
            self.HCI = np.transpose(mat["Y"], (1, 0))
            self.HCI = self.HCI.reshape(self.shape).astype(np.float32)
            self.HCI = self.padImage(self.HCI)
        else:
            self.shape = (mat["nCol"][0][0].astype(np.int32)*mat["nRow"][0][0].astype(np.int32), mat["nBand"][0][0].astype(np.int32))
            self.HCI = np.transpose(mat["Y"], (1, 0)).astype(np.float32)

    def __len__(self):
        if self.patches:
            xsize = int((self.HCI.shape[1] - (self.patch_size[1] - 1) - 1) / self.stride) + 1
            ysize = int((self.HCI.shape[0] - (self.patch_size[0] - 1) - 1) / self.stride) + 1
            return xsize*ysize - 1
        else:
            return self.shape[0]

    def __getitem__(self, idx):

        if self.patches:
            sample = self.HCI[self.ypos:self.ypos+self.patch_size[0], self.xpos:self.xpos + self.patch_size[1], :]
            self.xpos += self.stride
            if self.xpos + self.patch_size[1] > self.HCI.shape[1]:
                self.xpos = 0
                self.ypos += self.stride
        else:
            sample = self.HCI[idx, :]

        if self.transform:
            sample = self.transform(sample)
            if self.patches:
                sample = sample.permute(2, 0, 1)

        return sample

    def padImage(self, HCI):
        # get padding if required
        if HCI.shape[1] // self.stride != HCI.shape[1] / self.stride:
            leftpad = (HCI.shape[1] // self.stride * self.stride + self.patch_size[1]) - HCI.shape[1]
            rightpad = leftpad // 2 if leftpad % 2 == 0 else leftpad // 2 + 1
            leftpad = leftpad // 2
        else:
            leftpad, rightpad = 0, 0

        if HCI.shape[0] // self.stride != HCI.shape[0] / self.stride:
            toppad = (HCI.shape[0] // self.stride * self.stride + self.patch_size[0]) - HCI.shape[0]
            bottompad = toppad // 2 if toppad % 2 == 0 else toppad // 2 + 1
            toppad = toppad // 2
        else:
            toppad, bottompad = 0, 0

        padding = ((toppad, bottompad), (leftpad, rightpad), (0, 0))
        HCI = np.pad(HCI, padding)

        return HCI


class SamsonDataset(Dataset):
    """https://rslab.ut.ac.ir/data"""
    def __init__(self, data_file, gt_file, transform=None):
        super(SamsonDataset, self).__init__()
        self.data_file = data_file
        self.gt_file = gt_file
        self.transform = transform

        mat = scipy.io.loadmat(self.data_file)
        self.shape = (mat["nCol"][0][0].astype(np.int32)*mat["nRow"][0][0].astype(np.int32), mat["nBand"][0][0])
        self.HCI = np.transpose(mat["V"], (1, 0))

        mat = scipy.io.loadmat(self.gt_file)

        # print(mat["cood"])# -> labels
        # ground truth abundances
        a = mat["A"][0, :].reshape((95, 95), order="F")
        b = mat["A"][1, :].reshape((95, 95), order="F")
        c = mat["A"][2, :].reshape((95, 95), order="F")
        self.gt_A = np.dstack((a, b, c))

        self.end_members = mat["M"]

    def __len__(self):
        return self.HCI.shape[0]

    def __getitem__(self, idx):

        sample = self.HCI[idx, :]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ArtificalDataset(Dataset):
    """docstring for ArtificalDataset"""
    def __init__(self, img_size, target_snr, transform=None):
        super(ArtificalDataset, self).__init__()
        self.snr = target_snr
        self.h = img_size[0]
        self.w = img_size[1]
        self.c = img_size[2]
        self.shape = [self.h*self.w, self.c]
        self.transform = transform

        self.images = self.create_data(img_size, target_snr)

    def create_data(self, img_size, target_snr):

        w, h, channels = img_size

        files = ["fad.csv", "free_nadh.csv", "cytc.csv", "porphyrin.csv"]

        nadh_spec = utils.read_fluro_spec(files[0], channels=channels)
        fad_spec = utils.read_fluro_spec(files[1], channels=channels)
        porphyrin_spec = utils.read_fluro_spec(files[2], channels=channels)
        cytc_spec = utils.read_fluro_spec(files[3], channels=channels)

        specs = [fad_spec, nadh_spec, cytc_spec, porphyrin_spec]

        abundance = artifical.create_abundance_map((w, h), "strips", ratios=[1., 1., 1., 1.])
        images, noises = artifical.create_images(abundance, specs, (w, h), target_snr=target_snr, channels=channels, show=False)
        return images

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, idx):

        sample = self.images[idx, :]

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):

    def __call__(self, sample):

        return torch.from_numpy(sample)


if __name__ == '__main__':
    tmp = ArtificalDataset((24, 25, 18), 20,
                           transform=transforms.Compose([ToTensor()]))
    loader = DataLoader(tmp)
    for i in loader:
        print(i.shape)
