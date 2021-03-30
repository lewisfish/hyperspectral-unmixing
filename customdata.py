import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import artifical
import utils


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


class SamsonDataset(Dataset):
    """https://rslab.ut.ac.ir/data"""
    def __init__(self, data_file, gt_file, transform=None):
        super(SamsonDataset, self).__init__()
        self.data_file = data_file
        self.gt_file = gt_file
        self.transform = transform

        mat = scipy.io.loadmat(self.data_file)
        self.shape = (mat["nCol"][0][0].astype(np.float32)*mat["nRow"][0][0].astype(np.float32), mat["nBand"][0][0])
        self.HCI = np.transpose(mat["V"], (1, 0))

        mat = scipy.io.loadmat(self.gt_file)

        # print(mat["cood"])# -> labels
        # ground truth abundances
        a = mat["A"][0, :].reshape((95, 95), order="F")
        b = mat["A"][1, :].reshape((95, 95), order="F")
        c = mat["A"][2, :].reshape((95, 95), order="F")
        self.gt_A = np.dstack((a, b, c))

        self.end_members = mat["M"]

        # plt.imshow(np.sum(self.HCI, axis=1).reshape((self.shape[0], self.shape[1]), order="F"))

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
