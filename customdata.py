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
        shape = self.HCI.shape
        self.HCI = self.HCI.reshape((shape[0] * shape[1], shape[2]))
        self.HCI = self.HCI.astype(np.float32)

        self.transform = transform

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
        self.transform = transform

        self.specs, self.images = self.create_data(img_size, target_snr)

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

        return np.array(specs), images

    def __len__(self):
        return self.c

    def __getitem__(self, idx):

        image = self.images[:, idx]
        specs = self.specs

        sample = {"image": image, "specs": specs}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):

    def __call__(self, sample):

        spec, image = sample["specs"], sample["image"]

        return {"image": torch.from_numpy(image),
                "specs": torch.from_numpy(spec)}


if __name__ == '__main__':
    tmp = ArtificalDataset((24, 25, 18), 20,
                           transform=transforms.Compose([ToTensor()]))
