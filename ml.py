from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import artifical
import customdata
import hysime
import vca
import models


def cli_main():
    # torch.autograd.set_detect_anomaly(True)
    pl.seed_everything(1234)

    dataset = customdata.AusData("data/aus_data/D/D230201.mat", pad=False, normalise=True)
    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = int(0.2 * len(dataset))

    split_sum = train_size + val_size + test_size
    diff = len(dataset) - split_sum
    if diff > 0:
        while True:
            train_size += 1
            diff -= 1
            if diff == 0:
                break
            val_size += 1
            diff -= 1
            if diff == 0:
                break
            test_size += 1
            diff -= 1
            if diff == 0:
                break

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = pl.Trainer(logger=tb_logger, gpus=1, auto_select_gpus=True, max_epochs=50)
    dataset = customdata.SamsonDataset("sample_data/samson/samson_1.mat", "sample_data/samson/end3.mat", transform=transforms.Compose([customdata.ToTensor()]))

    # shape = (610, 340, 103) 103==bands, 9 pixel types
    # dataset = customdata.PaviaUDataset("sample_data/paviaU/PaviaU.mat", transform=transforms.Compose([customdata.ToTensor()]))
    # dataset = customdata.ArtificalDataset((128, 128, 18), 5,
    #                                       transform=transforms.Compose([customdata.ToTensor()]))
    train_loader = DataLoader(dataset, batch_size=256, num_workers=32, shuffle=True)

    # test_loader = DataLoader(dataset, batch_size=64, num_workers=32)
    val_loader = DataLoader(dataset, batch_size=600, num_workers=32)

    # ------------
    # model
    # ------------
    data = dataset.HCI
    # data = dataset.images
    # noises, cov = hysime.est_noise(data)
    # n_endmembers, b = hysime.hysime(data, noises, cov)
    data = np.transpose(data, (1, 0))

    weights, _, _ = vca.vca(data, 3)
    for i in range(3):
        weights[:, i] /= np.max(weights[:, i])
    model = models.DAN(dataset.shape[1], 3, noise=0.1, lr=1e-3, weights=torch.from_numpy(weights).float())

    # ------------
    # training
    # ------------
    trainer.fit(model, train_loader, val_loader)
    model.eval()
    img = np.array([])
    gt = []
    for batch in test_loader:
        pixels = model.test_custom(batch.float())
    fig, axs = plt.subplots(1, 3)
    axs = axs.ravel()
    pixels = pixels.detach().numpy().reshape((95, 95, 3), order="F")
    axs[0].imshow(pixels[:, :, ::-1])
    axs[1].imshow(dataset.gt_A)

    axs[2].plot(model.decoder[0].weight[0, :].detach().numpy(), color="blue")
    axs[2].plot(model.decoder[0].weight[1, :].detach().numpy(), color="red")
    axs[2].plot(model.decoder[0].weight[2, :].detach().numpy(), color="orange")
    axs[2].plot(dataset.end_members[:, 0], color="orange", marker="+")
    axs[2].plot(dataset.end_members[:, 1], color="red", marker="+")
    axs[2].plot(dataset.end_members[:, 2], color="blue", marker="+")

    plt.show()


def cnn_main():

    # torch.autograd.set_detect_anomaly(True)
    pl.seed_everything(1234)
    tb_logger = pl_loggers.TensorBoardLogger('cnn_logs/')
    trainer = pl.Trainer(logger=tb_logger, gpus=1, auto_select_gpus=True, max_epochs=30, gradient_clip_val=1.)

    folder = Path("data/aus_data/expanded/")
    dataset = customdata.AusDataImg(folder, normalise=True)

    train_size = int(0.6 * len(dataset))
    val_size = int(0.2 * len(dataset))
    test_size = int(0.2 * len(dataset))

    split_sum = train_size + val_size + test_size
    diff = len(dataset) - split_sum
    if diff > 0:
        while True:
            train_size += 1
            diff -= 1
            if diff == 0:
                break
            val_size += 1
            diff -= 1
            if diff == 0:
                break
            test_size += 1
            diff -= 1
            if diff == 0:
                break

    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_set, batch_size=576, num_workers=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=576, num_workers=32)
    test_loader = DataLoader(test_set, batch_size=576, num_workers=32)

    model = models.SCNN(70, 4)
    trainer.fit(model, train_loader, val_loader)

    # torch.set_grad_enabled(False)
    # model.eval()
    # trainer.test(model, test_set)


if __name__ == '__main__':
    plt.switch_backend('Qt5Agg')

    # cli_main()
    cnn_main()