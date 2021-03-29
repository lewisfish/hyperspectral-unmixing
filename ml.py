from argparse import ArgumentParser

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
from torch import nn
import torch.nn.functional as F
from torch import linalg as LA
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

import artifical
import customdata
import hysime
import vca


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        eps = 1e-6
        criterion = nn.MSELoss(reduce="sum")
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss

class SADLoss(nn.Module):
    def __init__(self):
        super(SADLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, x, y):
        """
        cos = nn.CosineSimilarity(dim=1, eps=1e-08)
        output = cos(input1, input2)

        shape: [batch, bands]

        """

        batch_size = x.size()[0]
        divisor = self.cos(x, y)
        output = torch.sum(torch.acos(divisor)) / batch_size

        return output


class SIDLoss(nn.Module):
    def __init__(self):
        super(SIDLoss, self).__init__()

    def forward(self, x, y):
        """

        """
        a = F.kl_div(x.log_softmax(0), y.softmax(0), reduction="sum")  # input needs to be logged, target does not...
        b = F.kl_div(y.log_softmax(0), x.softmax(0), reduction="sum")
        output = (a + b) / x.shape[0]

        return output


class softReLu(nn.Module):

    def __init__(self, size_in):
        super().__init__()

        self.size_in = size_in
        alphas = torch.Tensor(size_in).to("cuda:0")
        self.alphas = nn.Parameter(alphas)
        self.alphas.requires_grad = True
        torch.nn.init.constant_(self.alphas, 1e-9)
        self.zeros = torch.zeros(size_in, dtype=torch.float32, device="cuda:0")

    def forward(self, x):
        return torch.maximum(self.zeros, x - self.alphas)


class normIt(nn.Module):
    """docstring for normIt"""
    def __init__(self):
        super(normIt, self).__init__()
        self.eps = 1e-8

    def forward(self, x):
        x = torch.abs(x) / (self.eps + torch.sum(torch.abs(x), dim=1, keepdims=True))
        return x


class LitAutoEncoder(pl.LightningModule):

    def __init__(self, input_size, end_members):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 9*end_members),
            nn.LeakyReLU(),
            nn.Linear(9*end_members, 6*end_members),
            nn.LeakyReLU(),
            nn.Linear(6*end_members, 3*end_members),
            nn.LeakyReLU(),
            nn.Linear(3*end_members, end_members),
            nn.LeakyReLU(),
            nn.BatchNorm1d(end_members),
            nn.LogSoftmax())

        self.decoder = nn.Sequential(
            nn.Linear(end_members, input_size)
        )

        self.RMSEloss = RMSELoss()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        return embedding

    def training_step(self, batch, batch_idx):
        x = batch
        x = x.float()
        z = self.encoder(x.float())
        x_hat = self.decoder(z)
        # print(z)
        # stop
        # loss = F.mse_loss(x_hat, x)
        loss = self.RMSEloss(x, x_hat)
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer


def cli_main():

    # tmp = torch.Tensor([[.5, .25, .1], [0., .75, .3]])
    # print(tmp.shape)
    # print("yo", tmp, torch.sum(tmp, axis=1))
    # for i in range(tmp.shape[0]):
    #     print(tmp[i] / torch.sum(tmp, axis=1)[i])
    # stop

    pl.seed_everything(1234)
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = pl.Trainer(logger=tb_logger, gpus=1, auto_select_gpus=True,max_epochs=100)
    # ------------
    # args
    # ------------
    # parser = ArgumentParser()
    # parser.add_argument('--batch_size', default=64, type=int)
    # parser.add_argument('--hidden_dim', type=int, default=128)
    # parser = trainer.add_argparse_args(parser)
    # args = parser.parse_args()

    # ------------
    # data
    # ------------
    # mnist_train, mnist_val = random_split(dataset, [55000, 5000])

    # dataset = customdata.ArtificalDataset((24, 25, 18), 20,
    #                                       transform=transforms.Compose([customdata.ToTensor()]))

    dataset = customdata.PaviaUDataset("PaviaU.mat")

    train_loader = DataLoader(dataset, batch_size=32, num_workers=32)
    # val_loader = DataLoader(mnist_val, batch_size=64, num_workers=32)
    test_loader = DataLoader(dataset, batch_size=64, num_workers=32)
    val_loader = DataLoader(dataset, batch_size=64, num_workers=32)

    # ------------
    # model
    # ------------
    model = LitAutoEncoder(103, 4)

    # ------------
    # training
    # ------------
    # trainer = trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)

    # ------------
    # testing
    # ------------
    result = trainer.test(test_dataloaders=test_loader)
    print(result)


if __name__ == '__main__':
    cli_main()
