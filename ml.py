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
    def __init__(self, eps=1e-8):
        super(RMSELoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        """
        https://gist.github.com/jamesr2323/33c67ba5ac29880171b63d2c7f1acdc5
        """

        loss = torch.sqrt(F.mse_loss(x, y) + self.eps)
        return loss


class SADLoss(nn.Module):
    def __init__(self):
        super(SADLoss, self).__init__()
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-8)

    def forward(self, x, y):
        """
        spectral angle distance
        see https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html#torch.nn.CosineSimilarity

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
        https://douglasrizzo.com.br/kl-div-pytorch/
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


class PosLinear(nn.Module):
    def __init__(self, in_dim, out_dim, weights):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.transpose(weights, 1, 0))

    def forward(self, x):
        return torch.matmul(x, torch.abs(self.weight))


class DAN(pl.LightningModule):
    """docstring for DAN"""
    def __init__(self, input_size, end_members, weights, train=True):
        super(DAN, self).__init__()
        self.input_size = input_size
        self.end_members = end_members
        self.weights = weights

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 32*end_members),
            nn.ReLU(),
            nn.Linear(32*end_members, 16*end_members),
            nn.ReLU(),
            nn.Linear(16*end_members, 4*end_members),
            nn.ReLU(),
            nn.Linear(4*end_members, end_members),
            nn.BatchNorm1d(end_members),
            normIt())

        # def init_weights(m):
        #     # https://discuss.pytorch.org/t/initialize-nn-linear-with-specific-weights/29005/6
        #     # init weights with vca guess of end members
        #     if type(m) == nn.Linear and m.bias is None:
        #         with torch.no_grad():
        #             m.weight.copy_(self.weights)

        self.decoder = nn.Sequential(
            PosLinear(end_members, input_size, weights))
        # if train:
        #     self.decoder.apply(init_weights)

        self.save_hyperparameters("input_size", "end_members")
        # self.loss_func = RMSELoss()
        self.loss_func = SADLoss()
        # self.loss_func = SIDLoss()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        embedding = self.encoder(x)
        output = self.decoder(embedding)

        return embedding, output, self.decoder[0].weight

    def training_step(self, batch, batch_idx):
        x = batch
        x = x.float()
        z = self.encoder(x)
        x_hat = self.decoder(z)

        # loss = F.mse_loss(x, x_hat)
        loss = self.loss_func(x, x_hat)

        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


def cli_main():
    torch.autograd.set_detect_anomaly(True)
    pl.seed_everything(1234)
    tb_logger = pl_loggers.TensorBoardLogger('logs/')
    trainer = pl.Trainer(logger=tb_logger, gpus=1, auto_select_gpus=True, max_epochs=50)

    # shape = (610, 340, 103) 103==bands, 9 pixel types
    # dataset = customdata.PaviaUDataset("PaviaU.mat", transform=transforms.Compose([customdata.ToTensor()]))
    dataset = customdata.ArtificalDataset((128, 128, 18), 20,
                                          transform=transforms.Compose([customdata.ToTensor()]))
    train_loader = DataLoader(dataset, batch_size=256, num_workers=32, shuffle=True)

    # test_loader = DataLoader(dataset, batch_size=64, num_workers=32)
    val_loader = DataLoader(dataset, batch_size=600, num_workers=32)

    # ------------
    # model
    # ------------
    # data = dataset.HCI
    data = dataset.images
    data = np.transpose(data, (1, 0))
    weights, _, _ = vca.vca(data, 4)
    for i in range(4):
        weights[:, i] /= np.max(weights[:, i])
    model = DAN(dataset.shape[1], 4, torch.from_numpy(weights).float(), train=False)

    # ------------
    # training
    # ------------
    trainer.fit(model, train_loader, val_loader)
    model.eval()
    input_data = dataset[10000].unsqueeze(0).float()
    embed, out, endmem = model.forward(input_data)
    print(embed)

    plt.plot(torch.abs(out.detach().cpu()[0]))
    plt.plot(input_data[0], label="real")
    plt.legend()
    plt.show()
    for i in range(4):
        plt.plot(weights[:, i])
        plt.plot(endmem[i, :].detach().cpu().numpy())
    plt.show()
    stop


if __name__ == '__main__':
    plt.switch_backend('Qt5Agg')

    cli_main()
