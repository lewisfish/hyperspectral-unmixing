import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

import loss as ll
import layers as nl

__all__ = ["DAN"]


class DAN(pl.LightningModule):
    """docstring for DAN"""
    def __init__(self, input_size, end_members, noise, lr, weights, lossFN="sad"):
        super(DAN, self).__init__()
        self.input_size = input_size
        self.end_members = end_members
        self.weights = weights
        self.noise = noise
        self.lr = lr
        self.lossFN = lossFN

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 9*end_members),
            nn.LeakyReLU(),
            nn.Linear(9*end_members, 6*end_members),
            nn.LeakyReLU(),
            nn.Linear(6*end_members, 3*end_members),
            nn.LeakyReLU(),
            nn.Linear(3*end_members, end_members),
            nn.BatchNorm1d(end_members),
            nl.normIt(),
            nl.GaussianNoise(self.noise))

        self.decoder = nn.Sequential(
            nl.PosLinear(end_members, input_size, weights))

        self.save_hyperparameters("input_size", "end_members", "noise", "lr", "lossFN")
        if self.lossFN == "rmse":
            self.loss_func = ll.RMSELoss()
        elif self.lossFN == "sad":
            self.loss_func = ll.SADLoss()
        elif self.lossFN == "sid":
            self.loss_func = ll.SIDLoss()
        else:
            raise NotImplementedError

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

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/train", avg_loss, self.current_epoch)

    def test_custom(self, x):
        embed = self.encoder(x)
        return embed#torch.argmax(embed, dim=1)

    # def test_step(self, batch, batch_idx):
    #     x = batch

    #     embed = self.encoder(x.float())
    #     output = self.decoder(embed)
    #     end_members = self.decoder[0].weight
        # return torch.argmax(embed)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
