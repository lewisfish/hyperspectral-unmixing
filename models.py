import math
import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init
import torchmetrics

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


class SCNN(pl.LightningModule):
    """docstring for SCNN"""
    def __init__(self, input_channels, n_classes, patch_size=64):
        super(SCNN, self).__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.patch_size = patch_size
        self.features_size = 512*13*13

        self.loss_func = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy()

        self.conv1 = nn.Sequential(
                     nn.Conv2d(1, 96, (6, 6), stride=(2, 2)),
                     nn.BatchNorm2d(96),
                     nn.LeakyReLU(),
                     nn.MaxPool2d((2, 2)))

        self.conv2 = nn.Sequential(
                     nn.Conv2d(96, 256, (3, 3), stride=(2, 2)),
                     nn.BatchNorm2d(256),
                     nn.LeakyReLU(),
                     nn.MaxPool2d((2, 2)))

        self.conv3 = nn.Sequential(
                     nn.Conv2d(256, 512, (3, 3), stride=(1, 1)),
                     nn.LeakyReLU())

        self.fc = nn.Sequential(
                    nn.Linear(self.features_size, 1024),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(1024, self.n_classes))

        self.apply(self.weight_init)

    @staticmethod
    def weight_init(m):
        if isinstance(m, (nn.Linear, nn.Conv3d)):
            init.kaiming_normal_(m.weight)
            init.zeros_(m.bias)

    def training_step(self, batch, batch_idx):

        loss, preds = self.shared_step(batch)
        preds = torch.nn.functional.softmax(preds, 1)
        self.log("loss", loss)
        _, y = batch
        self.log("train_acc_step", self.accuracy(preds, y))

        return loss

    def validation_step(self, batch, batch_idx):

        loss, preds = self.shared_step(batch)
        self.log("val_loss", loss)
        _, y = batch
        preds = torch.nn.functional.softmax(preds, 1)
        self.log("val_acc_step", self.accuracy(preds, y))
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch)
        return loss

    def shared_step(self, batch):
        x, target = batch
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.shape)
        x = x.view(-1, self.features_size)
        x_hat = self.fc(x)

        loss = self.loss_func(x_hat, target)
        return loss, x_hat

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("train_acc_epoch", self.accuracy.compute(), self.current_epoch)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.9, weight_decay=0.0005)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30 // 2, (5 * 30) // 6], gamma=0.1)
        # return [optimizer], [scheduler]
