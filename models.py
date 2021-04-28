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

__all__ = ["DAN", "SCNN3D", "SCNN"]


# Wang et al 2019: Nonlinear Unmixing of Hyperspectral Data via Deep Autoencoder Networks
# Zhao et al 2019: Hyperspectral Unmixing via Deep Autoencoder Networks for a Generalized Linear-Mixture/Nonlinear-Fluctuation Model
# Palsson et al 2018: Hyperspectral Unmixing Using a Neural Network Autoencoder

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
        self.automatic_optimization = False

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

        opt = self.optimizers()
        opt.zero_grad()
        loss = self.loss_func(x, x_hat)
        # print(loss)
        self.log("loss", loss, prog_bar=True)
        self.manual_backward(loss)
        opt.step()
        # https://discuss.pytorch.org/t/proper-way-to-constraint-the-final-linear-layer-to-have-non-negative-weights/103498/4
        # with torch.no_grad():
        #     self.decoder[0].weight.copy_(self.decoder[0].weight.data.clamp(min=0))

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        x = x.float()
        z = self.encoder(x)
        x_hat = self.decoder(z)

        loss = self.loss_func(x, x_hat)

        self.log("val_loss", loss)
        return loss

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/train", avg_loss, self.current_epoch)

    def test_custom(self, x):
        embed = self.encoder(x)
        return embed#torch.argmax(embed, dim=1)

    def test_step(self, batch, batch_idx):
        x = batch
        x = x.float()
        pixels = self.encoder(x)
        fig, axs = plt.subplots(1, 2)
        axs = axs.ravel()
        pixels = pixels.detach().cpu().numpy().reshape((702, 632, 5), order="F")
        axs[0].imshow(pixels[:, :, :3])
        real = x.detach().cpu().numpy().reshape((702, 632, 70), order="F")
        axs[1].imshow(np.sum(real, axis=-1))

        plt.show()
        # return embed

        # return torch.argmax(embed)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class SCNN(pl.LightningModule):
    """docstring for SCNN"""
    def __init__(self, input_channels, n_classes):
        super(SCNN, self).__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.features_size = 256 * 15 * 15# 512*13*13

        self.loss_func = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        # self.auroc = torchmetrics.AUROC(num_classes=n_classes)

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
        self.log("train_acc_batch", self.train_accuracy(preds, y))

        return loss

    def validation_step(self, batch, batch_idx):

        loss, preds = self.shared_step(batch)
        self.log("val_loss", loss)
        _, y = batch
        preds = torch.nn.functional.softmax(preds, 1)
        _, preds_new = torch.max(preds, 1)
        correct = torch.sum(preds_new == y.data)

        self.log("val_acc_batch", self.val_accuracy(preds, y))
        return {"loss": loss}

    # def test_step(self, batch, batch_idx):
    #     loss, preds = self.shared_step(batch)
    #     _, y = batch
    #     preds = torch.nn.functional.softmax(preds, 1)
    #     _, preds_new = torch.max(preds, 1)
    #     correct = torch.sum(preds_new == y.data)

    #     return {"Test_correct": correct, "acc": self.accuracy(preds, y)}

    def shared_step(self, batch):
        x, target = batch
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        # print(x.shape)
        x = x.view(-1, self.features_size)
        x_hat = self.fc(x)

        loss = self.loss_func(x_hat, target)
        return loss, x_hat

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("train_acc_epoch", self.train_accuracy.compute(), self.current_epoch)

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        accuracy = self.val_accuracy.compute()

        # self.log("AUROC", self.auroc(preds, y))
        self.logger.experiment.add_scalar("val_loss_epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("val_acc_epoch", accuracy, self.current_epoch)

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        # return optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30 // 2, (5 * 30) // 6], gamma=0.1)
        return [optimizer], [scheduler]


class SCNN3D(pl.LightningModule):
    """docstring for SCNN3D"""
    def __init__(self, input_channels, n_classes):
        super(SCNN3D, self).__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.features_size = 144

        self.loss_func = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        self.conv1 = nn.Sequential(
                     nn.Conv3d(1, 4, (9, 3, 3), stride=(2, 2, 2)),
                     nn.BatchNorm3d(4),
                     nn.LeakyReLU(),
                     nn.MaxPool3d((2, 2, 2)),
                     nn.Dropout(p=0.25))

        self.conv2 = nn.Sequential(
                     nn.Conv3d(4, 8, (9, 3, 3), stride=(2, 2, 2)),
                     nn.BatchNorm3d(8),
                     nn.LeakyReLU(),
                     nn.MaxPool3d((2, 2, 2)))

        self.fc1 = nn.Sequential(
                    nn.Linear(self.features_size, 72),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(72, self.n_classes))

    def training_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch)
        preds = torch.nn.functional.softmax(preds, 1)
        self.log("loss", loss)
        _, y = batch
        self.log("train_acc_batch", self.train_accuracy(preds, y))

        return loss

    def validation_step(self, batch, batch_idx):

        loss, preds = self.shared_step(batch)
        self.log("val_loss", loss)
        _, y = batch
        preds = torch.nn.functional.softmax(preds, 1)
        _, preds_new = torch.max(preds, 1)
        correct = torch.sum(preds_new == y.data)

        self.log("val_acc_batch", self.val_accuracy(preds, y))
        return {"loss": loss}

    def shared_step(self, batch):
        x, target = batch
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.features_size)
        x_hat = self.fc1(x)
        loss = self.loss_func(x_hat, target)
        return loss, x_hat

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("train_acc_epoch", self.train_accuracy.compute(), self.current_epoch)

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        accuracy = self.val_accuracy.compute()

        # self.log("AUROC", self.auroc(preds, y))
        self.logger.experiment.add_scalar("val_loss_epoch", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("val_acc_epoch", accuracy, self.current_epoch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
        # return torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9, weight_decay=0.0005)

        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30 // 2, (5 * 30) // 6], gamma=0.1)

        # return [optimizer], [scheduler]


class SSNet(pl.LightningModule):
    """docstring for SSNet"""
    def __init__(self, input_channels, n_classes):
        super(SSNet, self).__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes

        self.features_size = 8000

        self.loss_func = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        self.conv1 = nn.Sequential(
                        nn.Conv3d(1, 8, (7, 3, 3), stride=1),
                        nn.LeakyReLU(),
                        nn.MaxPool3d(2))

        self.conv2 = nn.Sequential(
                        nn.Conv3d(8, 16, (5, 3, 3), stride=1),
                        nn.LeakyReLU(),
                        nn.MaxPool3d(2))

        self.conv3 = nn.Sequential(
                        nn.Conv3d(16, 32, (3, 3, 3), stride=1),
                        nn.LeakyReLU(),
                        nn.AdaptiveMaxPool3d((10, 5, 5)))

        # self.conv4 = nn.Sequential(
        #                 nn.Conv2d(32, 64, (3, 3), stride=1),
        #                 nn.LeakyReLU())

        self.fc1 = nn.Sequential(
                      nn.Linear(self.features_size, 256),
                      nn.LeakyReLU(),
                      nn.Dropout(0.5))

        self.fc2 = nn.Sequential(
                      nn.Linear(256, 128),
                      nn.LeakyReLU(),
                      nn.Dropout(0.5),
                      nn.Linear(128, self.n_classes))

    def training_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch)
        preds = torch.nn.functional.softmax(preds, 1)
        self.log("loss", loss)
        _, y = batch
        self.log("train_acc_batch", self.train_accuracy(preds, y))

        return loss

    def validation_step(self, batch, batch_idx):

        loss, preds = self.shared_step(batch)
        self.log("val_loss", loss)
        _, y = batch
        preds = torch.nn.functional.softmax(preds, 1)
        _, preds_new = torch.max(preds, 1)
        correct = torch.sum(preds_new == y.data)

        self.log("val_acc_batch", self.val_accuracy(preds, y))
        return {"loss": loss}

    def shared_step(self, batch):
        x, target = batch
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        # print(x.shape)

        x = x.view(-1, self.features_size)

        x = self.fc1(x)
        x_hat = self.fc2(x)

        loss = self.loss_func(x_hat, target)
        return loss, x_hat

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class HybridSN(pl.LightningModule):
    """docstring for HybridSN"""
    def __init__(self, input_channels, n_classes, input_bands, patch_size, lr, batch_size):
        super(HybridSN, self).__init__()
        self.input_channels = input_channels
        self.n_classes = n_classes
        self.input_bands = input_bands
        self.patch_size = patch_size
        self.lr = lr
        self.batch_size = batch_size

        self.example_input_array = torch.zeros((1, 1, self.input_bands, self.patch_size, self.patch_size))

        self.loss_func = nn.CrossEntropyLoss()
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()

        self.save_hyperparameters('patch_size', 'lr', 'batch_size', 'input_bands')

        self.conv1 = nn.Sequential(
                        nn.Conv3d(1, 8, (7, 3, 3), stride=1),
                        nn.ReLU(),
                        nn.MaxPool3d(2))

        self.conv2 = nn.Sequential(
                        nn.Conv3d(8, 16, (5, 3, 3), stride=1),
                        nn.ReLU(),
                        nn.MaxPool3d(2))

        self.conv3 = nn.Sequential(
                        nn.Conv3d(16, 32, (3, 3, 3), stride=1),
                        nn.ReLU())

        self.features_2D = self._get_feature_size()

        self.conv4 = nn.Sequential(
                        nn.Conv2d(self.features_2D, 64, (3, 3)),
                        nn.LeakyReLU())

        self.features_size = self._get_flatten_size()

        self.fc1 = nn.Sequential(
                      nn.Linear(self.features_size, 256),
                      nn.ReLU(),
                      nn.Dropout(0.4))

        self.fc2 = nn.Sequential(
                      nn.Linear(256, 128),
                      nn.ReLU(),
                      nn.Dropout(0.4),
                      nn.Linear(128, self.n_classes))

    def forward(self, batch):
        x = batch
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        x = self.conv4(x)
        x = x.view(-1, self.features_size)

        x = self.fc1(x)
        x_hat = self.fc2(x)
        return x_hat

    def training_step(self, batch, batch_idx):
        loss, preds = self.shared_step(batch)
        preds = torch.nn.functional.softmax(preds, 1)
        self.log("Loss/train", loss)
        _, y = batch
        self.log("Acc_batch/train", self.train_accuracy(preds, y))

        return loss

    def validation_step(self, batch, batch_idx):

        loss, preds = self.shared_step(batch)
        self.log("Loss/val", loss)
        _, y = batch
        preds = torch.nn.functional.softmax(preds, 1)
        _, preds_new = torch.max(preds, 1)
        correct = torch.sum(preds_new == y.data)

        self.log("Acc_batch/val", self.val_accuracy(preds, y))
        return {"loss": loss}

    def shared_step(self, batch):
        x, target = batch
        x = x.float()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
        x = self.conv4(x)
        x = x.view(-1, self.features_size)

        x = self.fc1(x)
        x_hat = self.fc2(x)

        loss = self.loss_func(x_hat, target)
        return loss, x_hat

    def on_train_start(self):
        hp_dict = {"hp/val_acc": 0.}
        self.logger.log_hyperparams(self.hparams, hp_dict)

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss_epoch/train", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Acc_epoch/train", self.train_accuracy.compute(), self.current_epoch)

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        accuracy = self.val_accuracy.compute()

        # self.log("AUROC", self.auroc(preds, y))
        self.logger.experiment.add_scalar("Loss_epoch/val", avg_loss, self.current_epoch)
        self.logger.experiment.add_scalar("Acc_epoch/val", accuracy, self.current_epoch)
        self.log("hp/val_acc", accuracy)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def _get_feature_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_bands, self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            return x.shape[1] * x.shape[2]

    def _get_flatten_size(self):
        with torch.no_grad():
            x = torch.zeros((1, 1, self.input_bands, self.patch_size, self.patch_size))
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            x = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4]))
            x = self.conv4(x)
            _, c, h, w = x.shape
            return c * h * w
