import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["RMSELoss", "sad_loss", "SADLoss", "SIDLoss"]


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


def sad_loss(x, y, dim=1, eps=1e-8):

    batch_size = x.size()[0]

    divisor = F.cosine_similarity(x, y, dim=dim, eps=eps)
    output = torch.sum(torch.acos(divisor)) / batch_size

    return output


class SADLoss(nn.Module):
    def __init__(self, dim=1, eps=1e-8):
        super(SADLoss, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x, y):
        """
        spectral angle distance
        see https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html#torch.nn.CosineSimilarity

        shape: [batch, bands]

        """

        output = sad_loss(x, y, self.dim, self.eps)

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
