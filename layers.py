import torch
import torch.nn.functional as F
from torch import nn


class squeeze(nn.Module):
    """docstring for squeeze"""
    def __init__(self):
        super(squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze(-1)


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
    """docstring fir PosLinear"""
    def __init__(self, in_dim, out_dim, weights):
        super(PosLinear, self).__init__()
        self.weight = nn.Parameter(torch.transpose(weights, 1, 0))

    def forward(self, x):
        return torch.matmul(x, torch.abs(self.weight))


class GaussianNoise(nn.Module):
    """docstring for GaussianNoise
    https://discuss.pytorch.org/t/writing-a-simple-gaussian-noise-layer-in-pytorch/4694"""
    def __init__(self, sigma, is_relative_detach=True):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma
        self.noise = torch.tensor(0).to("cuda:0")
        self.is_relative_detach = is_relative_detach

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.expand(*x.size()).float().normal_() * scale
            x = x + sampled_noise
        return x
