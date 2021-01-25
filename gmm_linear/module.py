import math

import torch
from torch import nn
from torch.nn import functional


class GMMLinear(nn.Module):
    """
    A Pytorch implementation of the paper ``Processing of missing data by neural networks``.

    This module is capable of processing inputs containing NaN values
    by approximating them with a Gaussian Mixture Model (GMM).

    Parameters
    ----------
    in_features: int
        the number of input features
    out_features: int
        the number of output features
    n_modes: int
        the number of modes (gaussians) in the GMM
    bias: bool
        whether to use a bias term

    References
    ----------
    https://papers.nips.cc/paper/2018/file/411ae1bf081d1674ca6091f8c59a266f-Paper.pdf
    """

    def __init__(self, in_features: int, out_features: int, n_modes: int, bias: bool = True):
        super().__init__()
        if n_modes < 1:
            raise ValueError('GMM must have at least one mode, %d provided.' % n_modes)

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.mean = nn.Parameter(torch.zeros(n_modes, in_features, requires_grad=True), requires_grad=True)
        self.log_std2 = nn.Parameter(torch.zeros(n_modes, in_features, requires_grad=True), requires_grad=True)
        if n_modes == 1:
            self.log_weights = None
        else:
            self.log_weights = nn.Parameter(torch.randn(n_modes))

    @staticmethod
    def nr(x):
        # eq (2) from the paper
        pdf = 1 / math.sqrt(2 * math.pi) * torch.exp(-0.5 * x ** 2)
        return pdf + x * 0.5 * (1 + torch.erf(x * math.sqrt(0.5)))

    def forward(self, x: torch.Tensor):
        mask = torch.isnan(x)
        empty = mask.any(-1)
        # small optimization
        if not empty.any():
            return functional.relu(self.linear(x))

        mask = mask[..., None, :]

        # w @ m + b
        mean = torch.where(mask, self.mean, x[..., None, :])
        num = self.linear(mean).transpose(-1, -2)

        # w.T @ cov @ w (cov is diagonal)
        zero = torch.tensor(0).to(self.log_std2)
        std2 = torch.where(mask, torch.exp(self.log_std2), zero)
        den = (std2[..., None, :, :] * self.linear.weight[:, None] ** 2).sum(-1).sqrt()

        nr = den * self.nr(num / (den + 1e-9))
        if self.log_weights is None:
            # not a mixture in this case
            mixture = nr.squeeze(-1)
        else:
            mixture = nr @ functional.softmax(self.log_weights, 0)

        return torch.where(empty.unsqueeze(-1), mixture, functional.relu(num[..., 0]))
