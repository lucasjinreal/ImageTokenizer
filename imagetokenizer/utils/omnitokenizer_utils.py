import torch
import torch.nn.functional as F
import torch.nn as nn


def Normalize(in_channels, norm_type="group"):
    assert norm_type in ["group", "batch"]
    if norm_type == "group":
        return torch.nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
    elif norm_type == "batch":
        return torch.nn.SyncBatchNorm(in_channels)


def logits_laplace(x, x_recons, logit_laplace_eps=0.1):
    # [-0.5, 0.5] -> [0, 1]
    x += 0.5
    x_recons += 0.5
    # [0, 1] -> [eps, 1-eps]
    x_laplace = (1 - 2 * logit_laplace_eps) * x + logit_laplace_eps
    x_recons_laplace = (1 - 2 * logit_laplace_eps) * x_recons + logit_laplace_eps
    return F.l1_loss(x_laplace, x_recons_laplace)


def divisible_by(numer, denom):
    return (numer % denom) == 0


def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


def silu(x):
    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return silu(x)
