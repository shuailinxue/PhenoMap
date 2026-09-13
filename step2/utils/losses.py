import torch


def rbf_kernel(x, y, alphas=(0.1, 1.0, 10.0)):
    x_sq = torch.sum(x ** 2, dim=1, keepdim=True)
    y_sq = torch.sum(y ** 2, dim=1, keepdim=True)
    dist_sq = x_sq + y_sq.t() - 2.0 * torch.matmul(x, y.t())
    dist_sq = torch.clamp(dist_sq, min=0.0)
    kernel_val = torch.zeros_like(dist_sq)
    for alpha in alphas:
        kernel_val += torch.exp(-alpha * dist_sq)
    return kernel_val


def mmd_loss(x, y, alphas=(0.1, 1.0, 10.0)):
    xx_kernel = rbf_kernel(x, x, alphas)
    yy_kernel = rbf_kernel(y, y, alphas)
    xy_kernel = rbf_kernel(x, y, alphas)
    mmd2 = xx_kernel.mean() + yy_kernel.mean() - 2.0 * xy_kernel.mean()
    return torch.sqrt(torch.clamp(mmd2, min=1e-8))
