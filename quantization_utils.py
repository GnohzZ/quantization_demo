# 用于量化权重和INT8激活
import torch
import torch.nn as nn

class FakeQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_range):
        ctx.save_for_backward(x, torch.tensor(-128 / x_range), torch.tensor(127 / x_range))
        x = x.mul(x_range).round().clamp(-128, 127).mul(1 / x_range)  # 量化反量化
        return x

    @staticmethod
    def backward(ctx, grad_output):
        x, x_min, x_max = ctx.saved_tensors
        zeros = torch.zeros_like(x)
        ones = torch.ones_like(x)
        mask0 = torch.where(x < x_min, zeros, ones)
        mask1 = torch.where(x > x_max, zeros, ones)
        mask = mask0 * mask1
        grad = grad_output * mask
        return grad, None


def cut_scale(x, bias_scale, next_scale):
    return (x * bias_scale / next_scale).round().clamp(-128, 127)


class WeightQuantizer(nn.Module):
    def __init__(self, x_range):
        super().__init__()
        self.x_range = x_range

    def forward(self, x):
        return FakeQuantize.apply(x, self.x_range)


class FakeQuantizeBias(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_range):
        ctx.save_for_backward(x)
        x = x.mul(x_range).round().mul(1 / x_range)  # 量化反量化
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class BiasQuantizer(nn.Module):
    def __init__(self, x_range):
        super().__init__()
        self.x_range = x_range

    def forward(self, x):
        return FakeQuantizeBias.apply(x, self.x_range)

def cut(x, in_cut_start: int = 0, en=True, bit_width=8):
    """
    result = (x >> in_cut_start).clip()
    type_out: 0-int32, 1-int8
    """
    if en:
        in_cut_start = in_cut_start
        qmax = 2 ** (bit_width - 1) - 1
        qmin = - 2 ** (bit_width - 1)
        return x.div(2 ** in_cut_start).round().clamp(min=qmin, max=qmax)
    else:
        return x