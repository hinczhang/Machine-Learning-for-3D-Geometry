import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias_size ,bias=False):
        super(LocallyConnected2d, self).__init__()
        self.out_channels = out_channels
        self.bias_size = bias_size
        self.weight = nn.Parameter(
            torch.randn(1, out_channels,in_channels)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, bias_size)
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        out = torch.matmul(self.weight, x)
        out = torch.nn.functional.interpolate(out.unsqueeze(0), size = (self.out_channels, self.bias_size), mode = 'bicubic', align_corners=True)[0]
        if self.bias is not None:
            out += self.bias
        return out

