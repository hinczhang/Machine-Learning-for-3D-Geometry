import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

class LocallyConnected2d(nn.Module):
    def __init__(self, batch_size, in_channels, out_channels, bias_size ,bias=False):
        super(LocallyConnected2d, self).__init__()
        self.out_channels = out_channels
        self.bias_size = bias_size
        self.batch_size = batch_size
        self.convlayers = []
        self.normlayers = []
        #print("H1:", in_channels//self.batch_size)
        for i in range(batch_size):
            self.convlayers.append(nn.Conv2d(1,1,(3,5),1,(1,2)).cuda())
            self.normlayers.append(nn.Sequential(nn.BatchNorm1d(in_channels//self.batch_size).cuda(), nn.ReLU().cuda()))
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, bias_size)
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        # out = torch.matmul(self.weight, x)
        height = x.shape[0]//self.batch_size
        #print("HH2: ", height)
        out = torch.zeros(x.shape).cuda()
        for i in range(self.batch_size):
            row = self.convlayers[i](x[height*i:height*(i+1),:].unsqueeze(0).unsqueeze(0))[0]
            row = self.normlayers[i](row)[0]
            out[height*i:height*(i+1),:] = row
        out = torch.nn.functional.interpolate(out.unsqueeze(0).unsqueeze(0), size = (self.out_channels, self.bias_size), mode = 'bicubic', align_corners=True)[0]
        if self.bias is not None:
            out += self.bias
        return out

