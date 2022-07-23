import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import sys
from SoftPool import soft_pool2d, SoftPool2d
from LocallyConnected2d import LocallyConnected2d
sys.path.append("./expansion_penalty/")
import expansion_penalty_module as expansion
sys.path.append("./MDS/")
import MDS_module

class STN3d(nn.Module):
    def __init__(self, num_points = 2500):
        super(STN3d, self).__init__()
        self.num_points = num_points
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class PseudoSPNet(nn.Module):
    def __init__(self, num_points = 8192, n_primitives = 16):
        super(PseudoSPNet, self).__init__()

        self.num_points = num_points
        self.n_primitives = n_primitives

        self.stn = STN3d(num_points = num_points)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.bn5 = torch.nn.BatchNorm1d(512)
        self.bn6 = torch.nn.BatchNorm1d(1024)
        self.pool = SoftPool2d(kernel_size=(1,1), stride=(4,4))
        self.localConv = LocallyConnected2d(32,64,5000,True)

        self.deconv1 = torch.nn.ConvTranspose1d(64, 128, 1)
        self.deconv2 = torch.nn.ConvTranspose1d(3+128, 512, 1)
        self.deconv3 = torch.nn.ConvTranspose1d(512, 1024, 1)
        self.expansion = expansion.expansionPenaltyModule()
    
    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        R = x.transpose(2,1)
        R = torch.bmm(R, trans)
        R = x.transpose(2,1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x, _ = torch.sort(x, descending=True, dim = 1)
        trucated_size = int(x.shape[1]//x.shape[0])
        t_x = x[0,:trucated_size,:]
        for pts in x[1:]:
            t_x = torch.cat((t_x, pts[:trucated_size]))
        x = t_x
        x = self.localConv(x)
        x = x.transpose(2,0)
        x = torch.nn.functional.interpolate(x,size=batchsize)
        x = x.transpose(2,0)
        #print('after pool: ', x.shape)
        
        x = self.bn3(x)
        x = self.bn4(self.deconv1(x))
        x = torch.cat((x, R.transpose(2,1)), 1)
        x = self.bn5(self.deconv2(x))
        
        _, _, mean_mst_dis = self.expansion(x, self.num_points//self.n_primitives, 1.5)
        resampled_idx = MDS_module.minimum_density_sample(x.transpose(1,2).contiguous(), x.transpose(1, 2).contiguous().shape[1], mean_mst_dis) 
        x = MDS_module.gather_operation(x, resampled_idx) # (16,512,5000)
        x= self.bn6(self.deconv3(x))
        x,_ = torch.max(x, 2)
        
        x = x.view(-1, 1024)

        return x

