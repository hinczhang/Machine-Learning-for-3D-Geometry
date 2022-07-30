import sys
import open3d as o3d
from model import *
from utils import *
import argparse
import random
import numpy as np
import torch
import os
sys.path.append("./emd/")
import emd_module as emd

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default = './trained_model/network.pth',  help='optional reload model path')
parser.add_argument('--num_points', type=int, default = 4096,  help='number of points')
parser.add_argument('--n_primitives', type=int, default = 16,  help='number of primitives in the atlas')

opt = parser.parse_args()
print (opt)

network = MSN(64, num_points = opt.num_points, n_primitives = opt.n_primitives, if_train = False) 
network.cuda()
network.apply(weights_init)


if opt.model != '':
    network.load_state_dict(torch.load(opt.model))
    print("Previous weight loaded ")

network.eval()
with open(os.path.join('./data/val.list')) as file:
    model_list = [line.strip().replace('/', '_') for line in file]

partial_dir = "./data/val/"
gt_dir = "./data/complete/" 

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

EMD = emd.emdModule()

labels_generated_points = torch.Tensor(range(1, (opt.n_primitives+1)*(opt.num_points//opt.n_primitives)+1)).view(opt.num_points//opt.n_primitives,(opt.n_primitives+1)).transpose(0,1)
labels_generated_points = (labels_generated_points)%(opt.n_primitives+1)
labels_generated_points = labels_generated_points.contiguous().view(-1)

with torch.no_grad():
    for i, model in enumerate(model_list[300:]):
        print(model)
        #partial = torch.zeros((50, 5000, 3), device='cuda')
        #gt = torch.zeros((50, opt.num_points, 3), device='cuda')
        partial = torch.zeros((50, 5000, 3), device='cuda')
        gt = torch.zeros((50, opt.num_points, 3), device='cuda')
        for j in range(50):
            pcd = o3d.io.read_point_cloud(os.path.join(partial_dir, model + '_' + str(j) + '_denoised.pcd'))
            partial[j, :, :] = torch.from_numpy(resample_pcd(np.array(pcd.points), 5000))
            pcd = o3d.io.read_point_cloud(os.path.join(gt_dir, model + '.pcd'))
            gt[j, :, :] = torch.from_numpy(resample_pcd(np.array(pcd.points), opt.num_points))
        
        output1, output2, expansion_penalty = network(partial.transpose(2,1).contiguous())
        
        pcd1 = o3d.geometry.PointCloud()
        res = output1.cpu().numpy().reshape(-1,3)
        max_c = res.max(axis=0)
        min_c = res.min(axis=0)
        pcd1.points = o3d.utility.Vector3dVector(output1.cpu().numpy().reshape(-1,3))
        pcd2 = o3d.geometry.PointCloud()
        out = output2.cpu().numpy().reshape(-1,3)
        new_out = out[0,:]
        for i in range(1, out.shape[0]):
            if(out[i,0]>max_c[0] or out[i,0]<min_c[0] or out[i,1]>max_c[1] or out[i,1]<min_c[1] or out[i,2]>max_c[2] or out[i,2]<min_c[2]):
                new_out = np.vstack((new_out, out[i,:]))
        pcd2.points = o3d.utility.Vector3dVector(new_out)
        pcd3 = o3d.geometry.PointCloud()
        pcd3.points = o3d.utility.Vector3dVector(gt.cpu().numpy().reshape(-1,3))
        o3d.io.write_point_cloud(os.path.join('/home/hinczhang/Projects/Machine-Learning-for-3D-Geometry/MSN','output1.pcd'), pcd1)
        o3d.io.write_point_cloud(os.path.join('/home/hinczhang/Projects/Machine-Learning-for-3D-Geometry/MSN','output2.pcd'), pcd2)
        o3d.io.write_point_cloud(os.path.join('/home/hinczhang/Projects/Machine-Learning-for-3D-Geometry/MSN','gt.pcd'), pcd3)
        
        output_int = torch.from_numpy(new_out)
        output_int = nn.functional.interpolate(output_int.unsqueeze(0).transpose(2,1), size = 64*4096).transpose(2,1)[0].reshape(64,4096,3)

        gt = nn.functional.interpolate(gt.transpose(0,2), size = 64).transpose(0,2)
        print(output1.shape, output_int.shape, gt.shape)
        dist, _ = EMD(output1, gt, 0.002, 10000)
        emd1 = torch.sqrt(dist).mean()
        dist, _ = EMD(output_int, gt, 0.002, 10000)
        emd2 = torch.sqrt(dist).mean()
        idx = random.randint(0, 49)

        print(' val [%d/%d]  emd1: %f emd2: %f expansion_penalty: %f' %(i + 1, len(model_list), emd1.item(), emd2.item(), expansion_penalty.mean().item()))
