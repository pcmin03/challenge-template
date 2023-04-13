import torch
import numpy as np 
import torch.nn.functional as F
from torch import nn 

def torch_nan_mean(x, axis=0):
    return torch.nansum(torch.where(torch.isnan(x), torch.zeros_like(x), x), dim=axis) / \
                torch.nansum(torch.where(torch.isnan(x), torch.zeros_like(x), torch.ones_like(x)), dim=axis)

def torch_nan_std(x, axis=0):
    d = x - torch_nan_mean(x, axis=axis)
    return torch.sqrt(torch_nan_mean(d * d, axis=axis))

def torch_flatten_means_and_stds(x, reshape_size, axis=0):
    # Get means and stds
    x_mean = torch_nan_mean(x, axis=axis)
    x_std  = torch_nan_std(x,  axis=axis)

    x_out = torch.cat([x_mean, x_std], 0)
    if x_out.shape[0] > 168:
        x_out = x_out[:168,:]
    x_out = x_out.reshape((1, reshape_size[1]*2))
    x_out = torch.where(torch.isfinite(x_out), x_out, torch.zeros_like(x_out))
    return x_out

class FeatureGen(nn.Module):
    def __init__(self, landmarks, point_landmarks, avg_set):
        super(FeatureGen, self).__init__()
        self.landmarks = landmarks
        self.point_landmarks = point_landmarks
        self.avg_set = avg_set

    def forward(self, x_in, num_frames, segments, input_shape):
#         print(right_hand_percentage(x))
        
        x_list = [torch.mean(x_in[:, av_set[0]:av_set[0]+av_set[1], :], dim=1, keepdim=True) for av_set in self.avg_set]
        x_list.append(torch.index_select(x_in, dim=1, index=torch.tensor(self.point_landmarks)))
        x = torch.cat(x_list, dim=1)
        x_padded = x.numpy()
        for i in range(segments):
            p0 = 1 if (x_padded.shape[0] % segments) > 0 and i % 2 != 0 else 0
            p1 = 1 if (x_padded.shape[0] % segments) > 0 and i % 2 == 0 else 0
            paddings = [(0, 0), (p0, p1), (0, 0)]
            x_padded = np.pad(x_padded, paddings, mode="symmetric")
        x_list = torch.tensor_split(torch.tensor(x_padded), segments)
        x_list = [torch_flatten_means_and_stds(_x, input_shape, axis=0) for _x in x_list]
        x_list.append(torch_flatten_means_and_stds(x, input_shape, axis=0))
        torch_x = torch.where(torch.isfinite(x), x, torch.mean(x[torch.isfinite(x)], dim=0, keepdim=True))
        x = x.permute(2, 0, 1)
        x = F.interpolate(x.unsqueeze(0), size=[num_frames, self.landmarks], mode='bilinear')
        x = x.squeeze(0).permute(1, 2, 0)
        x = torch.reshape(x, (1, input_shape[0]*input_shape[1]))
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x_list.append(x)
        x = torch.cat(x_list, dim=1)
        return x

class SimpleFeatureGen(nn.Module):
    def __init__(self,drop:bool=True):
        super(SimpleFeatureGen, self).__init__()

        lipsUpperOuter =  [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
        lipsLowerOuter = [146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
        lipsUpperInner = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
        lipsLowerInner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
        self.lips = lipsUpperOuter + lipsLowerOuter + lipsUpperInner + lipsLowerInner
        self.dim = 2 if drop else 3
    def forward(self, x):
        x = x[:,:,:self.dim]
        lips_x = x[:,self.lips,:].contiguous().view(-1, 43*self.dim)
        lefth_x = x[:,468:489,:].contiguous().view(-1, 21*self.dim)
        pose_x = x[:,489:522,:].contiguous().view(-1, 33*self.dim)
        righth_x = x[:,522:,:].contiguous().view(-1, 21*self.dim)
        
        lefth_x = lefth_x[~torch.any(torch.isnan(lefth_x), dim=1),:]
        righth_x = righth_x[~torch.any(torch.isnan(righth_x), dim=1),:]
        
        x1m = torch.mean(lips_x, 0)
        x2m = torch.mean(lefth_x, 0)
        x3m = torch.mean(pose_x, 0)
        x4m = torch.mean(righth_x, 0)

        x1s = torch.std(lips_x, 0)
        x2s = torch.std(lefth_x, 0)
        x3s = torch.std(pose_x, 0)
        x4s = torch.std(righth_x, 0)
        
        xfeat = torch.cat([x1m,x2m,x3m,x4m, x1s,x2s,x3s,x4s], axis=0)
        xfeat = torch.where(torch.isnan(xfeat), torch.tensor(0.0, dtype=torch.float32), xfeat)
        
        return xfeat
    

