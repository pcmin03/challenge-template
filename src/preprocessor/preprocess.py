import torch
import torch.nn as nn
import numpy as np


class asl_preprocessor(nn.Module):
    def __init__(self, pre_cfg: dict = None):
        super().__init__()
        
        
        self.input_size = int(pre_cfg.INPUT_SIZE)
        self.n_cols = int(pre_cfg.N_COLS)
        self.lips_start = int(pre_cfg.LIPS_START)
        mean_std_meta = np.load(pre_cfg.MEAN_STD, allow_pickle=True).tolist()
        self.lips_mean = torch.tensor(mean_std_meta['lips_mean'])
        self.lips_std  = torch.tensor(mean_std_meta['lips_std'])
        self.lh_mean = torch.tensor(mean_std_meta['left_hands_mean'])
        self.lh_std = torch.tensor(mean_std_meta['left_hands_std'])
        self.rh_mean = torch.tensor(mean_std_meta['right_hands_mean'])
        self.rh_std = torch.tensor(mean_std_meta['right_hands_std'])
        self.pose_mean = torch.tensor(mean_std_meta['pose_mean'])
        self.pose_std = torch.tensor(mean_std_meta['pose_std'])
        
    def _preprocessing(self, x):

        x = x[:, :self.input_size, :self.n_cols, :2]
        ## lips
        lips = x[:,:self.input_size,self.lips_start:self.lips_start+40,:2]
        lips = torch.where(
                lips == 0.0,
                0.0,
                (lips - self.lips_mean.to(lips.device)) / self.lips_std.to(lips.device),
            )
        lips = torch.reshape(lips, [-1, self.input_size, 40*2])
        # LEFT HAND
        left_hand = x[:,:self.input_size,40:40+21,:2]
        left_hand = torch.where(
                left_hand == 0.0,
                0.0,
                (left_hand - self.lh_mean.to(left_hand.device)) / self.lh_std.to(left_hand.device),
            )
        left_hand = torch.reshape(left_hand, [-1, self.input_size, 21*2])
        # RIGHT HAND
        right_hand = x[:,:self.input_size,61:61+21,:2]
        right_hand = torch.where(
                right_hand == 0.0,
                0.0,
                (right_hand - self.rh_mean.to(right_hand.device)) / self.rh_std.to(right_hand.device),
            )
        right_hand = torch.reshape(right_hand, [-1, self.input_size, 21*2])
        # POSE
        pose = x[:,:self.input_size,82:82+10,:2]
        pose = torch.where(
                pose == 0.0,
                0.0,
                (pose - self.pose_mean.to(pose.device)) / self.pose_std.to(pose.device),
            )
        pose = torch.reshape(pose, [-1, self.input_size, 10*2])
        
        return (lips, left_hand, right_hand, pose)
        

    @torch.no_grad()
    def forward(self, x):
        x = self._preprocessing(x)
        return x