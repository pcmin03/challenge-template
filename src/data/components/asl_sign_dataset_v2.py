from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import torch
class gen_dataset(Dataset):
    def __init__(self, X, y, NON_EMPTY_FRAME_IDXS, input_size, n_dims, n_cols, lips_idxs,
                 hands_idxs, left_hands_idxs, right_hands_idxs, pose_idxs):
        
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.non_empty_frame_idxs = NON_EMPTY_FRAME_IDXS.astype(np.float32)
        
        self.input_size = input_size
        self.n_dims = n_dims
        self.n_cols = n_cols
        self.lips_idxs = lips_idxs
        self.lips_start = 0
        self.hands_idxs = hands_idxs
        self.left_hands_idxs = left_hands_idxs
        self.left_hands_start = self.lips_idxs.size
        self.right_hands_idxs = right_hands_idxs
        self.right_hands_start = self.left_hands_start + self.left_hands_idxs.size
        self.pose_idxs = pose_idxs
        self.pose_start = self.right_hands_start + self.right_hands_idxs.size
        
        self.lips_mean, self.lips_std = self.extract_mean_std('lips')
        self.lh_mean, self.lh_std, self.rh_mean, self.rh_std = self.extract_mean_std('hands')
        self.pose_mean, self.pose_std = self.extract_mean_std('pose')
        
        
        
    def extract_mean_std(self, name):
        if name == 'lips':
            # LIPS
            LIPS_MEAN_X = np.zeros([self.lips_idxs.size], dtype=np.float32)
            LIPS_MEAN_Y = np.zeros([self.lips_idxs.size], dtype=np.float32)
            LIPS_STD_X = np.zeros([self.lips_idxs.size], dtype=np.float32)
            LIPS_STD_Y = np.zeros([self.lips_idxs.size], dtype=np.float32)
            
            for col, ll in enumerate(tqdm(np.transpose(self.X[:,:,self.lips_idxs], [2,3,0,1]).reshape([self.lips_idxs.size, self.n_dims, -1]))):
                for dim, l in enumerate(ll):
                    v = l[np.nonzero(l)]
                    if dim == 0: # X
                        LIPS_MEAN_X[col] = v.mean()
                        LIPS_STD_X[col] = v.std()
                    if dim == 1: # Y
                        LIPS_MEAN_Y[col] = v.mean()
                        LIPS_STD_Y[col] = v.std()
                        
            LIPS_MEAN = np.array([LIPS_MEAN_X, LIPS_MEAN_Y]).T
            LIPS_STD = np.array([LIPS_STD_X, LIPS_STD_Y]).T
            return LIPS_MEAN, LIPS_STD
        
        elif name == 'hands':
            # LEFT HAND
            LEFT_HANDS_MEAN_X = np.zeros([self.left_hands_idxs.size], dtype=np.float32)
            LEFT_HANDS_MEAN_Y = np.zeros([self.left_hands_idxs.size], dtype=np.float32)
            LEFT_HANDS_STD_X = np.zeros([self.left_hands_idxs.size], dtype=np.float32)
            LEFT_HANDS_STD_Y = np.zeros([self.left_hands_idxs.size], dtype=np.float32)
            # RIGHT HAND
            RIGHT_HANDS_MEAN_X = np.zeros([self.right_hands_idxs.size], dtype=np.float32)
            RIGHT_HANDS_MEAN_Y = np.zeros([self.right_hands_idxs.size], dtype=np.float32)
            RIGHT_HANDS_STD_X = np.zeros([self.right_hands_idxs.size], dtype=np.float32)
            RIGHT_HANDS_STD_Y = np.zeros([self.right_hands_idxs.size], dtype=np.float32)
            
            for col, ll in enumerate(tqdm( np.transpose(self.X[:,:,self.hands_idxs], [2,3,0,1]).reshape([self.hands_idxs.size, self.n_dims, -1]) )):
                for dim, l in enumerate(ll):
                    v = l[np.nonzero(l)]
                    if dim == 0: # X
                        if col < self.right_hands_idxs.size: # LEFT HAND
                            LEFT_HANDS_MEAN_X[col] = v.mean()
                            LEFT_HANDS_STD_X[col] = v.std()
                        else:
                            RIGHT_HANDS_MEAN_X[col - self.left_hands_idxs.size] = v.mean()
                            RIGHT_HANDS_STD_X[col - self.left_hands_idxs.size] = v.std()
                    if dim == 1: # Y
                        if col < self.right_hands_idxs.size: # LEFT HAND
                            LEFT_HANDS_MEAN_Y[col] = v.mean()
                            LEFT_HANDS_STD_Y[col] = v.std()
                        else: # RIGHT HAND
                            RIGHT_HANDS_MEAN_Y[col - self.left_hands_idxs.size] = v.mean()
                            RIGHT_HANDS_STD_Y[col - self.left_hands_idxs.size] = v.std()

            LEFT_HANDS_MEAN = np.array([LEFT_HANDS_MEAN_X, LEFT_HANDS_MEAN_Y]).T
            LEFT_HANDS_STD = np.array([LEFT_HANDS_STD_X, LEFT_HANDS_STD_Y]).T
            RIGHT_HANDS_MEAN = np.array([RIGHT_HANDS_MEAN_X, RIGHT_HANDS_MEAN_Y]).T
            RIGHT_HANDS_STD = np.array([RIGHT_HANDS_STD_X, RIGHT_HANDS_STD_Y]).T
            return LEFT_HANDS_MEAN, LEFT_HANDS_STD, RIGHT_HANDS_MEAN, RIGHT_HANDS_STD
        
        elif name == 'pose':
            # POSE
            POSE_MEAN_X = np.zeros([self.pose_idxs.size], dtype=np.float32)
            POSE_MEAN_Y = np.zeros([self.pose_idxs.size], dtype=np.float32)
            POSE_STD_X = np.zeros([self.pose_idxs.size], dtype=np.float32)
            POSE_STD_Y = np.zeros([self.pose_idxs.size], dtype=np.float32)
            
            for col, ll in enumerate(tqdm( np.transpose(self.X[:,:,self.pose_idxs], [2,3,0,1]).reshape([self.pose_idxs.size, self.n_dims, -1]) )):
                for dim, l in enumerate(ll):
                    v = l[np.nonzero(l)]
                    if dim == 0: # X
                        POSE_MEAN_X[col] = v.mean()
                        POSE_STD_X[col] = v.std()
                    if dim == 1: # Y
                        POSE_MEAN_Y[col] = v.mean()
                        POSE_STD_Y[col] = v.std()
            
            POSE_MEAN = np.array([POSE_MEAN_X, POSE_MEAN_Y]).T
            POSE_STD = np.array([POSE_STD_X, POSE_STD_Y]).T
            
            return POSE_MEAN, POSE_STD
                        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
       
        X = self.X[idx]
        y = self.y[idx]
        non_empty_frame_idxs = self.non_empty_frame_idxs[idx]
    
        data = {}
        data['X'] = X
        data['label'] = y
        data['non_empty_frame_idxs'] = non_empty_frame_idxs
        return data