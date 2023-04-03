import torch
import torch.nn as nn


class PreprocessLayer(nn.Module):
    def __init__(self, INPUT_SIZE, HAND_IDXS0, LANDMARK_IDXS0, N_COLS, N_DIMS):
        super(PreprocessLayer, self).__init__()
        self.input_size = INPUT_SIZE
        self.hand_idx = HAND_IDXS0
        self.landmark_idx = LANDMARK_IDXS0
        self.n_col = N_COLS
        self.n_dim = N_DIMS
        
    def pad_edge(self, t, repeats, side):
        if side == 'LEFT':
            return torch.cat((t[:1].repeat(repeats, 1, 1), t), dim=0)
        elif side == 'RIGHT':
            return torch.cat((t, t[-1:].repeat(repeats, 1, 1)), dim=0)

    def forward(self, data0):
        
        data0 = torch.tensor(data0)
    
        # Number of Frames in Video
        N_FRAMES0 = data0.shape[0]
        # Filter Out Frames With Empty Hand Data
        frames_hands_nansum = torch.nanmean(torch.index_select(data0, 1, torch.tensor(self.hand_idx)), dim=[1, 2])
        # get the indices of non-empty frames
        non_empty_frames_idxs = torch.where(frames_hands_nansum > 0)[0]
        # select non-empty frames from data0
        data = torch.index_select(data0, 0, non_empty_frames_idxs)
        
        # Cast Indices in float32 to be compatible with Tensorflow Lite
        non_empty_frames_idxs = non_empty_frames_idxs.float()

        # Number of Frames in Filtered Video
        N_FRAMES = data.shape[0]

        # Gather Relevant Landmark Columns
        data = data[:, self.landmark_idx, :]

        # Video fits in INPUT_SIZE
        if N_FRAMES < self.input_size:
            # Pad With -1 to indicate padding
            non_empty_frames_idxs = nn.functional.pad(non_empty_frames_idxs, (0, self.input_size-N_FRAMES), value=-1)
            # Pad Data With Zeros
            data = nn.functional.pad(data, (0, 0, 0, 0, 0, self.input_size-N_FRAMES), value=0)
            # Fill NaN Values With 0
            data[torch.isnan(data)] = 0.0
            return data, non_empty_frames_idxs
        
        # Video needs to be downsampled to INPUT_SIZE
        else:
            # Repeat
            if N_FRAMES < self.input_size**2:
                repeats = (self.input_size * self.input_size) // N_FRAMES0
                data = torch.repeat_interleave(data, repeats, dim=0)
                non_empty_frames_idxs = torch.repeat_interleave(non_empty_frames_idxs, repeats, dim=0)
            

            # Pad To Multiple Of Input Size
            pool_size = len(data) // self.input_size
            if len(data) % self.input_size > 0:
                pool_size += 1

            if pool_size == 1:
                pad_size = (pool_size * self.input_size) - len(data)
            else:
                pad_size = (pool_size * self.input_size) % len(data)
                
            # Pad Start/End with Start/End value
            pad_left = pad_size // 2 + self.input_size // 2
            pad_right = pad_size // 2 + self.input_size // 2
            if pad_size % 2 > 0:
                pad_right += 1

            # Pad By Concatenating Left/Right Edge Values
            data = self.pad_edge(data, pad_left, 'LEFT')
            data = self.pad_edge(data, pad_right, 'RIGHT')
            
            
            # Pad Non Empty Frame Indices
            non_empty_frames_idxs = torch.cat((non_empty_frames_idxs[:1].repeat(pad_left), non_empty_frames_idxs), dim=0)
            non_empty_frames_idxs = torch.cat((non_empty_frames_idxs, non_empty_frames_idxs[-1:].repeat(pad_right)), dim=0)
        
        
            # Reshape to Mean Pool
            data = data.view(self.input_size, -1, self.n_col, self.n_dim)
            non_empty_frames_idxs = non_empty_frames_idxs.view(self.input_size, -1)
            
            # Mean Pool
            data = torch.nanmean(data, dim=1)
            non_empty_frames_idxs = torch.nanmean(non_empty_frames_idxs, dim=1)
            
            data[torch.isnan(data)] = 0.0
            
            return data, non_empty_frames_idxs
            

