

class ASLDataNPYDataset(Dataset):
    def __init__(self, X, y, NON_EMPTY_FRAME_IDXS):
        self.X = X
        self.y = y
        self.non_empty_frame_idxs = NON_EMPTY_FRAME_IDXS
        
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.non_empty_frame_idxs[idx]
    