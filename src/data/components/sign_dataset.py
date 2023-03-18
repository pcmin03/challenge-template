from torch.utils.data import Dataset, sampler

class ASLDataFrameDataset(Dataset):
    def __init__(self, df, dataset, labels, transform=None):
        self.df = df
        self.X = dataset
        self.y = labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Use df_index as idx due to folds splitting
        df_index = self.df.index.values[idx]
        x = self.X[df_index]
        y = self.y[df_index]

        x = torch.Tensor(x)
        y = torch.Tensor([y]).long()

        if self.transform:
            x = self.transform(x)

        return x, y