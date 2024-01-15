from torch.utils.data import dataloader
from torch.utils.data import Dataset

class DIQADataset(Dataset):
    def __init__(self, X, y):
        pass

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


