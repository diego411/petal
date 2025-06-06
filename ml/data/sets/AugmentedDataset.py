from torch.utils.data import Dataset


class AugmentedDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)