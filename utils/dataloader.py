from torch.utils.data import DataLoader, Dataset
import pandas as pd


class DatasetConverted(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        cqt = self.data.iloc[index, :-1]
        label = self.data.iloc[index, -1]
        if self.transform is not None:
            cqt = self.transform(cqt)
        return cqt, label


def get_dataloader(file_path):
    return DataLoader(file_path, batch_size=8, shuffle=True)
