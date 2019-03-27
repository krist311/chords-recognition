from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import pandas as pd


class DatasetConverted(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        cqt = self.data.iloc[index, :-1].values
        label = self.data.iloc[index, -1]
        if self.transform is not None:
            cqt = self.transform(cqt)
        return cqt, label


def get_dataloader(file_path):
    df = pd.read_csv(file_path,header = None)
    train = df.sample(frac=0.8, random_state=200)
    val = df.drop(train.index)
    return DataLoader(DatasetConverted(train), batch_size=8, shuffle=True), DataLoader(DatasetConverted(val), batch_size=8, shuffle=True)
