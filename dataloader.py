from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from numpy import genfromtxt

torch.set_default_dtype(torch.float64)


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


def get_train_val_dataloader(file_path):
    df = pd.read_csv(file_path, header=None)
    train = df.sample(frac=0.8, random_state=200)
    val = df.drop(train.index)
    return DataLoader(DatasetConverted(train), batch_size=8, shuffle=True), DataLoader(DatasetConverted(val),
                                                                                       batch_size=8, shuffle=True)


class SeqDatasetConverter(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        song_data = genfromtxt(self.data_list.iloc[index, 0], delimiter=',', dtype=float)
        # song_data = torch.from_numpy(song_data)
        return song_data[:, :-1], torch.from_numpy(song_data[:, -1]).long()


def get_train_val_seq_dataloader(file_path):
    df = pd.read_csv(file_path, header=None)
    train = df.sample(frac=0.8, random_state=200)
    val = df.drop(train.index)
    return DataLoader(SeqDatasetConverter(train), batch_size=4, shuffle=True), DataLoader(SeqDatasetConverter(val),
                                                                                          batch_size=1, shuffle=True)
