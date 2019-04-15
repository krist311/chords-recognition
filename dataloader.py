from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pandas as pd
from numpy import genfromtxt

torch.set_default_dtype(torch.float64)


class RFDataset(Dataset):
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


def get_train_val_rf_dataloader(file_path):
    df = pd.read_csv(file_path, header=None, sep=' ')
    train_list_df = df.sample(frac=0.8, random_state=200)
    val_list_df = df.drop(train_list_df.index)
    train_ds_list, val_ds_list = [], []
    for ds_path in train_list_df.iterrows():
        train_ds_list.append(RFDataset(pd.read_csv(ds_path[1][0])))
    for ds_path in val_list_df.iterrows():
        val_ds_list.append(RFDataset(pd.read_csv(ds_path[1][0])))
    return DataLoader(ConcatDataset(train_ds_list), batch_size=1000, shuffle=True), DataLoader(
        ConcatDataset(val_ds_list),
        batch_size=1000, shuffle=True)


def t_rf_dataloader(file_path):
    df = pd.read_csv(file_path, header=None, sep=' ')
    val_ds_list = []
    for ds_path in df.iterrows():
        val_ds_list.append(RFDataset(pd.read_csv(ds_path[1][0])))
    return DataLoader(ConcatDataset(val_ds_list), batch_size=1000, shuffle=True)


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
    df = pd.read_csv(file_path, header=None, sep=' ')
    train = df.sample(frac=0.8, random_state=200)
    val = df.drop(train.index)
    return DataLoader(SeqDatasetConverter(train), batch_size=4, shuffle=True), DataLoader(SeqDatasetConverter(val),
                                                                                          batch_size=1, shuffle=True)
