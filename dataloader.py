import sklearn
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


def get_test_rf_dataloader(file_path):
    df = pd.read_csv(file_path, header=None, sep=' ')
    val_ds_list = []
    for ds_path in df.iterrows():
        val_ds_list.append(RFDataset(pd.read_csv(ds_path[1][0])))
    return DataLoader(ConcatDataset(val_ds_list), batch_size=1000, shuffle=True)


class SeqDatasetConverter(Dataset):
    def __init__(self, data_list, y_ind):
        self.data_list = data_list
        self.y_ind = y_ind

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        song_data = genfromtxt(self.data_list.iloc[index, 0], delimiter=',', dtype=float)
        # chords_nums - [root, MirexMajMin, maj/min, MirexMajMinBass, 3/5 bass, MirexSevenths, maj/min/7,
        # MirexSeventhsBass, 3/5/7 bass]

        y = torch.from_numpy(song_data[:, self.y_ind]).long()

        return torch.from_numpy(song_data[:, :-9]), y


def collate_fn(data):
    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, gt_seqs = zip(*data)

    lengths = [len(seq) for seq in src_seqs]
    # pad src
    padded_seqs = torch.full((len(src_seqs), max(lengths), src_seqs[0].shape[1]), -1)
    for i, seq in enumerate(src_seqs):
        padded_seqs[i, :lengths[i]] = seq
    src_seqs = padded_seqs.double()

    #pad gt
    padded_seqs = torch.full((len(gt_seqs), max(lengths)), -1)
    for i, seq in enumerate(gt_seqs):
        padded_seqs[i, :lengths[i]] = seq
    gt_seqs = padded_seqs.long()

    return src_seqs, gt_seqs, lengths


def get_train_val_seq_dataloader(file_path, batch_size, y_ind):
    df = pd.read_csv(file_path, header=None, sep=' ')
    train = df.sample(frac=0.8, random_state=200)
    val = df.drop(train.index)
    return DataLoader(SeqDatasetConverter(train, y_ind), batch_size=batch_size, shuffle=True,
                      num_workers=4,
                      collate_fn=collate_fn), DataLoader(
        SeqDatasetConverter(val, y_ind),
        batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)


def get_test_seq_dataloader(file_path, batch_size=4):
    df = pd.read_csv(file_path, header=None, sep=' ')
    return DataLoader(SeqDatasetConverter(df), batch_size=batch_size, shuffle=True, num_workers=4)
