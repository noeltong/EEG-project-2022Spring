import torch
from torch.utils.data import Dataset
import pandas as pd

class eegDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        with open(txt_path, 'r') as f:
            csv_paths = []
            for line in f:
                line = line.strip('\n')
                line = line.rstrip()
                words = line.split('\t')

                csv_paths.append([words[0], int(words[1])])

        self.csvs = csv_paths
        self.transform = transform

    def __getitem__(self, index):
        path, label = self.csvs[index]
        csv = self.csv_to_tensor(path)

        return csv, label

    def __len__(self):
        return len(self.csvs)

    def csv_to_tensor(self, path):
        return torch.tensor(pd.read_csv(path, header=None).values.T)

def create_datasets(train, val, test):
    return eegDataset(train), eegDataset(val), eegDataset(test)