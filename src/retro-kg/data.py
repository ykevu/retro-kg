import numpy as np
from torch.utils import data


class TripleDataset(data.Dataset):
    def __init__(self, heads, relations, tails):
        self.heads = heads
        self.relations = relations
        self.tails = tails

    def __len__(self):
        return len(self.heads)

    def __getitem__(self, idx):
        return self.heads[idx], self.relations[idx], self.tails[idx]
