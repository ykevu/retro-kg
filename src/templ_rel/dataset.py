import misc
import numpy as np
from scipy import sparse
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple


def init_loader(args, dataset, batch_size: int, shuffle: bool = False):
    if args.local_rank != -1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=args.num_cores,
        pin_memory=True,
    )

    return loader


class FingerprintDataset(Dataset):
    """
    Dataset class for fingerprint representation of products
    for template relevance prediction
    """

    def __init__(self, fp_file: str, label_file: str):
        misc.log_rank_0(f"Loading pre-computed product fingerprints from {fp_file}")
        self.data = np.load(fp_file)

        misc.log_rank_0(f"Loading pre-computed target labels from {label_file}")
        self.labels = np.load(label_file)

        assert self.data.shape[0] == len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns tuple of product fingerprint, and label (template index)
        """
        fp = self.data[idx]
        label = self.labels[idx]

        return fp, label

    def __len__(self) -> int:
        return self.data.shape[0]
