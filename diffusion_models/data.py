import numpy as np
import torch
#from synthetic_datasets import toy_2d
from torch.utils.data import Dataset
from common_utils.random import RNG
import sklearn
import sklearn.datasets

## Manually add dataset to get over arg error
class ToyDatasetBase(Dataset):
    def __init__(self, *args, **kwargs):
        self.setup_data(*args, **kwargs)

    def setup_data(self, *args, **kwargs):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

class Checkerboard(ToyDatasetBase):
    def setup_data(self, n_samples):
        x1 = np.random.rand(n_samples) * 4 - 2
        x2_ = np.random.rand(n_samples) - np.random.randint(0, 2, n_samples) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1) / 2
        self.data = data

class SwissRoll(ToyDatasetBase):
    def setup_data(self, n_samples, noise=1.0):
        data = sklearn.datasets.make_swiss_roll(n_samples=n_samples, noise=noise)[0]
        data = data.astype("float64")[:, [0, 2]]
        data /= 20  # Makes the data fit in the unit square
        self.data = data

def get_train_set(dataset_name, n_samples=100000, slack=None):
    with RNG(456):
        return Checkerboard(n_samples=n_samples)


def get_test_set(dataset_name, n_samples=1000, slack=None):
    with RNG(123):
        return Checkerboard(n_samples=n_samples)


def get_datasets(data_config):
    if data_config.train_set_size is not None:
        train_set = get_train_set(data_config.dataset, n_samples=data_config.train_set_size)
    else:
        train_set = get_train_set(data_config.dataset)
    test_set = get_test_set(data_config.dataset, n_samples=1000)
    return train_set, test_set