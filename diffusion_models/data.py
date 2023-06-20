import numpy as np
import torch
from synthetic_datasets import toy_2d
from torch.utils.data import Dataset


def get_train_set(dataset_name, n_samples=100000, slack=None):
    return toy_2d.get_dataset(dataset_name, n_samples=n_samples, seed=123)


def get_test_set(dataset_name, n_samples=1000, slack=None):
    return toy_2d.get_dataset(dataset_name, n_samples=n_samples, seed=456)


def get_datasets(data_config):
    if data_config.train_set_size is not None:
        train_set = get_train_set(data_config.dataset, n_samples=data_config.train_set_size)
    else:
        train_set = get_train_set(data_config.dataset)
    test_set = get_test_set(data_config.dataset, n_samples=1000)
    return train_set, test_set