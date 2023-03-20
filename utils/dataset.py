import torch
import numpy as np
import os

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(
    data_dir,
    batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the task 1 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.

    - shuffle: whether to shuffle the train/validation indices.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    train_file = torch.load(os.path.join(data_dir, "torch_train_dataset"))
    val_file = torch.load(os.path.join(data_dir, "torch_train_dataset"))

    train_loader = torch.utils.data.DataLoader(
        train_file,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_file,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return (train_loader, val_loader)


def get_test_loader(
    data_dir, batch_size, shuffle=True, num_workers=4, pin_memory=False
):
    """
    Utility function for loading and returning a multi-process
    test iterator over task 1 dataset.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    test_file = torch.load(os.path.join(data_dir, "torch_test_dataset"))
    data_loader = torch._utils.data.DataLoader(
        test_file,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return data_loader
