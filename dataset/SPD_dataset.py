from importlib_metadata import distribution
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
import pandas as pd

import os
import copy

# For Consistency
dataset_dir = os.path.join(os.path.dirname(os.path.realpath(
    __file__)), 'SPD/healthcare-dataset-stroke-data.csv')


# Stroke Prediction Dataset
class SPDDataset(Dataset):
    def __init__(
        self,
        dataset_dir=dataset_dir,
        config={},
    ) -> None:
        self.dataset = pd.read_csv(dataset_dir).dropna()
        self.feature_to_loc = {c: i for i,
                               c in enumerate(self.dataset.columns)}
        self.dataset = self.dataset[self.dataset['gender'] != 'Other']
        for header_str in ["gender", "Residence_type", "ever_married", "work_type", "smoking_status"]:
            self.dataset[header_str] = self.dataset[header_str].astype(
                'category').cat.codes
        self.dataset = self.dataset.to_numpy()
        self.dataset = np.array(self.dataset, dtype=float)

        train_size = int(len(self.dataset) *
                         config.train_val_split)
        old_val_size = len(self.dataset) - train_size
        new_val_size = int(old_val_size * config.val_test_split)
        test_size = old_val_size - new_val_size

        train_dataset, val_test_dataset = torch.utils.data.random_split(
            self.dataset, [train_size, old_val_size])
        val_dataset, test_dataset = torch.utils.data.random_split(
            val_test_dataset, [new_val_size, test_size])
        self.train_dataset = np.array(train_dataset)
        self.val_dataset = np.array(val_dataset)
        self.test_dataset = np.array(test_dataset)


class SPDTrainDataset(SPDDataset):
    def __init__(
        self,
        dataset_dir=dataset_dir,
        config={}
    ) -> None:
        super().__init__(dataset_dir, config)
        self.dataset = self.train_dataset
        self.dataset = self.redistribute_dataset(
            self.train_dataset,
            config.protected_feature,
            config.feature_distribution,
            config.equalize_dataset
        )

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

    def redistribute_dataset(self, dataset, feature_name, feature_distribution, equalize_dataset):
        feature_index = self.feature_to_loc[feature_name]
        total = len(dataset)
        _, counts = np.unique(
            np.array(dataset[:, feature_index]), return_counts=True)

        assert len(counts) == len(feature_distribution)
        if equalize_dataset:
            total = np.min(counts)*len(counts)

        feature_distribution = np.array(
            feature_distribution) / sum(feature_distribution)
        intended_counts = total * feature_distribution
        feature_masks = [np.ma.masked_less(np.arange(
            count), intended_counts[i], copy=True).mask for i, count in enumerate(counts)]

        new_datasets = []

        for l in range(len(counts)):
            dataset_data = copy.deepcopy(dataset)
            new_dataset = dataset_data[dataset[:, feature_index] == l]
            if feature_masks[l].shape == ():
                continue

            new_dataset = new_dataset[feature_masks[l]]
            new_datasets.append(new_dataset)

        return torch.tensor(np.concatenate(new_datasets, axis=0))


def get_spd_datasets(dataset_dir=dataset_dir, config={}, verbose=False):
    datasets = SPDDataset(dataset_dir, config)
    train_dataset, val_dataset, test_dataset = datasets.train_dataset, datasets.val_dataset, datasets.test_dataset
    if verbose:
        print("Data Shape: train_size, data_fields")
        print("  ", train_dataset.shape)
        print("\""*40)
        print("Train Size: ", len(train_dataset))
        print("Val Size:   ", len(val_dataset))
        print("Test Size:  ", len(test_dataset))
    return train_dataset, val_dataset, test_dataset

# Download and Test Dataset
if __name__ == "__main__":
    get_spd_datasets(verbose=True)
