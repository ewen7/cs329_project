import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np

import os
import copy

# For Consistency
dataset_dir = os.path.dirname(os.path.realpath(__file__))


class MINSTDataset(Dataset):
    def __init__(
        self,
        config={}
    ) -> None:
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomRotation(config.get('rotation_degree', 30)),
            transforms.Normalize((0.5), (0.5)) if config.get(
                'normalize_images', True) else None,
            transforms.RandomHorizontalFlip(
                config.get('horizontal_flip_prob', 0.5)),
            transforms.RandomVerticalFlip(
                config.get('vertical_flip_prob', 0.5)),
        ])


class MINSTTrainDataset(MINSTDataset):
    def __init__(
        self,
        dataset_dir=dataset_dir,
        config={}
    ) -> None:
        super().__init__(config)
        self.MINST_train_dataset = torchvision.datasets.MNIST(
            root=dataset_dir, transform=self.transforms, train=True, download=True)
        self.MINST_train_dataset = self.redistribute_dataset(
            self.MINST_train_dataset,
            config.get('label_distribution', [0.5]*10),
            config.get('equalize_dataset', True),
        )

    def redistribute_dataset(self, dataset, distribution, equalize_dataset):
        total = len(dataset)
        _, counts = np.unique(
            np.array(dataset.targets.tolist()), return_counts=True)

        if equalize_dataset:
            total = np.min(counts)*len(counts)

        distribution = np.array(distribution) / sum(distribution)
        intended_counts = total * distribution
        masks = [np.ma.masked_less(np.arange(
            count), intended_counts[i], copy=True).mask for i, count in enumerate(counts)]

        new_datasets = []
        new_dataset_targets = []

        for l in range(len(counts)):
            dataset_data = copy.deepcopy(dataset.data)
            new_dataset = dataset_data[dataset.targets == l]
            new_dataset = new_dataset[masks[l], :, :]
            new_datasets.append(new_dataset)

            dataset_targets = copy.deepcopy(dataset.targets)
            new_dataset_target = dataset_targets[dataset.targets == l]
            new_dataset_target = new_dataset_target[masks[l]]
            new_dataset_targets.append(new_dataset_target)

        self.MINST_train_dataset.targets = torch.tensor(
            np.concatenate(new_dataset_targets, axis=0))
        self.MINST_train_dataset.data = torch.tensor(
            np.concatenate(new_datasets, axis=0))
        return self.MINST_train_dataset

    def __getitem__(self, index):
        return self.MINST_train_dataset[index]

    def __len__(self):
        return len(self.MINST_train_dataset)


class MINSTTestDataset(MINSTDataset):
    def __init__(
        self,
        dataset_dir=dataset_dir,
        config={}
    ) -> None:
        super().__init__(config)
        self.MINST_test_dataset = torchvision.datasets.MNIST(
            root=dataset_dir, transform=self.transforms, train=False, download=True)

    def __getitem__(self, index):
        return self.MINST_test_dataset[index]

    def __len__(self):
        return len(self.MINST_test_dataset)


def get_minst_datasets(dataset_dir=dataset_dir, config={}, verbose=False):
    train_val_dataset = MINSTTrainDataset(dataset_dir, config)
    train_size = int(len(train_val_dataset) *
                     config.get('train_val_split', 0.8))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_val_dataset, [train_size, val_size])

    train_dataset = train_dataset.dataset
    val_dataset = val_dataset.dataset

    train_dataloader = DataLoader(train_dataset, batch_size=config.get(
        'batch_size', 64), shuffle=config.get('shuffle', True))
    val_dataloader = DataLoader(val_dataset, batch_size=config.get(
        'batch_size', 64), shuffle=config.get('shuffle', True))
    test_dataloader = DataLoader(MINSTTestDataset(dataset_dir, config), batch_size=config.get(
        'batch_size', 64), shuffle=config.get('shuffle', True))

    if verbose:
        img, label = next(iter(train_dataloader))
        print("Image Shape: batch_size, color, width, height")
        print("  ", img.shape)
        print("Label Shape: batch_size, ")
        print("  ", label.shape)
        print("\""*40)
        print("Train Size: ", len(train_dataset))
        print("Train Class Sizes: [0, ..., 9] ", [
              sum(train_dataset.MINST_train_dataset.targets == l) for l in range(10)])
        print("Val Size:   ", len(val_dataset))
        print("Test Size:  ", len(MINSTTestDataset(dataset_dir, config)))
    return train_dataloader, val_dataloader, test_dataloader


# Download and Test Dataset
if __name__ == "__main__":
    get_minst_datasets(verbose=True)
