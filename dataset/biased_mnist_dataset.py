import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import os
import argparse
import numpy as np
import random


dataset_dir = os.path.join(os.path.dirname(os.path.realpath(
    __file__)), 'MNIST/')

class BiasedMNISTDataset(Dataset):

    def __init__(self, args, dataset_dir=dataset_dir):
        self.train_dataset = MNIST(root=dataset_dir, train=True, download=True)
        self.test_dataset = MNIST(root=dataset_dir, train=False, download=True)

        self.all_X_train = self.train_dataset.data.float()
        self.all_Y_train = self.train_dataset.targets
        self.all_X_test = self.test_dataset.data.float()
        self.all_Y_test = self.test_dataset.targets

        # process dataset to bias it by args.feature_distribution
        self.labeled_X_train = []
        self.labeled_Y_train = []
        self.unlabeled_X_train = []
        self.unlabeled_Y_train = []
        for label in range(10):
            indices = np.where(self.all_Y_train == label)[0]
            num_to_select = int(len(self.train_dataset) * args.feature_distribution[label] * args.dataset_split)
            if num_to_select == 0:
                continue
            # select num_to_select data points randomly
            selected_indices = np.random.choice(indices, num_to_select, replace=False)
            self.labeled_X_train.append(self.all_X_train[selected_indices])
            self.labeled_Y_train.append(self.all_Y_train[selected_indices])
            # add everything else to unlabeled dataset
            still_unlabeled = np.zeros(len(self.train_dataset), dtype=bool)
            still_unlabeled[indices] = True
            still_unlabeled[selected_indices] = False
            self.unlabeled_X_train.append(self.all_X_train[still_unlabeled])
            self.unlabeled_Y_train.append(self.all_Y_train[still_unlabeled])
        
        self.labeled_X_train = torch.cat(self.labeled_X_train, dim=0)
        self.labeled_Y_train = torch.cat(self.labeled_Y_train, dim=0)
        self.unlabeled_X_train = torch.cat(self.unlabeled_X_train, dim=0)
        self.unlabeled_Y_train = torch.cat(self.unlabeled_Y_train, dim=0)

        # randomly shuffle indices of labeled data
        shuffled_indices = torch.randperm(self.labeled_X_train.size()[0])
        self.labeled_X_train = self.labeled_X_train[shuffled_indices]
        self.labeled_Y_train = self.labeled_Y_train[shuffled_indices]

    def __getitem__(self, index):
        return self.labeled_train_split[index]
    
    def update(self, proposed_data_indices):
        new_data_points = torch.index_select(self.unlabeled_train_split, 0, proposed_data_indices)
        
        self.labeled_train_split = torch.tensor(np.concatenate([self.labeled_train_split, new_data_points], axis=0))
        self.labeled_train_split = self.labeled_train_split[torch.randperm(self.labeled_train_split.size()[0])]
        
        still_unlabeled = torch.ones(self.unlabeled_train_split.shape[0], dtype=bool)
        still_unlabeled[proposed_data_indices] = False
        self.unlabeled_train_split = self.unlabeled_train_split[still_unlabeled]

    def get_xy_split(self, split): # split: 'labeled', 'unlabeled', 'test'
        if split == 'labeled':
            return self.labeled_X_train, self.labeled_Y_train
        elif split == 'unlabeled':
            return self.unlabeled_X_train, self.unlabeled_Y_train
        elif split == 'test':
            return self.all_X_test, self.all_Y_test
        else: raise Exception("unknown split")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-distribution', nargs='+', default=[0.1]*10, help='redistributed partition')
    parser.add_argument('--dataset-split', type=float, default=0.1, help='labeled/unlabeled dataset split')
    args = parser.parse_args()
    args.feature_distribution = [float(x) for x in args.feature_distribution]
    bmd = BiasedMNISTDataset(args)
    X_train, Y_train = bmd.get_xy_split('labeled')