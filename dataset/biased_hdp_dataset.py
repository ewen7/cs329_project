import argparse
from torch.utils.data import Dataset
import numpy as np
import copy
import torch
import os
import sys

sys.path.insert(0, "./")
from dataset.HDP_dataset import HDPDataset

dataset_dir = os.path.join(os.path.dirname(os.path.realpath(
    __file__)), 'HDP/heart_2020_cleaned.csv')

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

class BiasedHDPDataset(Dataset):

    # prep_dataset basically calls init
    def __init__(self, args, dataset_dir=dataset_dir):
        # load entire HDP dataset
        # run the logic of redistribute
        # args.percent_labelled
        datasets = HDPDataset(dataset_dir=dataset_dir, config=args)
        self.train_dataset, self.val_dataset, self.test_dataset = datasets.train_dataset, datasets.val_dataset, datasets.test_dataset

        self.balanced_train_dataset, self.leftover_train_dataset = self.redistribute_dataset(
            self.train_dataset,
            datasets.feature_to_loc,
            Namespace(dataset_split=1, protected_feature=args.feature_to_predict, feature_to_predict=args.feature_to_predict, feature_distribution= [0.5, 0.5], equalize_dataset=True)
        )   

        indexes = torch.randperm(self.balanced_train_dataset.shape[0])
        self.balanced_train_dataset = self.balanced_train_dataset[indexes]

        self.labeled_train_split, self.unlabeled_train_split = self.redistribute_dataset(
            self.balanced_train_dataset,
            datasets.feature_to_loc,
            args
        )
        self.test_dataset = torch.Tensor(self.test_dataset)
        self.feature_to_predict = datasets.feature_to_predict
        self.protected_feature = datasets.protected_feature if datasets.feature_to_predict >= datasets.protected_feature else datasets.protected_feature - 1

    def __getitem__(self, index):
        return self.labeled_train_split[index]

    def __len__(self):
        return len(self.labeled_train_split)

    def redistribute_dataset(
        self, 
        dataset, 
        feature_to_loc, 
        args):
        feature_index = feature_to_loc[args.protected_feature]
        assert args.dataset_split <= 1
        total = len(dataset) * args.dataset_split
        _, counts = np.unique(
            np.array(dataset[:, feature_index]), return_counts=True)
        assert len(counts) == len(args.feature_distribution)
        
        if args.equalize_dataset:
            total = np.min(counts) * len(counts)

        feature_distribution = np.array(args.feature_distribution) / sum(args.feature_distribution)
        intended_counts = total * feature_distribution
        feature_masks = [np.ma.masked_less(np.arange(
            count), intended_counts[i], copy=True).mask for i, count in enumerate(counts)]

        labelled_datasets = []
        unlabelled_datasets = []

        for f in range(len(counts)):
            dataset_data = copy.deepcopy(dataset)
            dataset_with_feature = dataset_data[dataset[:, feature_index] == f]
            if feature_masks[f].shape == ():
                continue
            
            labelled_dataset = dataset_with_feature[feature_masks[f]]
            unlabelled_dataset = dataset_with_feature[np.logical_not(feature_masks[f])]

            labelled_datasets.append(labelled_dataset)
            unlabelled_datasets.append(unlabelled_dataset)
        
        labelled_datasets = torch.tensor(np.concatenate(labelled_datasets, axis=0))
        unlabelled_datasets = torch.tensor(np.concatenate(unlabelled_datasets, axis=0))
        return labelled_datasets, unlabelled_datasets

    def update(self, proposed_data_indices):
        new_data_points = torch.index_select(self.unlabeled_train_split, 0, proposed_data_indices)
        
        self.labeled_train_split = torch.tensor(np.concatenate([self.labeled_train_split, new_data_points], axis=0))
        self.labeled_train_split = self.labeled_train_split[torch.randperm(self.labeled_train_split.size()[0])]
        
        still_unlabeled = torch.ones(self.unlabeled_train_split.shape[0], dtype=bool)
        still_unlabeled[proposed_data_indices] = False
        self.unlabeled_train_split = self.unlabeled_train_split[still_unlabeled]
    
    def get_xy_split(self, split, verbose=False): # split should be 'labeled' or 'unlabeled' or 'test
        if split == 'labeled': to_split = self.labeled_train_split 
        elif split == 'unlabeled': to_split = self.unlabeled_train_split 
        elif split == 'test': to_split = self.test_dataset
        else: raise Exception("unknown split")
        if verbose:
            print("train_split", split, to_split.shape, to_split[:, 0 : self.feature_to_predict].shape)
        X = torch.cat((to_split[:, 0 : self.feature_to_predict], to_split[:, self.feature_to_predict + 1:]), axis=1)
        y = to_split[:, self.feature_to_predict]
        if verbose:
            print("balance", X.shape, y.shape)
        # breakpoint()
        return X, y

# Download and Test Dataset
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--protected-feature', type=str, default='Sex', help='protected feature to balance')
    parser.add_argument('--feature-to-predict', type=str, default='HeartDisease', help='feature to predict')
    parser.add_argument('--feature-distribution', nargs='+', default=[0.5, 0.5], help='redistributed partition')
    parser.add_argument('--equalize-dataset', action='store_true', help='equalize dataset')
    parser.add_argument('--dataset-split', type=float, default=0.2, help='dataset split')
    parser.add_argument('--train-val-split', type=float, default=0.8, help='train/val split')
    parser.add_argument('--val-test-split', type=float, default=0.5, help='val/test split')
    parser.add_argument('--verbose', action='store_false', help='Verbose Output dataset')

    args = parser.parse_args()

    dataset = BiasedHDPDataset(args)
    if args.verbose:
        print("dataset.labeled_train_split: ", len(dataset.labeled_train_split), dataset.labeled_train_split[0])
        print("dataset.unlabeled_train_split: ", len(dataset.unlabeled_train_split), dataset.unlabeled_train_split[0])
    
    dataset.get_xy_split('labeled')
    dataset.get_xy_split('unlabeled')
    
    dataset.update(torch.tensor([0, 1, 2, 3]))
    if args.verbose:
        print("dataset.labeled_train_split: ", len(dataset.labeled_train_split))
        print("dataset.unlabeled_train_split: ", len(dataset.unlabeled_train_split))
    dataset.update(torch.tensor([0, 1, 2, 3]))
    if args.verbose:
        print("dataset.labeled_train_split: ", len(dataset.labeled_train_split))
        print("dataset.unlabeled_train_split: ", len(dataset.unlabeled_train_split))

    x, y = dataset.get_xy_split('labeled')
    x, y = dataset.get_xy_split('unlabeled')
