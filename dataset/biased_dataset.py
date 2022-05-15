from SPD_dataset import SPDDataset
from torch.utils.data import Dataset
import numpy as np
import copy
import torch
import os

dataset_dir = os.path.join(os.path.dirname(os.path.realpath(
    __file__)), 'SPD/healthcare-dataset-stroke-data.csv')

class BiasedSPDDataset(Dataset):

    # prep_dataset basically calls init
    def __init__(self, args, dataset_dir=dataset_dir):
        # load entire SPD dataset
        # run the logic of redistribute
        # args.percent_labelled
        datasets = SPDDataset(dataset_dir=dataset_dir, config=args)
        self.train_dataset, self.val_dataset, self.test_dataset = datasets.train_dataset, datasets.val_dataset, datasets.test_dataset
        print("train", self.train_dataset[0].shape)
        self.labeled_train_split, self.unlabeled_train_split = self.redistribute_dataset(
            self.train_dataset,
            datasets.feature_to_loc,
            args.get('feature_name', 'gender'),
            args.get('feature_distribution', [0.5]*2),
            args.get('equalize_dataset', False),
            args.get('dataset_split', 0.5),
        )

    def __getitem__(self, index):
        return self.labeled_train_split[index]

    def __len__(self):
        return len(self.labeled_train_split)

    def redistribute_dataset(
        self, 
        dataset, 
        feature_to_loc, 
        feature_name, 
        feature_distribution, 
        equalize_dataset, 
        dataset_split):
        feature_index = feature_to_loc[feature_name]
        assert dataset_split <= 1
        total = len(dataset) * dataset_split
        _, counts = np.unique(
            np.array(dataset[:, feature_index]), return_counts=True)
        assert len(counts) == len(feature_distribution)
        
        if equalize_dataset:
            total = np.min(counts) * len(counts)

        feature_distribution = np.array(
            feature_distribution) / sum(feature_distribution)
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
        
        return labelled_datasets, unlabelled_dataset

    def update(self, proposed_data_indices):
        new_data_points = torch.index_select(torch.tensor(self.unlabeled_train_split), 0, proposed_data_indices)
        self.labeled_train_split = torch.tensor(np.concatenate([self.labeled_train_split, new_data_points], axis=0))
        # Random Permutation of labelled dataset
        self.labeled_train_split = self.labeled_train_split[torch.randperm(self.labeled_train_split.size()[0])]
        
        self.unlabeled_train_split = np.ma.masked_where(
            np.arange(len(self.unlabeled_train_split)) == proposed_data_indices, self.unlabeled_train_split
        )
    
    def get_xy_split(self, split): # split should be 'labeled' or 'unlabeled'
        
        return X_train, y_train

# Download and Test Dataset
if __name__ == "__main__":
    dataset = BiasedSPDDataset({})
    print("dataset.labeled_train_split: ", len(dataset.labeled_train_split), dataset.labeled_train_split[0])
    print("dataset.unlabeled_train_split: ", len(dataset.unlabeled_train_split), dataset.unlabeled_train_split[0])
    dataset.update(torch.tensor([0, 1, 2, 3]))
    print("dataset.labeled_train_split: ", len(dataset.labeled_train_split))
    print("dataset.unlabeled_train_split: ", len(dataset.unlabeled_train_split))
    dataset.update(torch.tensor([0, 1, 2, 3]))
    print("dataset.labeled_train_split: ", len(dataset.labeled_train_split))
    print("dataset.unlabeled_train_split: ", len(dataset.unlabeled_train_split))
