

class BiasedSPDDataset(Dataset):

    # prep_dataset basically calls init
    def __init__(self, dir, args):
        # load entire SPD dataset
        # run the logic of redistribute
        self.labeled_train_split = ...
        self.unlabeled_train_split = ...
        # self.val_split = ... # probably don't need val for spd?
        self.test_split = ...

    def __getitem__(self, index):
        return self.labeled_train_split[index]

    def __len__(self):
        return len(self.labeled_train_split)
    
    def update(self, proposed_data):
        # transfer proposed_data from unlabeled to labeled