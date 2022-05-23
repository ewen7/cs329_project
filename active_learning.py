import numpy as np
import torch
import torch.distributions as td

def run(model, dataset, args):
    """
    Runs active learning on the dataset, with the method specified by args. Returns the indices of the proposed unlabeled data.
    """
    if args.al_method == 'random':
        return al_random(model, dataset, args)
    elif args.al_method == 'hdp':
        return al_hdp(model, dataset, args)
    else:
        raise ValueError(f"Unknown active learning method: {args.al_method}")


def al_random(model, dataset, args):
    """
    Randomly select a batch of unlabeled data to label. Returns indices of the selected unlabeled data.
    """
    return torch.from_numpy(np.random.choice(dataset.unlabeled_train_split.shape[0], size=args.al_proposal_size, replace=False))

def al_hdp(model, dataset, args):
    """
    Active learning for HPD. Returns indices of the selected unlabeled data.
    """
    pool = dataset.unlabeled_train_split
    y_pred = torch.from_numpy(model.predict(pool.X))
    entropies = td.Categorical(logits=y_pred).entropy()
    print("entropies", entropies)
    return torch.from_numpy(np.argsort(entropies)[-args.al_proposal_size:])
