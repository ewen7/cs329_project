import numpy as np
import torch
import torch.distributions as td
from tqdm import tqdm
from builtins import breakpoint

def run(model, dataset, args):
    """
    Runs active learning on the dataset, with the method specified by args. Returns the indices of the proposed unlabeled data.
    """
    if args.al_method == 'random':
        return al_random(model, dataset, args)
    elif args.al_method == 'entropy':
        return al_entropy(model, dataset, args)
    elif args.al_method == 'entropy_classrep':
        return al_entropy_classrep(model, dataset, args)
    elif args.al_method == 'cnn_distance':
        return al_cnn_distance(model, dataset, args)
    else:
        raise ValueError(f"Unknown active learning method: {args.al_method}")

def al_random(model, dataset, args):
    """
    Randomly select a batch of unlabeled data to label. Returns indices of the selected unlabeled data.
    """
    weights = np.ones(dataset.unlabeled_train_split.shape[0])
    if args.dataset == 'hdp':
        unlabeled_pc = dataset.unlabeled_train_split[:, dataset.protected_feature]
        labeled_pc = dataset.labeled_train_split[:, dataset.protected_feature]
        p_pc = labeled_pc.mean()
        weights[unlabeled_pc == 0] = (1 - p_pc) / (1 - unlabeled_pc[unlabeled_pc == 0]).sum()
        weights[unlabeled_pc == 1] = p_pc / (unlabeled_pc[unlabeled_pc == 1]).sum()
        # m, n unlabeled
        # a, b labeled
        # a/m(a+b), b/n(a+b) weights for unlabeled to produce a, b overall ratio
    elif args.dataset == 'mnist':
        unlabeled_labels = dataset.unlabeled_y_train
        for i in range(10):
            weights[unlabeled_labels == i] = args.feature_distribution[i] * unlabeled_labels.shape[0] / unlabeled_labels[unlabeled_labels == i].shape[0]

    weights /= weights.sum()
    return torch.from_numpy(np.random.choice(dataset.unlabeled_train_split.shape[0], size=args.al_proposal_size, replace=False, p=weights))

def al_entropy(model, dataset, args):
    """
    Active learning with entropy for HPD and MNIST. Returns indices of the selected unlabeled data.
    """
    poolX, _ = dataset.get_xy_split('unlabeled')
    y_probs = torch.from_numpy(model.predict_proba(poolX))
    entropies = td.Categorical(probs=y_probs).entropy()
    if args.al_sampling == 'top':
        return torch.argsort(entropies, descending=True)[:args.al_proposal_size]
    elif args.al_sampling == 'weighted':
        weights = np.exp(args.kappa*entropies.numpy())
        weights /= weights.sum()
        return torch.from_numpy(np.random.choice(dataset.unlabeled_train_split.shape[0], size=args.al_proposal_size, replace=False, p=weights))
    else:
        raise ValueError(f"Unknown active learning sampling method: {args.al_sampling}")

# note: this is only written for binary classification
def al_entropy_classrep(model, dataset, args):
    """
    Active learning with entropy and class representation for HPD. Returns indices of the selected unlabeled data.
    """
    assert args.dataset == 'hdp'
    poolX, _ = dataset.get_xy_split('unlabeled')
    y_probs = torch.from_numpy(model.predict_proba(poolX))
    entropies = td.Categorical(probs=y_probs).entropy()
    unlabeled_pc = dataset.unlabeled_train_split[:, dataset.protected_feature]
    labeled_pc = dataset.labeled_train_split[:, dataset.protected_feature]
    prevalence_1 = labeled_pc.mean()
    representation_scores = (unlabeled_pc == 0) / (1 - prevalence_1) + (unlabeled_pc == 1) / prevalence_1
    
    scores = entropies * representation_scores
    if args.al_sampling == 'top':
        return torch.argsort(scores, descending=True)[:args.al_proposal_size]
    elif args.al_sampling == 'weighted':
        weights = np.exp(args.kappa*scores.numpy())
        weights /= weights.sum()
        return torch.from_numpy(np.random.choice(dataset.unlabeled_train_split.shape[0], size=args.al_proposal_size, replace=False, p=weights))
    else:
        raise ValueError(f"Unknown active learning sampling method: {args.al_sampling}")

# note: this is only written for mnist / multi-class classification
def al_cnn_distance(model, dataset, args):
    """
    Active learning with distance for a CNN model on MNIST.
    """
    assert args.dataset == 'mnist' and args.model == 'cnn'
    poolX, _ = dataset.get_xy_split('unlabeled')
    labeledX, _ = dataset.get_xy_split('labeled')
    pool_embeds = model.forward(poolX, return_embedding=True)
    labeled_embeds = model.forward(labeledX, return_embedding=True)
    # compute the average Euclidean distance from each unlabeled point to all the labeled points
    # vectorized (TOO SLOW, DON'T RUN):
    # # distances = (((pool_embeds.unsqueeze(1) - labeled_embeds.unsqueeze(0))**2).sum(dim=2)**0.5).sum(dim=1)
    # print('Computing distances...')
    distances = np.zeros(pool_embeds.shape[0])
    # for i in tqdm(range(pool_embeds.shape[0])):
    for i in range(pool_embeds.shape[0]):
        distances[i] = (((pool_embeds[i].unsqueeze(0) - labeled_embeds)**2).sum(dim=1)**0.5).mean().item()
    distances = torch.from_numpy(distances)
    if args.al_sampling == 'top':
        return torch.argsort(distances, descending=True)[:args.al_proposal_size]
    elif args.al_sampling == 'weighted':
        weights = np.exp(args.kappa*distances.numpy())
        weights /= weights.sum()
        return torch.from_numpy(np.random.choice(dataset.unlabeled_train_split.shape[0], size=args.al_proposal_size, replace=False, p=weights))
    else:
        raise ValueError(f"Unknown active learning sampling method: {args.al_sampling}")
