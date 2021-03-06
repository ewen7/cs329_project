import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def train_model(model, dataset, args, verbose=0):
    if args.dataset == 'spd':
        train_spd(model, dataset, args, verbose)
    elif args.dataset == 'hdp':
        train_hdp(model, dataset, args, verbose)
    elif args.dataset == 'mnist':
        train_mnist(model, dataset, args, verbose)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

def train_spd(model, dataset, args, verbose=0):
    """
    Trains the model on the dataset.
    """
    if args.model == 'lr':
        X_train, y_train = dataset.get_xy_split('labeled')
        if verbose > 0: print(f'Training Logistic Regression on {X_train.shape[0]} datapoints...')
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model: {args.model}")

def train_hdp(model, dataset, args, verbose=0):
    """
    Trains the model on the dataset.
    """
    if args.model == 'lr':
        X_train, y_train = dataset.get_xy_split('labeled')
        if verbose > 0: print(f'Training Logistic Regression on {X_train.shape[0]} datapoints...')
        model.fit(X_train, y_train)
    elif args.model == 'rfc':
        X_train, y_train = dataset.get_xy_split('labeled')
        if verbose > 0: print(f'Training Random Forest Classifier on {X_train.shape[0]} datapoints...')
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model: {args.model}")

def train_mnist(model, dataset, args, verbose=1):
    """
    Trains the MNIST model on the dataset
    """
    if args.model == 'lr':
        X_train, y_train = dataset.get_xy_split('labeled')
        if verbose > 0: print(f'Training Logistic Regression on {X_train.shape[0]} datapoints...')
        model.fit(X_train, y_train)
    elif args.model == 'cnn':
        model.training = True
        X_train, y_train = dataset.get_xy_split('labeled')
        if verbose > 0: print(f'Training CNN on {X_train.shape[0]} datapoints...')
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        verbose_tqdm = tqdm if verbose > 0 else lambda x: x
        for epoch in verbose_tqdm(range(args.epochs)):
            avg_loss = []
            for start_idx in range(0, X_train.shape[0], args.batch_size):
                batch_size = min(args.batch_size, X_train.shape[0] - start_idx)
                data = X_train[start_idx : start_idx+batch_size]
                target = y_train[start_idx : start_idx+batch_size]
                optimizer.zero_grad()
                output = model(data)
                loss = F.nll_loss(output, target)
                avg_loss.append(loss.item())
                loss.backward()
                optimizer.step()
            # print(f'Epoch {epoch+1}/{args.epochs} | Loss: {sum(avg_loss)/len(avg_loss)}')
        model.training = False
