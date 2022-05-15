

def train_model(model, dataset, args, verbose=False):
    if args.dataset == 'spd':
        train_spd(model, dataset, args, verbose)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def train_spd(model, dataset, args, verbose=False):
    """
    Trains the model on the dataset.
    """
    if args.model == 'lr':
        model.fit(dataset.unlabeled_train_split[0], dataset.unlabeled_train_split[1])
    else:
        raise ValueError(f"Unknown model: {args.model}")