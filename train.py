

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
        X_train, y_train = dataset.get_xy_split('labeled')
        print(f'Training Logistic Regression on {X_train.shape[0]} datapoints...')
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model: {args.model}")