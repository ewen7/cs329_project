from sklearn.linear_model import LogisticRegression


def init_model(args):
    """
    Initializes a model based on the value of args.model (either 'LR' or 'NN'?)
    """
    if args.dataset == 'spd':
        if args.model == 'lr':
            return LogisticRegression(max_iter=1000)
        else:
            raise ValueError(f"Unknown model: {args.model}")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


class MNISTModel():
    # TODO
    pass