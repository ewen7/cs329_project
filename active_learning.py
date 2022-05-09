

def run(model, dataset, args):
    """
    Runs active learning on the dataset, with the method specified by args. Returns the indices of the proposed unlabeled data.
    """
    if args.al_method == 'random':
        return random(model, dataset, args)
    else:
        raise ValueError(f"Unknown active learning method: {args.al_method}")


def random(model, dataset, args):
    """
    Randomly select a batch of unlabeled data to label. Returns indices of the selected unlabeled data.
    """
    # TODO
    pass