

def eval(model, dataset, args, verbose=False):
    """
    Runs evaluation on the test dataset.
    """
    # compute accuracy / log to tensorboard?

    eval_fairness(model, dataset, args, verbose)


def eval_fairness(model, dataset, args, verbose=False):
    """
    Computes predictive parity for the different protected classes specified in args.protected_classes
    """
    # TODO
    pass