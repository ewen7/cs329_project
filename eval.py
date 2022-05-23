import numpy as np
import torch
from builtins import breakpoint

def eval(model, dataset, args, verbose=False):
    """
    Runs evaluation on the test dataset.
    """
    # compute accuracy / log to tensorboard?

    print("eval", eval_accuracy(model, dataset, args, verbose))

# protected class: gender (male: 1, female: 0)
def eval_accuracy(model, dataset, args, verbose=False):
    X_test, y_test = dataset.get_xy_split('test')
    y_hat = torch.from_numpy(model.predict(X_test))
    
    X_protected = X_test[:, dataset.protected_feature]
    accuracy_class0 = (torch.eq(y_hat, y_test) * (1.0 - X_protected)).sum() / (1.0 - X_protected).sum()
    accuracy_class1 = (torch.eq(y_hat, y_test) * (X_protected)).sum() / X_protected.sum()

    return accuracy_class0, accuracy_class1


def eval_fairness(model, dataset, args, verbose=False):
    """
    Computes predictive parity for the different protected classes specified in args.protected_classes
    """
    # TODO
    pass