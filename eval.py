

def eval(model, dataset, args, verbose=False):
    """
    Runs evaluation on the test dataset.
    """
    # compute accuracy / log to tensorboard?

    print(eval_accuracy(model, dataset, args, verbose))

# protected class: gender (male: 1, female: 0)
def eval_accuracy(model, dataset, args, verbose=False):
    X_test, y_test = dataset.get_xy_split('test')
    y_hat = model.predict(X_test)
    accuracy_class0 = ((y_hat == y_test) * (1.0 - X_test[:, 1])).sum() / (1.0 - X_test[:, 1]).sum()
    accuracy_class1 = ((y_hat == y_test) * (X_test[:, 1])).sum() / X_test[:, 1].sum()
    
    return accuracy_class0, accuracy_class1


def eval_fairness(model, dataset, args, verbose=False):
    """
    Computes predictive parity for the different protected classes specified in args.protected_classes
    """
    # TODO
    pass