import numpy as np
import torch
from builtins import breakpoint
from fairness import BinaryClassificationBiasDataset


def eval(model, dataset, args, verbose=False):
    """
    Runs evaluation on the test dataset.
    """
    # compute accuracy / log to tensorboard?

    print(f"eval (acc): {eval_accuracy(model, dataset, args, verbose):.4f}")
    if args.dataset == 'hdp' or args.dataset == 'spd':
        print("eval (fairness): ")
        eval_fairness(model, dataset, args, verbose=True)


def eval_accuracy(model, dataset, args, verbose=False):
    X_test, y_test = dataset.get_xy_split('test')
    y_hat = torch.from_numpy(model.predict(X_test))
    
    return (y_hat == y_test).sum().item() / y_test.shape[0]


def eval_fairness(model, dataset, args, verbose=True):
    """
    Computes predictive parity for the different protected classes specified in args.protected_classes
    """
    X_test, y_test = dataset.get_xy_split('test')
    y_hat = torch.from_numpy(model.predict(X_test))
    fairness_metrics = ['selection_rate', 'true_positive_rate', 'false_positive_rate', 'true_negative_rate', 'false_negative_rate', 'treatment_equality_rate',
                        'equality_of_opportunity', 'average_odds', 'acceptance_rate', 'rejection_rate']

    X_protected = X_test[:, dataset.protected_feature]

    y_hat0 = y_hat * (1.0 - X_protected)
    y_test0 = y_test * (1.0 - X_protected)
    y_hat1 = y_hat * (X_protected)
    y_test1 = y_test * (X_protected)

    for i, (y_hat, y_test) in enumerate([(y_hat0, y_test0), (y_hat1, y_test1), (y_hat, y_test)]):
        fairness_eval = BinaryClassificationBiasDataset(
            preds=y_hat, labels=y_test, positive_class_favored=False)
        if i == 2: print("BOTH CLASSES")
        else: print("CLASS ", i)
        for metric in fairness_metrics:
            eval = fairness_eval.get_bias_result_from_metric(metric)
            if verbose: print(f"{metric}: {eval:.4f}")

def eval_robustness(model, dataset, args, verbose=False):
    """
    Computes robustness metrics  for the different protected classes specified in args.protected_classes
    """
    # TODO
    pass