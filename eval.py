import numpy as np
import torch
import pandas as pd
from builtins import breakpoint
from fairness import BinaryClassificationBiasDataset
from explainability import ModelExplainer

def eval(model, dataset, args, verbose=False):
    """
    Runs evaluation on the test dataset.
    """
    # compute accuracy / log to tensorboard?

    print("eval (acc): ", eval_accuracy(model, dataset, args, verbose))
    print("eval (fairness): ")
    eval_fairness(model, dataset, args, verbose=True)
    print("eval (explainer): ")
    eval_explainability(model, dataset, args, verbose=True)


# protected class: gender (male: 1, female: 0)
def eval_accuracy(model, dataset, args, verbose=False):
    X_test, y_test = dataset.get_xy_split('test')
    y_hat = torch.from_numpy(model.predict(X_test))
    
    X_protected = X_test[:, dataset.protected_feature]
    accuracy_class0 = (torch.eq(y_hat, y_test) * (1.0 - X_protected)).sum() / (1.0 - X_protected).sum()
    accuracy_class1 = (torch.eq(y_hat, y_test) * (X_protected)).sum() / X_protected.sum()

    return accuracy_class0, accuracy_class1


def eval_fairness(model, dataset, args, verbose=True, save=True):
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
        if verbose:
            if i == 2: print("BOTH CLASSES")
            else: print("CLASS ", i)

        fairness_eval = BinaryClassificationBiasDataset(
            preds=y_hat, labels=y_test, positive_class_favored=False)
        if save:
            df = pd.DataFrame(columns=fairness_metrics, index=[0, 1, 2])
        for metric in fairness_metrics:
            eval = fairness_eval.get_bias_result_from_metric(metric)
            if verbose: print(metric, ": ", eval)
            if save: df.at[i, metric] = eval
    if save:
        print(df.head())
        df.to_excel('./results/eval_metrics.xlsx', sheet_name='sheet1', index=False)
        exit()

def eval_explainability(model, dataset, args, verbose=False):
    """
    Computes robustness metrics  for the different protected classes specified in args.protected_classes
    """
    model_explainer = ModelExplainer(model, dataset, args)
    model_explainer.explain_model()