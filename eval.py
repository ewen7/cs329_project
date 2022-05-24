from statistics import mode
import numpy as np
from regex import E
import torch
import pandas as pd
from builtins import breakpoint
from fairness import BinaryClassificationBiasDataset
from explainability import ModelExplainer
import datetime
import os
import tensorflow as tf
from tensorboard import summary

class Logger(object):
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
        #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        #self.writer.add_summary(summary, step)
    def merge(self, prefix):
         tf.summary.merge([prefix+str(i) for i in range(3)])
    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        with self.writer.as_default():
            for tag, value in tag_value_pairs:
                tf.summary.scalar(tag, value, step=step)
            self.writer.flush()
        #summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value) for tag, value in tag_value_pairs])
        #self.writer.add_summary(summary, step)

log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
summary_writer =  Logger(log_dir)

def eval(model, dataset, step, args, verbose=False):
    """
    Runs evaluation on the test dataset.
    """
    # compute accuracy / log to tensorboard?
    # print(f"eval (acc): {eval_accuracy(model, dataset, args, verbose):.4f}")
    # acc_c0, acc_c1 = eval_accuracy(model, dataset, args, verbose)
    # with summary_writer.as_default(step=10):

    print(f"eval (acc): {eval_accuracy(model, dataset, args, verbose):.4f}")
    if args.dataset == 'hdp' or args.dataset == 'spd':
        print("eval (fairness): ")
        fairness_evals, fairness_metrics = eval_fairness(model, dataset, args, verbose=True)
        for i, metric in enumerate(fairness_metrics):
            for j in range(3):
                summary_writer.scalar_summary(metric + '_c' + str(j), fairness_evals[j][i], step)
            summary_writer.scalar_summary(metric + '_c01_diff', abs(fairness_evals[0][i] - fairness_evals[1][i]), step)

        print("eval (Explainability): ")
        eval_explainability(model, dataset, args, verbose=True)


def eval_accuracy(model, dataset, args, verbose=False):
    X_test, y_test = dataset.get_xy_split('test')
    y_hat = torch.from_numpy(model.predict(X_test))
    
    return (y_hat == y_test).sum().item() / y_test.shape[0]


def eval_fairness(model, dataset, args, verbose=True, save=True):
    """
    Computes predictive parity for the different protected classes specified in args.protected_classes
    """
    fairness_evals = []
    X_test, y_test = dataset.get_xy_split('test')
    y_hat = torch.from_numpy(model.predict(X_test))
    fairness_metrics = ['accuracy', 'selection_rate', 'true_positive_rate', 'false_positive_rate', 'true_negative_rate', 'false_negative_rate', 'treatment_equality_rate',
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
        evals = []
        for metric in fairness_metrics:
            eval = fairness_eval.get_bias_result_from_metric(metric)
            evals.append(eval)
            if verbose: print(f"{metric}: {eval:.4f}")
        if save:
            if i == 0: df = pd.DataFrame(evals, columns=[i], index=fairness_metrics)
            else: df = pd.concat([df, pd.DataFrame(evals, columns=[i], index=fairness_metrics)], axis=1)
        fairness_evals.append(evals)
    if save:
        df.T.to_excel('./results/eval_metrics.xlsx', sheet_name='sheet1', index=False)
    return fairness_evals, fairness_metrics

def eval_explainability(model, dataset, args, verbose=False):
    """
    Computes robustness metrics  for the different protected classes specified in args.protected_classes
    """
    model_explainer = ModelExplainer(model, dataset, args)
    model_explainer.explain_model()


