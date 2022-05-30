import argparse
from statistics import mode
import numpy as np
from regex import E
import torch
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
from builtins import breakpoint
from fairness import BinaryClassificationBiasDataset
from explainability import ModelExplainer
from robustness import MNISTRobustness
import datetime
import os
import tensorflow as tf
from tensorboard import summary
from tqdm import tqdm

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

def eval(model, dataset, step, args, verbose=0):
    """
    Runs evaluation on the test dataset.
    """
    # compute accuracy / log to tensorboard?
    # print(f"eval (acc): {eval_accuracy(model, dataset, args, verbose):.4f}")
    # acc_c0, acc_c1 = eval_accuracy(model, dataset, args, verbose)
    # with summary_writer.as_default(step=10):

    if verbose > 0: print(f"eval (acc): {eval_accuracy(model, dataset, step, args, verbose):.4f}")
    
    if verbose > 0: print("eval (fairness): ")
    fairness_evals, fairness_metrics, result_update = eval_fairness(model, dataset, args, verbose, save_excel=False)

    if args.results_aggregator is not None:
        # send to aggregator
        result_update['class'] = list(range(10))
        result_update['n_train'] = [(dataset.labeled_y_train == i).sum().item() for i in range(10)]
        result_avg = result_update.mean()
        result_avg['class'] = 'avg'
        result_avg['n_train'] = step
        result_update = result_update.append(result_avg, ignore_index=True)
        args.results_aggregator.update_fairness(args.exp_name, result_update)

    # log
    for i, metric in enumerate(fairness_metrics):
        num_classes = 10 if args.dataset == 'mnist' else 3
        for j in range(num_classes):
            args.summary_writer.scalar_summary(metric + '_c' + str(j), fairness_evals[j][i], step)
        if args.dataset == 'hdp':
            args.summary_writer.scalar_summary(metric + '_c01_diff', abs(fairness_evals[0][i] - fairness_evals[1][i]), step)
        agg = []
        for j in range(len(fairness_evals)):
            agg.append(fairness_evals[j][i])
        args.summary_writer.scalar_summary(metric + '_agg', np.mean(agg), step)

    if verbose > 0: ("eval (Explainability): ")
    if args.dataset == 'hdp' or args.dataset == 'spd':
        eval_explainability(model, dataset, args, verbose)

    if verbose > 0: print(f"eval (Robustness): ")
    if args.dataset == 'mnist':
        robustness_results = eval_robustness(model, dataset, step, args, verbose)
        if args.results_aggregator is not None:
            args.results_aggregator.update_robustness(args.exp_name, step, robustness_results)
    

def eval_accuracy(model, dataset, step, args, verbose=1):
    X_test, y_test = dataset.get_xy_split('test')
    y_hat = torch.from_numpy(model.predict(X_test))
    
    acc = (y_hat == y_test).sum().item() / y_test.shape[0]

    args.summary_writer.scalar_summary('OVERALL_ACC', acc, step)
    return acc

def eval_explainability(model, dataset, args, verbose=1):
    """
    Computes robustness metrics  for the different protected classes specified in args.protected_classes
    """
    model_explainer = ModelExplainer(model, dataset, args)
    model_explainer.explain_model()

# https://towardsdatascience.com/comprehensive-guide-on-multiclass-classification-metrics-af94cfb83fbd
def eval_fairness(model, dataset, args, verbose=1, save_excel=True):
    """
    Computes predictive parity for the different protected classes specified in args.protected_classes
    """
    fairness_evals = []
    X_test, y_test = dataset.get_xy_split('test')
    y_hat = torch.from_numpy(model.predict(X_test))
    fairness_metrics = ['accuracy', 'selection_rate', 'true_positive_rate', 'false_positive_rate', 'true_negative_rate', 'false_negative_rate', 'treatment_equality_rate',
                        'equality_of_opportunity', 'average_odds', 'acceptance_rate', 'rejection_rate']

    y_hats, y_tests = [], []
    if args.dataset == 'hdp':
        num_classes = 3
        X_protected = X_test[:, dataset.protected_feature]
        y_hats = [y_hat * (1.0 - X_protected), y_hat * (X_protected), y_hat]
        y_tests = [y_test * (1.0 - X_protected),  y_test * (X_protected), y_test]
    elif args.dataset == 'mnist':
        num_classes = 10
        for c in range(num_classes):
            y_hats.append(y_hat == c)
            y_tests.append(y_test == c) 
    else:
        raise Exception("Unknown Dataset")
    
    for c in range(num_classes):
        y_hat = y_hats[c]
        y_test = y_tests[c]
        if verbose > 0: print("CLASS ", c)

        fairness_eval = BinaryClassificationBiasDataset(
            preds=y_hat, labels=y_test, positive_class_favored=False)
        evals = []
        for metric in fairness_metrics:
            eval = fairness_eval.get_bias_result_from_metric(metric)
            evals.append(eval)
            if verbose > 0: print(f"{metric}: {eval:.4f}")
        if c == 0: df = pd.DataFrame(evals, columns=[c], index=fairness_metrics)
        else: df = pd.concat([df, pd.DataFrame(evals, columns=[c], index=fairness_metrics)], axis=1)
        fairness_evals.append(evals)
    if verbose > 0: print("Eval Fairness Complete")
    if save_excel:
        df.T.to_excel('./results/multi_eval_metrics.xlsx', sheet_name='sheet1', index=False)
    return fairness_evals, fairness_metrics, df.T

# adv_type: 'gaussian'
def eval_robustness(model, dataset, step, args, verbose=1):
    # generate adversarial dataset
    X_test, _ = dataset.get_xy_split('test')
    if args.adv_type == 'gaussian':
        perturbation = torch.normal(mean=0.0, std=1.0, size=X_test.shape)
        X_adv = X_test + perturbation 
    else: 
        raise Exception("Unknown adv_type for robustness")

    robustness = MNISTRobustness(model, dataset)
    test_acc, adv_acc = robustness.accuracy(X_adv, verbose=verbose)
    args.summary_writer.scalar_summary('robustness_test', test_acc, step)
    args.summary_writer.scalar_summary('robustness_adv', adv_acc, step)
    return {'test': test_acc, 'adv': adv_acc}
