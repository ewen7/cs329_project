from enum import Enum
from enum import unique

import numpy as np
import pandas as pd
from sklearn import metrics

# Lab 4 modified

class BinaryClassificationBiasDataset(object):
    def __init__(self,
                 preds,
                 labels,
                 positive_class_favored=False):
        self.preds = np.array(preds)
        self.labels = np.array(labels)

        self.positive_class_favored = positive_class_favored 
        self._set_confusion_matrix()

    def _set_confusion_matrix(self):
        self.num_true_negatives, self.num_false_positives, self.num_false_negatives, self.num_true_positives = metrics.confusion_matrix(
            self.labels, self.preds).ravel()

    def true_positives(self):
        return self.num_true_positives

    def true_negatives(self):
        return self.num_true_negatives

    def false_positives(self):
        return self.num_false_positives

    def false_negatives(self):
        return self.num_false_negatives

    def true_positive_rate(self):
        return self.num_true_positives / (self.num_true_positives + self.num_false_negatives)

    def false_positive_rate(self):
        return self.num_false_positives / (self.num_false_positives + self.num_true_negatives)

    def true_negative_rate(self):
        return 1 - self.false_positive_rate()

    def false_negative_rate(self):
        return 1 - self.true_positive_rate()

    def selection_rate_positive(self):
        """
        Ratio of the number of positive predictions to the total number of predictions
        """
        return np.sum(self.preds) / len(self.preds)
    
    def accuracy(self):
        return (self.preds == self.labels).sum().item() / self.labels.shape[0]

    def selection_rate(self):
        """
        Equal to selection_rate_positive if positive class is favored, 1 - selection_rate_positive otherwise
        """
        selection_rate_positive = self.selection_rate_positive()
        return selection_rate_positive if self.positive_class_favored else 1 - selection_rate_positive

    def treatment_equality_rate(self):
        return self.num_false_positives / self.num_false_negatives

    def equality_of_opportunity(self):
        return self.true_positive_rate(
        ) if self.positive_class_favored else self.true_negative_rate()

    def average_odds(self):
        """
        Average of TPR and FPR
        """
        return (self.false_positive_rate() + self.true_positive_rate()) / 2

    def acceptance_rate(self):
        """
        Equal to positive acceptance rate (how many of the predicted positives were actually correct)
        if positive class is favored, else equal to negative acceptance rate
        """
        negative_acceptance_rate = self.num_true_negatives / (self.num_true_negatives + self.num_false_negatives)
        positive_acceptance_rate = self.num_true_positives / (self.num_true_positives + self.num_false_positives)
        return positive_acceptance_rate if self.positive_class_favored else negative_acceptance_rate

    def rejection_rate(self):
        return 1 - self.acceptance_rate()

    def get_bias_result_from_metric(self, metric_fn_str):
        metric_fn = getattr(self, metric_fn_str, None)
        if metric_fn is None:
            return None
        return metric_fn()