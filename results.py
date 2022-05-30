import pandas as pd
import numpy as np
import pickle

class ResultsAggregator:

    def __init__(self, args, load_path=None):
        self.experiments = {}
        if load_path is not None:
            with open(load_path, 'rb') as f:
                self.experiments = pickle.load(f)
    
    def update(self, exp_name, results):
        if exp_name not in self.experiments:
            self.experiments[exp_name] = results
        else:
            self.experiments[exp_name] = self.experiments[exp_name].append(results, ignore_index=True)

    def save(self, path='./results/mnist_results.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self.experiments, f)