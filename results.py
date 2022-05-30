import pandas as pd
import numpy as np
import pickle

class ResultsAggregator:

    def __init__(self, args, load_path=None):
        self.experiments = {}
        if load_path is not None:
            with open(load_path, 'rb') as f:
                self.experiments = pickle.load(f)
        self.overall_name = args.name
    
    def start(self, exp_name):
        self.experiments[exp_name] = {'fairness': [], 'robustness': {'test': [], 'adv': []}}

    def update_fairness(self, exp_name, results):
        self.experiments[exp_name]['fairness'].append(results)
    
    def update_robustness(self, exp_name, step, results):
        self.experiments[exp_name]['robustness']['test'].append(results['test'])
        self.experiments[exp_name]['robustness']['adv'].append(results['adv'])

    def finish(self, exp_name, save_excel=True):
        self.experiments[exp_name]['fairness'] = pd.concat(self.experiments[exp_name]['fairness'])
        self.experiments[exp_name]['robustness']['test'] = np.array(self.experiments[exp_name]['robustness']['test'])
        self.experiments[exp_name]['robustness']['adv'] = np.array(self.experiments[exp_name]['robustness']['adv'])
        if save_excel:
            self.experiments[exp_name]['fairness'].to_excel(f'./results/{self.overall_name}/{exp_name}.xlsx', index=False, sheet_name='fairness')

    def save(self, path=None):
        if path is None:
            path = f'./results/{self.overall_name}/all_results.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.experiments, f)