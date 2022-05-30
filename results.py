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
        self.experiments[exp_name] = {'fairness': [], 'robustness': []}

    def update_fairness(self, exp_name, results):
        self.experiments[exp_name]['fairness'].append(results)
    
    def update_robustness(self, exp_name, results):
        self.experiments[exp_name]['robustness'].append(results)

    def finish(self, exp_name, save_excel=True):
        self.experiments[exp_name]['fairness'] = pd.concat(self.experiments[exp_name]['fairness'])
        self.experiments[exp_name]['robustness'] = pd.concat(self.experiments[exp_name]['robustness'])
        if save_excel:
            writer = pd.ExcelWriter(f'./results/{self.overall_name}/{exp_name}.xlsx', engine='xlsxwriter')
            self.experiments[exp_name]['fairness'].to_excel(writer, index=False, sheet_name='fairness')
            self.experiments[exp_name]['robustness'].to_excel(writer, index=False, sheet_name='robustness')
            writer.save()

    def save(self, path=None):
        if path is None:
            path = f'./results/{self.overall_name}/all_results.pkl'
        with open(path, 'wb') as f:
            pickle.dump(self.experiments, f)