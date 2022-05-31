import numpy as np 
import torch 
import argparse
import pickle
import matplotlib.pyplot as plt

def generate_plots(all_results, type): # type = 'fairness' or 'robustness'
    if type == 'fairness':
        metrics = ['accuracy', 'selection_rate', 'true_positive_rate',
        'false_positive_rate', 'true_negative_rate', 'false_negative_rate',
        'treatment_equality_rate', 'equality_of_opportunity',
        'average_odds', 'acceptance_rate', 'rejection_rate']
    elif type == 'robustness':
        metrics = [16, 43, 48]
    else:
        raise Exception('Unknown Type')
    results = {metric: {} for metric in metrics}

    trial_names = all_results.keys()
    for trial_name in trial_names:
        al_type = f"{trial_name.split('-')[3]}-{trial_name.split('-')[4]}"
        
        df = all_results[trial_name][type]
        if type == 'fairness': df = df[df['class'] == 'avg']
        breakpoint()
        for idx, row in df.iterrows():
            if type == 'fairness': idx = row['n_train']
            for metric in metrics:
                if metric == 'n_train': continue
                results_by_metric = results[metric]
                if al_type not in results_by_metric: results_by_metric[al_type] = {}
                if idx not in results_by_metric[al_type]: results_by_metric[al_type][idx] = []
                results_by_metric[al_type][idx].append(row[metric])

    for metric in metrics:
        al_types = results[metric].keys()
        fig, ax = plt.subplots(nrows=1, ncols=1)
        for al_type in al_types:
            results_by_type = results[metric][al_type]
            n_trains = results_by_type.keys()
            ax.plot(n_trains, [np.mean(results_by_type[n_train]) for n_train in n_trains], label=al_type)
        ax.legend()
        fig.savefig(f"results/{args.exp_name}/{type}_{metric}.png")
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='penguin', help='experiment name')
    args = parser.parse_args()
    
    with open(f"results/{args.exp_name}/all_results.pkl", 'rb') as f:
        all_results = pickle.load(f)

    generate_plots(all_results, 'fairness')
    generate_plots(all_results, 'robustness')

