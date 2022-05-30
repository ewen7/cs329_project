import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt
import active_learning
from train import train_model
from eval import eval
from models import init_model
from dataset import BiasedHDPDataset, BiasedSPDDataset, BiasedMNISTDataset
import os
from eval import Logger
import datetime
import random
from results import ResultsAggregator
from tqdm import tqdm

SEEDS = [1337, 42, 123, 2022, 329]

def run(args, verbose=1):
    if verbose > 0: print('Seed: ', args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dataset == 'spd':
        dataset = BiasedSPDDataset(args)
    elif args.dataset == 'hdp':
        dataset = BiasedHDPDataset(args)
    elif args.dataset == 'mnist':
        dataset = BiasedMNISTDataset(args)
    else:
        raise Exception("Unknown dataset.")

    verbose_tqdm = tqdm if verbose == 0 else lambda x: x
    for al_iter in verbose_tqdm(range(args.al_iters + 1)):
        if verbose > 0: print(f"AL iteration {al_iter}")

        model = init_model(args)

        train_model(model, dataset, args, verbose=verbose)

        eval(model, dataset, len(dataset), args, verbose=verbose)

        if al_iter < args.al_iters:
            proposed_data_indices = active_learning.run(model, dataset, args)
            
            dataset.update(proposed_data_indices)


if __name__ == '__main__':
    # parse arguments: 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hdp', help='dataset name')
    parser.add_argument('--model', type=str, default='lr', help='model name')
    parser.add_argument('--name', type=str, default='exp', help='experiment name')
    parser.add_argument('--seed', type=int, default=1729, help='random seed')
    parser.add_argument('--num-trials', type=int, default=1, help='number of times to run each experiment')
    parser.add_argument('--verbose', action='store_true', help='Verbose Output dataset')

    # al params
    parser.add_argument('--al-iters', type=int, default=100, help='number of loops of active learning')
    parser.add_argument('--al-method', type=str, default='random', help='active learning method')
    parser.add_argument('--al-proposal-size', type=int, default=100, help='number of unlabeled data to propose')
    parser.add_argument('--al-sampling', type=str, default='top', help='sampling method for active learning')
    parser.add_argument('--kappa', type=float, default=2.0, help='active learning weighted sampling pre-softmax scaling')

    # artificial bias params
    parser.add_argument('--protected-feature', type=str, default='Sex', help='protected feature to balance')
    parser.add_argument('--feature-to-predict', type=str, default='HeartDisease', help='feature to predict')
    parser.add_argument('--feature-distribution', nargs='+', default=[], help='redistributed partition')
    parser.add_argument('--dataset-split', type=float, default=0.01, help='dataset split')

    # hdp params
    parser.add_argument('--train-val-split', type=float, default=0.8, help='train/val split')
    parser.add_argument('--val-test-split', type=float, default=0.5, help='val/test split')
    parser.add_argument('--equalize-dataset', action='store_true', help='equalize dataset')
    parser.add_argument('--remove-protected-char', action='store_true', help='Removes protected Characteristic from training')

    # mnist cnn training params
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=20, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='momentum')

    # robustness params
    parser.add_argument('--adv-type', type=str, default='gaussian', help='method of generating adversarial data')
    parser.add_argument('--gaussian-stds', nargs='+', default=[16.0, 32.0, 48.0], help='gaussian noise levels')

    args = parser.parse_args()

    if len(args.feature_distribution) == 0:
        # Set default feature distribution if none provided
        if args.dataset == 'spd':
            args.feature_distribution = [0.5, 0.5]
        elif args.dataset == 'hdp':
            args.feature_distribution = [0.5, 0.5]
        elif args.dataset == 'mnist':
            args.feature_distribution = [0.1] * 10
    else:
        args.feature_distribution = [float(x) for x in args.feature_distribution]

    if args.al_method == 'all':
        if os.path.isdir(f'./results/{args.name}'):
            raise Exception(f"Experiment name {args.name} already exists.")
        os.mkdir(f'./results/{args.name}')

        # run a search across all methods and average over multiple trials
        args.results_aggregator = ResultsAggregator(args)
        for al_method, al_sampling in [('random', None), ('entropy', 'top'), ('entropy', 'weighted'), ('cnn_distance', 'top'), ('cnn_distance', 'weighted')]:
            args.al_method, args.al_sampling = al_method, al_sampling
            if args.al_method == 'cnn_distance': # fairly arbitrary hyperparameter lol
                args.kappa = 0.08
            else:
                args.kappa = 3.5
            assert args.num_trials <= len(SEEDS), 'Number of trials must be less than or equal to the number of predetermined seeds; please add more seeds.'
            for seed in SEEDS[:args.num_trials]:
                args.seed = seed
                args.exp_name = f'{args.name}-{args.dataset}-{args.model}-{args.al_method}-{args.al_sampling}-seed{args.seed}'
                log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+args.exp_name)
                args.summary_writer = Logger(log_dir)
                print('Running experiment: ', args.exp_name)
                args.results_aggregator.start(args.exp_name)
                run(args, verbose=0)
                args.results_aggregator.finish(args.exp_name)
                args.results_aggregator.save()

    else:
        # run once
        args.results_aggregator = None
        args.exp_name = f'{args.name}-{args.dataset}-{args.model}-{args.al_method}-{args.al_sampling}-seed{args.seed}-remove_protected_char{args.remove_protected_char}'
        log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'-'+args.exp_name)
        args.summary_writer = Logger(log_dir)

        # run(args, verbose=(args.verbose == 1))
        run(args)