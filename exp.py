import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt
import active_learning
from train import train_model
from eval import eval
from models import init_model
from dataset import BiasedHDPDataset, BiasedSPDDataset, BiasedMNISTDataset


def run(args):
    if args.dataset == 'spd':
        dataset = BiasedSPDDataset(args)
    elif args.dataset == 'hdp':
        dataset = BiasedHDPDataset(args)
    elif args.dataset == 'mnist':
        dataset = BiasedMNISTDataset(args)
    else:
        raise Exception("Unknown dataset.")

    for al_iter in range(args.al_iters + 1):
        print(f"AL iteration {al_iter}")

        model = init_model(args)

        train_model(model, dataset, args)

        eval(model, dataset, args)

        if al_iter < args.al_iters:
            proposed_data_indices = active_learning.run(model, dataset, args)
            
            dataset.update(proposed_data_indices)


if __name__ == '__main__':
    # parse arguments: 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hdp', help='dataset name')
    parser.add_argument('--model', type=str, default='lr', help='model name')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--al-iters', type=int, default=1, help='number of loops of active learning')
    parser.add_argument('--al-method', type=str, default='random', help='active learning method')
    parser.add_argument('--al-proposal-size', type=int, default=500, help='number of unlabeled data to propose')

    parser.add_argument('--protected-feature', type=str, default='Sex', help='protected feature to balance')
    parser.add_argument('--feature-to-predict', type=str, default='HeartDisease', help='feature to predict')
    parser.add_argument('--feature-distribution', nargs='+', default=[0.5, 0.5], help='redistributed partition')
    parser.add_argument('--equalize-dataset', action='store_true', help='equalize dataset')
    parser.add_argument('--dataset-split', type=float, default=0.9, help='dataset split')

    parser.add_argument('--train-val-split', type=float, default=0.8, help='train/val split')
    parser.add_argument('--val-test-split', type=float, default=0.5, help='val/test split')

    args = parser.parse_args()

    run(args)