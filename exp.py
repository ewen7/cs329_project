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

        eval(model, dataset, len(dataset), args)

        print()

        if al_iter < args.al_iters:
            proposed_data_indices = active_learning.run(model, dataset, args)
            
            dataset.update(proposed_data_indices)


if __name__ == '__main__':
    # parse arguments: 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='hdp', help='dataset name')
    parser.add_argument('--model', type=str, default='lr', help='model name')
    parser.add_argument('--name', type=str, default='exp', help='experiment name')

    # al params
    parser.add_argument('--al-iters', type=int, default=100, help='number of loops of active learning')
    parser.add_argument('--al-method', type=str, default='random', help='active learning method')
    parser.add_argument('--al-proposal-size', type=int, default=100, help='number of unlabeled data to propose')
    parser.add_argument('--al-sampling', type=str, default='top', help='sampling method for active learning')
    parser.add_argument('--kappa', type=float, default=2.0, help='active learning weighted sampling pre-softmax scaling')

    parser.add_argument('--protected-feature', type=str, default='Sex', help='protected feature to balance')
    parser.add_argument('--feature-to-predict', type=str, default='HeartDisease', help='feature to predict')
    parser.add_argument('--feature-distribution', nargs='+', default=[], help='redistributed partition')

    parser.add_argument('--dataset-split', type=float, default=0.01, help='dataset split')
    parser.add_argument('--verbose', action='store_true', help='Verbose Output dataset')

    # hdp params
    parser.add_argument('--train-val-split', type=float, default=0.8, help='train/val split')
    parser.add_argument('--val-test-split', type=float, default=0.5, help='val/test split')
    parser.add_argument('--equalize-dataset', action='store_true', help='equalize dataset')

    # mnist cnn training params
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=20, help='batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='momentum')
    parser.add_argument('--adv-type', type=str, default='gaussian', help='method of generating adversarial data')

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

    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+args.dataset+"-"+args.model+"-"+args.al_method+"-"+args.name)
    args.summary_writer = Logger(log_dir)

    run(args)