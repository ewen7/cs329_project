import argparse
import torch
import numpy as np
from matplotlib import pyplot as plt
import active_learning
from train import train
from eval_fairness import eval_fairness
import models
import dataset


def prep_dataset(args):
    """
    Based on the value of args.dataset (either 'MNIST' or 'SPD'), prepares an initial Dataset object.

    This object should contain a biased labeled portion and a remaining unlabeled portion.
    """
    # TODO
    dataset = None
    # get from dataset/*.py
    return dataset
    

def init_model(args):
    """
    Initializes a model based on the value of args.model (either 'LR' or 'NN'?)
    """
    # TODO
    model = None
    # get from models.py
    return model


def run(args):
    dataset = prep_dataset(args)

    for al_iter in range(args.al_iters):
        print(f"AL iteration {al_iter}")

        model = init_model(args)

        train(model, dataset, args)

        eval_fairness(model, dataset, args)

        proposed_data = active_learning.run(model, dataset, args)
        
        dataset.update(proposed_data)


if __name__ == '__main__':
    # parse arguments: 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='spd', help='dataset name')
    parser.add_argument('--al-iters', type=int, default=1, help='number of loops of active learning')
    args = parser.parse_args()

    run(args)