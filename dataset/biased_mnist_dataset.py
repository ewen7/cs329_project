import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST
import os


dataset_dir = os.path.join(os.path.dirname(os.path.realpath(
    __file__)), 'MNIST/')

class BiasedMNISTDataset(Dataset):

    def __init__(self, args, dataset_dir=dataset_dir):
        datasets = MNIST(dataset_dir=dataset_dir, train=True, download=True)
        breakpoint()
    
