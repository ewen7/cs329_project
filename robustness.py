import numpy as np
import torch

class MNISTRobustness(object):
    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset
        self.X_test, self.y_test = self.dataset.get_xy_split('test')

    def accuracy(self, X_adv):
        y_hat = torch.from_numpy(self.model.predict(self.X_test))
        y_adv_hat = torch.from_numpy(self.model.predict(X_adv))
        
        test_acc = (y_hat == self.y_test).sum().item() / self.y_test.shape[0]
        adv_acc = (y_adv_hat == self.y_test).sum().item() / self.y_test.shape[0]
        
        print(f"robustness accuracies: (test) {test_acc}  (adv) {adv_acc}")
        return test_acc, adv_acc

# pass in trained model and perturbated images
