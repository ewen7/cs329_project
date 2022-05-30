import numpy as np
import torch


def gaussian_perturb(model, dataset, stds, verbose=False):
    X_test, y_test = dataset.get_xy_split('test')
    results = np.zeros(len(stds))
    for i, std in enumerate(stds):
        perturbation = torch.normal(mean=0.0, std=std, size=X_test.shape)
        X_adv = torch.clamp(X_test + perturbation, min=0., max=255.)
        y_hat = torch.from_numpy(model.predict(X_adv))
        results[i] = (y_hat == y_test).sum().item() / y_test.shape[0]
        if verbose > 0: print(f"{std:.2f} {results[i]:.4f}")
    return results