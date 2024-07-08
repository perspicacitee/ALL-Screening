import os.path

import torch
import numpy as np


class EarlyStopping(object):
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, path, patience=7, verbose=False, delta=0.0001, ascending=True):
        self.path = path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta  # if ascending else -delta
        self.ascending = ascending

        if not os.path.exists(path):
            os.makedirs(path)

    def __call__(self, val_loss, model):
        print("val_loss={}".format(val_loss))
        if self.ascending:
            score = val_loss
        else:
            score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, self.path)
        # count the times about current score exceed best score
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}; loss {self.val_loss_min:.6f} --> {val_loss:.6f}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = np.sqrt(self.best_score * score) * (-1 if self.best_score < 0 else 1)
            self.save_checkpoint(val_loss, model, self.path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path: str):
        if self.verbose:
            print(
                f"Validation loss improved ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), path + '/' + "model_checkpoint.pth")
        self.val_loss_min = val_loss
