from typing import Optional
import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(
        self,
        patience: int = 10,
        delta: Optional[float] = None,
        max_gap: Optional[float] = None,
        verbose: bool = True,
        path: str = 'checkpoint.pt',
        trace_func: function = print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: None
            max_gap (float, optional): Maximum difference between between training and validation loss.
                            Default: None
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: True
            path (str): Path for the checkpoint saving. Ignored if no model is passed.
                            Default: 'checkpoint.pt'
            trace_func (function): Trace print function.
                            Default: print            
        """

        self.patience = patience
        if delta is None:
            self.delta = 0
        else:
            self.delta = delta
        self.max_gap = max_gap
        self.verbose = verbose
        self.path = path
        self.trace_func = trace_func

        self.early_stop = False
        self.counter = 0
        self.best_score = None
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, train_loss=None, model=None):
        """Set `model=None` if model is saved elsewhere"""
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.max_gap:
            if train_loss is None:
                raise ValueError("cannot compute gap because no train_loss is provided to EarlyStopping")
            gap = val_loss - train_loss
            if gap > self.max_gap:
                self.trace_func(f'EarlyStopping activated because difference between validation and training loss exceeds max_gap of {self.max_gap}')
                self.early_stop = True


        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            if model:
                self.trace_func('\tSaving model...')
                torch.save(model.state_dict(), self.path)
                self.val_loss_min = val_loss



# This module is modified from https://github.com/Bjarten/early-stopping-pytorch, under the following license:


# MIT License

# Copyright (c) 2018 Bjarte Mehus Sunde

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
