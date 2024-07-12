from collections.abc import Callable


class EarlyStopping:
    """Terminate training upon trigger.

    Triggered if validation loss doesn't improve after a given patience or if a maximum gap between validation and training loss is reached.

    Args:
        patience: How long to wait after last time validation loss improved. Defaults to 10.
        delta: Minimum change required to reset the early stopping counter. Defaults to 0.
        maxgap: Maximum difference between between training and validation loss. Defaults to None.
        min_epoch: Minimum epoch to be reached before looking at maxgap. Defaults to 10.
        verbose: If True, prints a message for each validation loss improvement. Defaults to True.
        trace_func: Function used for recording EarlyStopping status. Defaults to print.
    """

    def __init__(
        self,
        patience: int = 10,
        delta: float = 0,
        maxgap: float | None = None,
        min_epoch: int = 10,
        verbose: bool = True,
        trace_func: Callable = print,
    ):
        self.patience = patience
        self.delta = delta
        self.maxgap = maxgap
        self.min_epoch = min_epoch
        self.verbose = verbose
        self.trace_func = trace_func

        self.early_stop = False
        self.counter = 0
        self.best_score = None
        self.val_loss_min = None

    def __call__(  # noqa: C901
        self,
        epoch: int,
        val_loss: float,
        train_loss: float | None = None,
    ):
        score = -val_loss

        # initialize
        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss

        # check patience
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                extra_trace = ""
                if self.delta:
                    extra_trace = f"more than {self.delta} "
                self.trace_func(
                    f"Validation loss did not decrease {extra_trace}({self.val_loss_min:.6f} --> {val_loss:.6f}). "
                    f"EarlyStopping counter: {self.counter} out of {self.patience}",
                )
            if self.counter >= self.patience:
                self.trace_func(f"EarlyStopping activated at epoch # {epoch} because patience of {self.patience} has been reached.")
                self.early_stop = True
        else:
            if self.verbose:
                self.trace_func(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).")
            self.best_score = score
            self.counter = 0

        if score >= self.best_score:
            self.best_score = score
            self.val_loss_min = val_loss

        # check maxgap
        if self.maxgap and epoch > self.min_epoch:
            if train_loss is None:
                msg = "Cannot compute gap because no train_loss is provided to EarlyStopping."
                raise ValueError(msg)
            gap = val_loss - train_loss
            if gap > self.maxgap:
                self.trace_func(
                    f"EarlyStopping activated at epoch # {epoch} due to overfitting. "
                    f"The difference between validation and training loss of {gap} exceeds the maximum allowed ({self.maxgap})",
                )
                self.early_stop = True


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
