import logging
import os
import random
from math import sqrt

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from torch import argmax, tensor
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter

_log = logging.getLogger(__name__)


class OutputExporter:
    """The class implements a general exporter to be called when a neural network generates outputs."""

    def __init__(self, directory_path: str | None = None):
        if directory_path is None:
            directory_path = "./output"
        self._directory_path = directory_path

        if not os.path.exists(self._directory_path):
            os.makedirs(self._directory_path)

    def __enter__(self):
        """Overridable."""
        return self

    def __exit__(self, exception_type, exception, traceback):  # noqa: ANN001
        """Overridable."""

    def process(
        self,
        pass_name: str,
        epoch_number: int,
        entry_names: list[str],
        output_values: list,
        target_values: list,
        loss: float,
    ) -> None:
        """The entry_names, output_values, target_values MUST have the same length."""

    def is_compatible_with(
        self,
        output_data_shape: int,  # noqa: ARG002
        target_data_shape: int | None = None,  # noqa: ARG002
    ) -> bool:
        """True if this exporter can work with the given data shapes."""
        return True


class OutputExporterCollection:
    """It allows a series of output exporters to be used at the same time."""

    def __init__(self, *args: list[OutputExporter]):
        self._output_exporters = args

    def __enter__(self):
        for output_exporter in self._output_exporters:
            output_exporter.__enter__()

        return self

    def __exit__(self, exception_type, exception, traceback):  # noqa: ANN001
        for output_exporter in self._output_exporters:
            output_exporter.__exit__(exception_type, exception, traceback)

    def process(
        self,
        pass_name: str,
        epoch_number: int,
        entry_names: list[str],
        output_values: list,
        target_values: list,
        loss: float,
    ) -> None:
        for output_exporter in self._output_exporters:
            output_exporter.process(
                pass_name,
                epoch_number,
                entry_names,
                output_values,
                target_values,
                loss,
            )

    def __iter__(self):
        return iter(self._output_exporters)


class TensorboardBinaryClassificationExporter(OutputExporter):
    """Exporter for tensorboard, works for binary classification only.

    Currently outputs to tensorboard:
    - Mathews Correlation Coefficient (MCC)
    - Accuracy
    - ROC area under the curve
    Outputs are computed for each epoch.
    """

    def __init__(self, directory_path: str):
        super().__init__(directory_path)
        self._writer = SummaryWriter(log_dir=directory_path)

    def __enter__(self):
        self._writer.__enter__()
        return self

    def __exit__(self, exception_type, exception, traceback):  # noqa: ANN001
        self._writer.__exit__(exception_type, exception, traceback)

    def process(
        self,
        pass_name: str,
        epoch_number: int,
        entry_names: list[str],
        output_values: list,
        target_values: list,
        loss: float,  # noqa: ARG002
    ) -> None:
        """Write to tensorboard."""
        ce_loss = cross_entropy(tensor(output_values), tensor(target_values)).item()
        self._writer.add_scalar(
            f"{pass_name} cross entropy loss",
            ce_loss,
            epoch_number,
        )

        probabilities = []
        fp, fn, tp, tn = 0, 0, 0, 0
        for entry_index, _ in enumerate(entry_names):
            probability = output_values[entry_index][1]
            probabilities.append(probability)

            prediction_value = argmax(tensor(output_values[entry_index]))
            target_value = target_values[entry_index]

            if prediction_value > 0 and target_value > 0:
                tp += 1

            elif prediction_value <= 0 and target_value <= 0:
                tn += 1

            elif target_value <= 0 < prediction_value:
                fp += 1

            elif prediction_value <= 0 < target_value:
                fn += 1

        mcc_numerator = tn * tp - fp * fn
        if mcc_numerator == 0:
            self._writer.add_scalar(f"{pass_name} MCC", 0.0, epoch_number)
        else:
            mcc_denominator = sqrt((tn + fn) * (fp + tp) * (tn + fp) * (fn + tp))

            if mcc_denominator != 0:
                mcc = mcc_numerator / mcc_denominator
                self._writer.add_scalar(f"{pass_name} MCC", mcc, epoch_number)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        self._writer.add_scalar(f"{pass_name} accuracy", accuracy, epoch_number)

        # for ROC curves to work, we need both class values in the set
        if len(set(target_values)) == 2:  # noqa:PLR2004
            roc_auc = roc_auc_score(target_values, probabilities)
            self._writer.add_scalar(f"{pass_name} ROC AUC", roc_auc, epoch_number)

    def is_compatible_with(
        self,
        output_data_shape: int,
        target_data_shape: int | None = None,
    ) -> bool:
        """For regression, target data is needed and output data must be a list of two-dimensional values."""
        return output_data_shape == 2 and target_data_shape == 1  # noqa:PLR2004


class ScatterPlotExporter(OutputExporter):
    """An output exporter that can make scatter plots, containing every single data point.

    On the X-axis: targets values
    On the Y-axis: output values

    Args:
        directory_path: Where to store the plots.
        epoch_interval: How often to make a plot, 5 means: every 5 epochs. Defaults to 1.
    """

    def __init__(self, directory_path: str, epoch_interval: int = 1):
        super().__init__(directory_path)
        self._epoch_interval = epoch_interval

    def __enter__(self):
        self._plot_data = {}
        return self

    def __exit__(self, exception_type, exception, traceback):  # noqa: ANN001
        self._plot_data.clear()

    def get_filename(self, epoch_number: int) -> str:
        """Returns the filename for the table."""
        return os.path.join(self._directory_path, f"scatter-{epoch_number}.png")

    @staticmethod
    def _get_color(pass_name: str) -> str:
        pass_name = pass_name.lower().strip()
        if pass_name in ("train", "training"):
            return "blue"
        if pass_name in ("eval", "valid", "validation"):
            return "red"
        if pass_name == ("test", "testing"):
            return "green"
        return random.choice(["yellow", "cyan", "magenta"])

    @staticmethod
    def _plot(
        epoch_number: int,
        data: dict[str, tuple[list[float], list[float]]],
        png_path: str,
    ) -> None:
        plt.title(f"Epoch {epoch_number}")

        for pass_name, (truth_values, prediction_values) in data.items():
            plt.scatter(
                truth_values,
                prediction_values,
                color=ScatterPlotExporter._get_color(pass_name),
                label=pass_name,
            )

        plt.xlabel("truth")
        plt.ylabel("prediction")

        plt.legend()
        plt.savefig(png_path)
        plt.close()

    def process(
        self,
        pass_name: str,
        epoch_number: int,
        entry_names: list[str],  # noqa: ARG002
        output_values: list,
        target_values: list,
        loss: float,  # noqa: ARG002
    ) -> None:
        """Make the plot, if the epoch matches with the interval."""
        if epoch_number % self._epoch_interval == 0:
            if epoch_number not in self._plot_data:
                self._plot_data[epoch_number] = {}

            self._plot_data[epoch_number][pass_name] = (target_values, output_values)

            path = self.get_filename(epoch_number)
            self._plot(epoch_number, self._plot_data[epoch_number], path)

    def is_compatible_with(
        self,
        output_data_shape: int,
        target_data_shape: int | None = None,
    ) -> bool:
        """For regression, target data is needed and output data must be a list of one-dimensional values."""
        return output_data_shape == 1 and target_data_shape == 1


class HDF5OutputExporter(OutputExporter):
    """An output exporter that saves every single data point in an hdf5 file.

    It is the most general output exporter implemented, and the information
    contained in the hdf5 file generated allows the user to compute any kind
    of metrics, for both classification and regression.
    Results saved are:
    - phase (train/valid/test)
    - epoch
    - entry name
    - output value/s
    - target value
    - loss per epoch
    The user can then read the content of the hdf5 file into a Pandas dataframe.
    """

    def __init__(self, directory_path: str):
        self.phase = None
        super().__init__(directory_path)

    def __enter__(self):
        self.d = {
            "phase": [],
            "epoch": [],
            "entry": [],
            "output": [],
            "target": [],
            "loss": [],
        }
        self.df = pd.DataFrame(data=self.d)

        return self

    def __exit__(self, exception_type, exception, traceback):  # noqa: ANN001
        if self.phase is not None:
            if self.phase == "validation":
                self.phase = "training"

            self.df.to_hdf(
                os.path.join(self._directory_path, "output_exporter.hdf5"),
                key=self.phase,
                mode="a",
            )

    def process(
        self,
        pass_name: str,
        epoch_number: int,
        entry_names: list[str],
        output_values: list,
        target_values: list,
        loss: float,
    ) -> None:
        self.phase = pass_name
        pass_name = [pass_name] * len(output_values)
        loss = [loss] * len(output_values)
        epoch_number = [epoch_number] * len(output_values)

        d_epoch = {
            "phase": pass_name,
            "epoch": epoch_number,
            "entry": entry_names,
            "output": output_values,
            "target": target_values,
            "loss": loss,
        }
        df_epoch = pd.DataFrame(data=d_epoch)

        self.df = pd.concat([self.df, df_epoch])
        self.df = self.df.reset_index(drop=True)
