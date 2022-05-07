import lzma
import os
import csv
from typing import List, Tuple, Any
from math import sqrt
import logging

from torch import argmax, tensor
from torch.nn.functional import cross_entropy
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score


_log = logging.getLogger(__name__)


class MetricsExporter:
    "The class implements an object, to be called when a neural network generates output"

    def __enter__(self):
        "overridable"
        return self

    def __exit__(self, exception_type, exception, traceback):
        "overridable"
        pass

    def process(self, pass_name: str, epoch_number: int,
                entry_names: List[str], output_values: List[Any], target_values: List[Any]):
        "the entry_names, output_values, target_values MUST have the same length"
        pass


class MetricsExporterCollection:
    "allows a series of metrics exporters to be used at the same time"

    def __init__(self, *args: Tuple[MetricsExporter]):
        self._metrics_exporters = args

    def __enter__(self):
        for metrics_exporter in self._metrics_exporters:
            metrics_exporter.__enter__()

        return self

    def __exit__(self, exception_type, exception, traceback):
        for metrics_exporter in self._metrics_exporters:
            metrics_exporter.__exit__(exception_type, exception, traceback)

    def process(self, pass_name: str, epoch_number: int,
                entry_names: List[str], output_values: List[Any], target_values: List[Any]):
        for metrics_exporter in self._metrics_exporters:
            metrics_exporter.process(pass_name, epoch_number, entry_names, output_values, target_values)


class TensorboardBinaryClassificationExporter(MetricsExporter):
    """ Exports to tensorboard, works for binary classification only.

        Currently outputs to tensorboard:
         - Mathews Correlation Coefficient (MCC)
         - Accuracy
         - ROC area under the curve

        Outputs are done per epoch.
    """

    def __init__(self, directory_path: str):
        self._directory_path = directory_path
        self._writer = SummaryWriter(log_dir=directory_path)

    def __enter__(self):
        self._writer.__enter__()
        return self

    def __exit__(self, exception_type, exception, traceback):
        self._writer.__exit__(exception_type, exception, traceback)

    def process(self, pass_name: str, epoch_number: int,
                entry_names: List[str], output_values: List[Any], target_values: List[Any]):
        "write to tensorboard"

        loss = cross_entropy(tensor(output_values), tensor(target_values))
        self._writer.add_scalar(f"{pass_name} loss", loss, epoch_number)

        probabilities = []
        fp, fn, tp, tn = 0, 0, 0, 0
        for entry_index, entry_name in enumerate(entry_names):
            probability = output_values[entry_index][1]
            probabilities.append(probability)

            prediction_value = argmax(tensor(output_values[entry_index]))
            target_value = target_values[entry_index]

            if prediction_value > 0.0 and target_value > 0.0:
                tp += 1

            elif prediction_value <= 0.0 and target_value <= 0.0:
                tn += 1

            elif prediction_value > 0.0 and target_value <= 0.0:
                fp += 1

            elif prediction_value <= 0.0 and target_value > 0.0:
                fn += 1

        mcc_numerator = tn * tp - fp * fn
        if mcc_numerator == 0.0:
             self._writer.add_scalar(f"{pass_name} MCC", 0.0, epoch_number)
        else:
            mcc_denominator = sqrt((tn + fn) * (fp + tp) * (tn + fp) * (fn + tp))

            if mcc_denominator != 0.0:
                mcc = mcc_numerator / mcc_denominator
                self._writer.add_scalar(f"{pass_name} MCC", mcc, epoch_number)

        accuracy = (tp + tn) / (tp + tn + fp + fn)
        self._writer.add_scalar(f"{pass_name} accuracy", accuracy, epoch_number)

        # for ROC curves to work, we need both class values in the set
        if len(set(target_values)) == 2:
            roc_auc = roc_auc_score(target_values, probabilities)
            self._writer.add_scalar(f"{pass_name} ROC AUC", roc_auc, epoch_number)


class OutputExporter(MetricsExporter):
    """ A metrics exporter that writes CSV output tables, containing every single data point

        Included are:
            - entry names
            - output values
            - target values

        The user can use these output tables to make a scatter plot for a particular epoch.

        Outputs are done per epoch.
    """

    def __init__(self, directory_path: str):
        self._directory_path = directory_path

    def get_filename(self, pass_name, epoch_number):
        "returns the filename for the table"
        return os.path.join(self._directory_path, f"output-{pass_name}-epoch-{epoch_number}.csv.xz")

    def process(self, pass_name: str, epoch_number: int,
                entry_names: List[str], output_values: List[Any], target_values: List[Any]):
        "write the output to the table"

        with lzma.open(self.get_filename(pass_name, epoch_number), 'wt', newline='\n') as f:
            w = csv.writer(f, delimiter=',')

            w.writerow(["entry", "output", "target"])

            for entry_index, entry_name in enumerate(entry_names):
                output_value = output_values[entry_index]
                target_value = target_values[entry_index]

                _log.debug(f"writerow [{entry_name}, {output_value}, {target_value}]")

                w.writerow([entry_name, str(output_value), str(target_value)])
