import lzma
import os
import csv
import shutil
from tempfile import mkdtemp
import logging

from unittest.mock import patch

from deeprank_gnn.models.metrics import (MetricsExporterCollection,
                                         TensorboardBinaryClassificationExporter,
                                         OutputExporter)

_log = logging.getLogger(__name__)


class TestMetrics:
    def setUp(self):
        self._work_dir = mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._work_dir)

    def test_collection(self):
        exporters = [TensorboardBinaryClassificationExporter(self._work_dir),
                     OutputExporter(self._work_dir)]

        collection = MetricsExporterCollection(*exporters)

        pass_name = "test"
        epoch_number = 0

        entry_names = ["entry1", "entry2", "entry3"]
        outputs = [[0.2, 0.1], [0.3, 0.8], [0.8, 0.9]]
        targets = [0, 1, 1]

        with collection:
            collection.process(pass_name, epoch_number,
                               entry_names,
                               outputs,
                               targets)

        assert len(os.listdir(self._work_dir)) == 2  # tensorboard & table

    def test_output_table(self):
        output_exporter = OutputExporter(self._work_dir)

        pass_name = "test"
        epoch_number = 0

        entry_names = ["entry1", "entry2", "entry3"]
        outputs = [[0.2, 0.1], [0.3, 0.8], [0.8, 0.9]]
        targets = [0, 1, 1]

        with output_exporter:
            output_exporter.process(pass_name, epoch_number,
                                    entry_names,
                                    outputs,
                                    targets)

        with lzma.open(output_exporter.get_filename(pass_name, epoch_number), 'rt', newline='\n') as table_file:
            r = csv.reader(table_file, delimiter=',')
            header = next(r)
            columns = {name: [] for name in header}
            for row in r:
                for column_index, column_name in enumerate(header):
                    columns[column_name].append(row[column_index])

        assert columns["entry"] == entry_names, f"{columns['entry']} != {entry_names}"
        assert columns["output"] == [str(z) for z in outputs], f"columns['output'] != {outputs}"
        assert columns["target"] == [str(y) for y in targets], f"columns['target'] != {targets}"

    @patch("torch.utils.tensorboard.SummaryWriter.add_scalar")
    def test_tensorboard(self, mock_add_scalar):
        tensorboard_exporter = TensorboardBinaryClassificationExporter(self._work_dir)

        pass_name = "test"
        epoch_number = 0

        entry_names = ["entry1", "entry2", "entry3"]
        outputs = [[0.2, 0.1], [0.3, 0.8], [0.8, 0.9]]
        targets = [0, 1, 1]

        def _check_scalar(name, scalar, timestep):
            if name == f"{pass_name} cross entropy loss":
                assert scalar < 1.0
            else:
                assert scalar == 1.0

        mock_add_scalar.side_effect = _check_scalar

        with tensorboard_exporter:
            tensorboard_exporter.process(pass_name, epoch_number,
                                         entry_names, outputs, targets)
        assert mock_add_scalar.called
