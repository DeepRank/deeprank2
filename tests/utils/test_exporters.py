import os
import shutil
import logging
import unittest
import pandas as pd
from tempfile import mkdtemp
from unittest.mock import patch
import h5py
from deeprankcore.utils.exporters import (
    OutputExporterCollection,
    TensorboardBinaryClassificationExporter,
    ScatterPlotExporter,
    HDF5OutputExporter
)

logging.getLogger(__name__)


class TestOutputExporters(unittest.TestCase):
    def setUp(self):
        self._work_dir = mkdtemp()

    def tearDown(self):
        shutil.rmtree(self._work_dir)

    def test_collection(self):
        exporters = [
            TensorboardBinaryClassificationExporter(self._work_dir),
            HDF5OutputExporter(self._work_dir),
        ]

        collection = OutputExporterCollection(*exporters)

        pass_name = "test"
        epoch_number = 0

        entry_names = ["entry1", "entry2", "entry3"]
        outputs = [[0.2, 0.1], [0.3, 0.8], [0.8, 0.9]]
        targets = [0, 1, 1]
        loss = 0.1

        with collection:
            collection.process(pass_name, epoch_number, entry_names, outputs, targets, loss)

        assert len(os.listdir(self._work_dir)) == 2  # tensorboard & table

    @patch("torch.utils.tensorboard.SummaryWriter.add_scalar")
    def test_tensorboard_binary_classif(self, mock_add_scalar):
        tensorboard_exporter = TensorboardBinaryClassificationExporter(self._work_dir)

        pass_name = "test"
        epoch_number = 0

        entry_names = ["entry1", "entry2", "entry3"]
        outputs = [[0.2, 0.1], [0.3, 0.8], [0.8, 0.9]]
        targets = [0, 1, 1]
        loss = 0.1

        def _check_scalar(name, scalar, timestep): # pylint: disable=unused-argument
            if name == f"{pass_name} cross entropy loss":
                assert scalar < 1.0
            else:
                assert scalar == 1.0

        mock_add_scalar.side_effect = _check_scalar

        with tensorboard_exporter:
            tensorboard_exporter.process(
                pass_name, epoch_number, entry_names, outputs, targets, loss
            )
        assert mock_add_scalar.called

    def test_scatter_plot(self):
        scatterplot_exporter = ScatterPlotExporter(self._work_dir)

        epoch_number = 0

        with scatterplot_exporter:
            scatterplot_exporter.process(
                "train",
                epoch_number,
                ["entry1", "entry1", "entry2"],
                [0.1, 0.65, 0.98],
                [0.0, 0.5, 1.0],
                0.1
            )

            scatterplot_exporter.process(
                "valid",
                epoch_number,
                ["entryA", "entryB", "entryC"],
                [0.3, 0.35, 0.25],
                [0.0, 0.5, 1.0],
                0.1
            )

        assert os.path.isfile(scatterplot_exporter.get_filename(epoch_number))

    def test_hdf5_output(self):
        output_exporter = HDF5OutputExporter(self._work_dir)
        path_output_exporter = os.path.join(self._work_dir, 'output_exporter.hdf5')
        entry_names = ["entry1", "entry2", "entry3"]
        outputs = [[0.2, 0.1], [0.3, 0.8], [0.8, 0.9]]
        targets = [0, 1, 1]
        loss = 0.1

        pass_name_1 = "test_1"
        n_epoch_1 = 10
        with output_exporter:
            for epoch_number in range(n_epoch_1):
                output_exporter.process(
                    pass_name_1, epoch_number, entry_names, outputs, targets, loss
                )

        pass_name_2 = "test_2"
        n_epoch_2 = 5
        with output_exporter:
            for epoch_number in range(n_epoch_2):
                output_exporter.process(
                    pass_name_2, epoch_number, entry_names, outputs, targets, loss
                )

        df_test_1 = pd.read_hdf(
            path_output_exporter,
            key=pass_name_1)
        df_test_2 = pd.read_hdf(
            path_output_exporter,
            key=pass_name_2)

        df_hdf5 = h5py.File(path_output_exporter,'r')
        df_keys = list(df_hdf5.keys())
        df_keys.sort()
        # assert that the hdf5 output file contains exactly 2 Groups, test_1 and test_2
        assert df_keys == ["test_1", "test_2"]
        df_hdf5.close()
        # assert there is one row for each epoch
        assert len(df_test_1.epoch.unique()) == n_epoch_1
        assert len(df_test_2.epoch.unique()) == n_epoch_2
        # assert entry column contains entry_names
        assert list(df_test_1.entry.unique()) == entry_names
        assert list(df_test_2.entry.unique()) == entry_names
        # assert there are len(entry_names) rows for each epoch
        assert df_test_1[df_test_1.phase == pass_name_1].groupby(['epoch'], as_index=False).count().phase.unique() == len(entry_names)
        assert df_test_2[df_test_2.phase == pass_name_2].groupby(['epoch'], as_index=False).count().phase.unique() == len(entry_names)
        # assert there are len(entry_names)*n_epoch rows
        assert df_test_1[df_test_1.phase == pass_name_1].shape[0] == len(entry_names)*n_epoch_1
        assert df_test_2[df_test_2.phase == pass_name_2].shape[0] == len(entry_names)*n_epoch_2
        # assert there are 6 columns ('phase', 'epoch', 'entry', 'output', 'target', 'loss')
        assert df_test_1[df_test_1.phase == pass_name_1].shape[1] == 6
        assert df_test_2[df_test_2.phase == pass_name_2].shape[1] == 6
