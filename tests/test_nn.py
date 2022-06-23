import tempfile
import shutil
import os
import unittest
import pytest
from deeprankcore.NeuralNet import NeuralNet
from deeprankcore.DataSet import HDF5DataSet
from deeprankcore.ginet import GINet
from deeprankcore.foutnet import FoutNet
from deeprankcore.sGAT import sGAT
from deeprankcore.models.metrics import (
    OutputExporter,
    TensorboardBinaryClassificationExporter,
    ScatterPlotExporter,
)


def _model_base_test( # pylint: disable=too-many-arguments
    hdf5_path,
    model_class,
    node_features,
    edge_features,
    task,
    target,
    metrics_exporters,
    transform_sigmoid,
):

    dataset = HDF5DataSet(
        root="./",
        database=hdf5_path,
        index=None,
        node_feature=node_features,
        edge_feature=edge_features,
        target=target,
        clustering_method='mcl',
    )

    nn = NeuralNet(
        dataset,
        model_class,
        task=task,
        batch_size=64,
        percent=[0.8, 0.2],
        metrics_exporters=metrics_exporters,
        transform_sigmoid=transform_sigmoid,
    )

    nn.train(nepoch=10, validate=True)

    nn.save_model("test.pth.tar")

    NeuralNet(dataset, model_class, pretrained_model="test.pth.tar")


class TestNeuralNet(unittest.TestCase):
    @classmethod
    def setUpClass(class_):
        class_.work_directory = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(class_):
        shutil.rmtree(class_.work_directory)

    def test_ginet_sigmoid(self):
        _model_base_test(
            "tests/hdf5/1ATN_ppi.hdf5",
            GINet,
            ["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            ["dist"],
            "reg",
            "irmsd",
            [OutputExporter(self.work_directory)],
            True,
        )

    def test_ginet(self):
        _model_base_test(
            "tests/hdf5/1ATN_ppi.hdf5",
            GINet,
            ["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            ["dist"],
            "reg",
            "irmsd",
            [OutputExporter(self.work_directory)],
            False,
        )

        assert len(os.listdir(self.work_directory)) > 0

    def test_ginet_class(self):
        _model_base_test(
            "tests/hdf5/variants.hdf5",
            GINet,
            ["size", "polarity", "sasa", "ic", "pssm"],
            ["dist"],
            "class",
            "bin_class",
            [TensorboardBinaryClassificationExporter(self.work_directory)],
            False,
        )

        assert len(os.listdir(self.work_directory)) > 0

    def test_fout(self):
        _model_base_test(
            "tests/hdf5/1ATN_ppi.hdf5",
            FoutNet,
            ["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            ["dist"],
            "reg",
            "irmsd",
            [],
            False,
        )

    def test_sgat(self):
        _model_base_test(
            "tests/hdf5/1ATN_ppi.hdf5",
            sGAT,
            ["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            ["dist"],
            "reg",
            "irmsd",
            [],
            False,
        )

    def test_incompatible_regression(self):
        with pytest.raises(ValueError):
            _model_base_test(
                "tests/hdf5/1ATN_ppi.hdf5",
                sGAT,
                ["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
                ["dist"],
                "reg",
                "irmsd",
                [TensorboardBinaryClassificationExporter(self.work_directory)],
                False,
            )

    def test_incompatible_classification(self):
        with pytest.raises(ValueError):
            _model_base_test(
                "tests/hdf5/variants.hdf5",
                GINet,
                ["size", "polarity", "sasa", "ic", "pssm"],
                ["dist"],
                "class",
                "bin_class",
                [ScatterPlotExporter(self.work_directory)],
                False,
            )


if __name__ == "__main__":
    unittest.main()
