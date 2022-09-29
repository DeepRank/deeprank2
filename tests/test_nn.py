import tempfile
import shutil
import os
import unittest
import pytest
import logging

import torch

from deeprankcore.NeuralNet import NeuralNet
from deeprankcore.DataSet import HDF5DataSet
from deeprankcore.ginet import GINet
from deeprankcore.foutnet import FoutNet
from deeprankcore.naive_gnn import NaiveNetwork
from deeprankcore.sGAT import sGAT
from deeprankcore.models.metrics import (
    OutputExporter,
    TensorboardBinaryClassificationExporter,
    ScatterPlotExporter
)


_log = logging.getLogger(__name__)


def _model_base_test( # pylint: disable=too-many-arguments, too-many-locals
    train_hdf5_path,
    val_hdf5_path,
    test_hdf5_path,
    model_class,
    node_features,
    edge_features,
    task,
    target,
    metrics_exporters,
    transform_sigmoid,
    clustering_method
):

    dataset_train = HDF5DataSet(
        hdf5_path=train_hdf5_path,
        root="./",
        node_feature=node_features,
        edge_feature=edge_features,
        target=target,
        task=task,
        clustering_method=clustering_method)

    if val_hdf5_path is not None:
        dataset_val = HDF5DataSet(
            hdf5_path=val_hdf5_path,
            root="./",
            node_feature=node_features,
            edge_feature=edge_features,
            target=target,
            task=task,
            clustering_method=clustering_method)
    else:
        dataset_val = None

    if test_hdf5_path is not None:
        dataset_test = HDF5DataSet(
            hdf5_path=test_hdf5_path,
            root="./",
            node_feature=node_features,
            edge_feature=edge_features,
            target=target,
            task=task,
            clustering_method=clustering_method)
    else:
        dataset_test = None

    nn = NeuralNet(
        model_class,
        dataset_train,
        dataset_val,
        dataset_test,
        batch_size=64,
        weight_decay=0.01,
        metrics_exporters=metrics_exporters,
        transform_sigmoid=transform_sigmoid,
    )

    if torch.cuda.is_available():
        _log.debug("cuda is available, testing that the model is cuda")
        for parameter in nn.model.parameters():
            assert parameter.is_cuda, f"{parameter} is not cuda"

        data = dataset_train.get(0)

        for name, data_tensor in (("x", data.x), ("y", data.y),
                                  ("edge_index", data.edge_index),
                                  ("edge_attr", data.edge_attr),
                                  ("pos", data.pos),
                                  ("cluster0",data.cluster0),
                                  ("cluster1", data.cluster1)):

            if data_tensor is not None:
                assert data_tensor.is_cuda, f"data.{name} is not cuda"
    else:
        _log.debug("cuda is not available")

    nn.train(nepoch=10, validate=True)

    nn.save_model("test.pth.tar")

    NeuralNet(
        model_class,
        dataset_train,
        dataset_val,
        dataset_test,
        pretrained_model="test.pth.tar")

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
            "tests/hdf5/1ATN_ppi.hdf5",
            "tests/hdf5/1ATN_ppi.hdf5",
            GINet,
            ["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            ["dist"],
            "regress",
            "irmsd",
            [OutputExporter(self.work_directory)],
            True,
            "mcl",
        )

    def test_ginet(self):
        _model_base_test(
            "tests/hdf5/1ATN_ppi.hdf5",
            "tests/hdf5/1ATN_ppi.hdf5",
            "tests/hdf5/1ATN_ppi.hdf5",
            GINet,
            ["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            ["dist"],
            "regress",
            "irmsd",
            [OutputExporter(self.work_directory)],
            False,
            "mcl",
        )

        assert len(os.listdir(self.work_directory)) > 0

    def test_ginet_class(self):
        _model_base_test(
            "tests/hdf5/variants.hdf5",
            "tests/hdf5/variants.hdf5",
            "tests/hdf5/variants.hdf5",
            GINet,
            ["polarity", "ic", "pssm"],
            ["dist"],
            "classif",
            "bin_class",
            [TensorboardBinaryClassificationExporter(self.work_directory)],
            False,
            "mcl",
        )

        assert len(os.listdir(self.work_directory)) > 0

    def test_fout(self):
        _model_base_test(
            "tests/hdf5/test.hdf5",
            "tests/hdf5/test.hdf5",
            "tests/hdf5/test.hdf5",
            FoutNet,
            ["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            ["dist"],
            "classif",
            "binary",
            [],
            False,
            "mcl",
        )

    def test_sgat(self):
        _model_base_test(
            "tests/hdf5/1ATN_ppi.hdf5",
            "tests/hdf5/1ATN_ppi.hdf5",
            "tests/hdf5/1ATN_ppi.hdf5",
            sGAT,
            ["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            ["dist"],
            "regress",
            "irmsd",
            [],
            False,
            "mcl",
        )

    def test_naive(self):
        _model_base_test(
            "tests/hdf5/test.hdf5",
            "tests/hdf5/test.hdf5",
            "tests/hdf5/test.hdf5",
            NaiveNetwork,
            ["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
            ["dist"],
            "regress",
            "BA",
            [OutputExporter(self.work_directory)],
            False,
            None,
        )

    def test_incompatible_regression(self):
        with pytest.raises(ValueError):
            _model_base_test(
                "tests/hdf5/1ATN_ppi.hdf5",
                "tests/hdf5/1ATN_ppi.hdf5",
                "tests/hdf5/1ATN_ppi.hdf5",
                sGAT,
                ["type", "polarity", "bsa", "depth", "hse", "ic", "pssm"],
                ["dist"],
                "regress",
                "irmsd",
                [TensorboardBinaryClassificationExporter(self.work_directory)],
                False,
                "mcl",
            )

    def test_incompatible_classification(self):
        with pytest.raises(ValueError):
            _model_base_test(
                "tests/hdf5/variants.hdf5",
                "tests/hdf5/variants.hdf5",
                "tests/hdf5/variants.hdf5",
                GINet,
                ["size", "polarity", "sasa", "ic", "pssm"],
                ["dist"],
                "classif",
                "bin_class",
                [ScatterPlotExporter(self.work_directory)],
                False,
                "mcl",
            )

    def test_no_val(self):
        _model_base_test(
            "tests/hdf5/test.hdf5",
            None,
            "tests/hdf5/test.hdf5",
            GINet,
            ["polarity", "ic", "pssm"],
            ["dist"],
            "classif",
            "binary",
            [TensorboardBinaryClassificationExporter(self.work_directory)],
            False,
            "mcl",
        )

    def test_incompatible_pretrained_no_test(self):
        with pytest.raises(ValueError):
            _model_base_test(
                "tests/hdf5/test.hdf5",
                None,
                None,
                GINet,
                ["polarity", "ic", "pssm"],
                ["dist"],
                "classif",
                "binary",
                [TensorboardBinaryClassificationExporter(self.work_directory)],
                False,
                "mcl",
            )

        assert len(os.listdir(self.work_directory)) > 0


if __name__ == "__main__":
    unittest.main()