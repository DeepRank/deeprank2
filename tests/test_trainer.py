import tempfile
import shutil
import os
import unittest
import pytest
import logging
import warnings
import torch
from deeprankcore.Trainer import Trainer
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
from deeprankcore.domain.features import groups, edgefeats
from deeprankcore.domain.features import nodefeats as Nfeat
from deeprankcore.domain import targettypes as targets


_log = logging.getLogger(__name__)

default_features = [Nfeat.RESTYPE, Nfeat.POLARITY, Nfeat.BSA, Nfeat.RESDEPTH, Nfeat.HSE, Nfeat.INFOCONTENT, Nfeat.PSSM]


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
    clustering_method,
    use_cuda = False
):

    dataset_train = HDF5DataSet(
        hdf5_path=train_hdf5_path,
        root="./",
        node_feature=node_features,
        edge_feature=edge_features,
        task = task,
        target=target,
        clustering_method=clustering_method)

    if val_hdf5_path is not None:
        dataset_val = HDF5DataSet(
            hdf5_path=val_hdf5_path,
            root="./",
            node_feature=node_features,
            edge_feature=edge_features,
            task = task,
            target=target,
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

    trainer = Trainer(
        dataset_train,
        dataset_val,
        dataset_test,
        model_class,
        batch_size=64,
        metrics_exporters=metrics_exporters,
        transform_sigmoid=transform_sigmoid,
    )

    if use_cuda:
        _log.debug("cuda is available, testing that the model is cuda")
        for parameter in trainer.model.parameters():
            assert parameter.is_cuda, f"{parameter} is not cuda"

        data = dataset_train.get(0)

        for name, data_tensor in (("x", data.x), ("y", data.y),
                                  (groups.INDEX, data.edge_index),
                                  ("edge_attr", data.edge_attr),
                                  (groups.POSITION, data.pos),
                                  ("cluster0",data.cluster0),
                                  ("cluster1", data.cluster1)):

            if data_tensor is not None:
                assert data_tensor.is_cuda, f"data.{name} is not cuda"

    trainer.train(nepoch=10, validate=True)

    trainer.save_model("test.pth.tar")

    Trainer(
        dataset_train,
        dataset_val,
        dataset_test,
        model_class,
        pretrained_model="test.pth.tar")

class TestTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(class_):
        class_.work_directory = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(class_):
        shutil.rmtree(class_.work_directory)

    def test_ginet_sigmoid(self):
        _model_base_test(
            "tests/data/hdf5/1ATN_ppi.hdf5",
            "tests/data/hdf5/1ATN_ppi.hdf5",
            "tests/data/hdf5/1ATN_ppi.hdf5",
            GINet,
            default_features,
            [edgefeats.DISTANCE],
            targets.REGRESS,
            targets.IRMSD,
            [OutputExporter(self.work_directory)],
            True,
            "mcl",
        )

    def test_ginet(self):
        _model_base_test(           
            "tests/data/hdf5/1ATN_ppi.hdf5",
            "tests/data/hdf5/1ATN_ppi.hdf5",
            "tests/data/hdf5/1ATN_ppi.hdf5",
            GINet,
            default_features,
            [edgefeats.DISTANCE],
            targets.REGRESS,
            targets.IRMSD,
            [OutputExporter(self.work_directory)],
            False,
            "mcl",
        )

        assert len(os.listdir(self.work_directory)) > 0

    def test_ginet_class(self):
        _model_base_test(
            "tests/data/hdf5/variants.hdf5",
            "tests/data/hdf5/variants.hdf5",
            "tests/data/hdf5/variants.hdf5",
            GINet,
            [Nfeat.POLARITY, Nfeat.INFOCONTENT, Nfeat.PSSM],
            [edgefeats.DISTANCE],
            targets.CLASSIF,
            targets.BINARY,
            [TensorboardBinaryClassificationExporter(self.work_directory)],
            False,
            "mcl",
        )

        assert len(os.listdir(self.work_directory)) > 0

    def test_fout(self):
        _model_base_test(
            "tests/data/hdf5/test.hdf5",
            "tests/data/hdf5/test.hdf5",
            "tests/data/hdf5/test.hdf5",
            FoutNet,
            default_features,
            [edgefeats.DISTANCE],
            targets.CLASSIF,
            targets.BINARY,
            [],
            False,
            "mcl",
        )

    def test_sgat(self):
        _model_base_test(
            "tests/data/hdf5/1ATN_ppi.hdf5",
            "tests/data/hdf5/1ATN_ppi.hdf5",
            "tests/data/hdf5/1ATN_ppi.hdf5",
            sGAT,
            default_features,
            [edgefeats.DISTANCE],
            targets.REGRESS,
            targets.IRMSD,
            [],
            False,
            "mcl",
        )

    def test_naive(self):
        _model_base_test(
            "tests/data/hdf5/test.hdf5",
            "tests/data/hdf5/test.hdf5",
            "tests/data/hdf5/test.hdf5",
            NaiveNetwork,
            default_features,
            [edgefeats.DISTANCE],
            targets.REGRESS,
            "BA",
            [OutputExporter(self.work_directory)],
            False,
            None,
        )

    def test_incompatible_regression(self):
        with pytest.raises(ValueError):
            _model_base_test(
                "tests/data/hdf5/1ATN_ppi.hdf5",
                "tests/data/hdf5/1ATN_ppi.hdf5",
                "tests/data/hdf5/1ATN_ppi.hdf5",
                sGAT,
                default_features,
                [edgefeats.DISTANCE],
                targets.REGRESS,
                targets.IRMSD,
                [TensorboardBinaryClassificationExporter(self.work_directory)],
                False,
                "mcl",
            )

    def test_incompatible_classification(self):
        with pytest.raises(ValueError):
            _model_base_test(
                "tests/data/hdf5/variants.hdf5",
                "tests/data/hdf5/variants.hdf5",
                "tests/data/hdf5/variants.hdf5",
                GINet,
                [Nfeat.RESSIZE, Nfeat.POLARITY, Nfeat.SASA, Nfeat.INFOCONTENT, Nfeat.PSSM],
                [edgefeats.DISTANCE],
                targets.CLASSIF,
                targets.BINARY,
                [ScatterPlotExporter(self.work_directory)],
                False,
                "mcl",
            )

    def test_incompatible_no_pretrained_no_train(self):
        with pytest.raises(ValueError):

            dataset = HDF5DataSet(
                hdf5_path="tests/data/hdf5/test.hdf5",
                target=targets.BINARY,
                root="./")

            Trainer(
                dataset_test = dataset,
                Net = NaiveNetwork,
            )

    def test_incompatible_no_pretrained_no_Net(self):
        with pytest.raises(ValueError):
            dataset = HDF5DataSet(
                hdf5_path="tests/data/hdf5/test.hdf5",
                target=targets.BINARY,
                root="./")

            Trainer(
                dataset_train = dataset,
            )

    def test_incompatible_pretrained_no_test(self):
        with pytest.raises(ValueError):
            dataset = HDF5DataSet(
                hdf5_path="tests/data/hdf5/test.hdf5",
                target=targets.BINARY,
                root="./")

            trainer = Trainer(
                dataset_train = dataset,
                Net = GINet,
            )

            trainer.train(nepoch=10, validate=True)

            trainer.save_model("test.pth.tar")

            Trainer(
                dataset_train = dataset,
                Net = GINet,
                pretrained_model="test.pth.tar")

    def test_incompatible_pretrained_no_Net(self):
        with pytest.raises(ValueError):
            dataset = HDF5DataSet(
                hdf5_path="tests/data/hdf5/test.hdf5",
                target=targets.BINARY,
                root="./")

            trainer = Trainer(
                dataset_train = dataset,
                Net = GINet,
            )

            trainer.train(nepoch=10, validate=True)

            trainer.save_model("test.pth.tar")

            Trainer(
                dataset_test = dataset,
                pretrained_model="test.pth.tar")

    def test_no_valid_provided(self):

        dataset = HDF5DataSet(
            hdf5_path="tests/data/hdf5/test.hdf5",
            target=targets.BINARY,
            root="./")

        trainer = Trainer(
            dataset_train = dataset,
            Net = GINet,
            batch_size = 1
        )

        assert len(trainer.train_loader) == int(0.75 * len(dataset))
        assert len(trainer.valid_loader) == int(0.25 * len(dataset))

    def test_no_valid_full_train(self):

        dataset = HDF5DataSet(
            hdf5_path="tests/data/hdf5/test.hdf5",
            target=targets.BINARY,
            root="./")

        trainer = Trainer(
            dataset_train = dataset,
            Net = GINet,
            val_size = 0,
            batch_size = 1
        )

        assert len(trainer.train_loader) == len(dataset)
        assert trainer.valid_loader is None

    def test_optim(self):

        dataset = HDF5DataSet(
            hdf5_path="tests/data/hdf5/test.hdf5",
            target=targets.BINARY,
            root="./")

        trainer = Trainer(
            dataset_train = dataset,
            Net = NaiveNetwork,
        )

        optimizer = torch.optim.Adamax
        lr = 0.1
        weight_decay = 1e-04

        trainer.configure_optimizers(optimizer, lr, weight_decay)

        assert isinstance(trainer.optimizer, optimizer)
        assert trainer.lr == lr
        assert trainer.weight_decay == weight_decay

        trainer.train(nepoch=10, validate=True)

        trainer.save_model("test.pth.tar")

        trainer_pretrained = Trainer(
            dataset_test=dataset,
            Net = NaiveNetwork,
            pretrained_model="test.pth.tar")

        assert isinstance(trainer_pretrained.optimizer, optimizer)
        assert trainer_pretrained.lr == lr
        assert trainer_pretrained.weight_decay == weight_decay

    def test_default_optim(self):

        dataset = HDF5DataSet(
            hdf5_path="tests/data/hdf5/test.hdf5",
            target=targets.BINARY,
            root="./")

        trainer = Trainer(
            dataset_train = dataset,
            Net = NaiveNetwork,
        )

        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert trainer.lr == 0.001
        assert trainer.weight_decay == 1e-05

    def test_cuda(self):    # test_ginet, but with cuda
        if torch.cuda.is_available():

            _model_base_test(           
                "tests/data/hdf5/1ATN_ppi.hdf5",
                "tests/data/hdf5/1ATN_ppi.hdf5",
                "tests/data/hdf5/1ATN_ppi.hdf5",
                GINet,
                default_features,
                [edgefeats.DISTANCE],
                targets.REGRESS,
                targets.IRMSD,
                [OutputExporter(self.work_directory)],
                False,
                "mcl",
                True # use_cuda
            )

            assert len(os.listdir(self.work_directory)) > 0

        else:
            warnings.warn("CUDA NOT AVAILABLE. test_cuda skipped")
            _log.debug("cuda is not available, test_cuda skipped")


if __name__ == "__main__":
    unittest.main()