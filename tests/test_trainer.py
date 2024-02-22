import glob
import logging
import os
import shutil
import tempfile
import unittest
import warnings

import h5py
import pandas as pd
import pytest
import torch

from deeprank2.dataset import GraphDataset, GridDataset
from deeprank2.domain import edgestorage as Efeat
from deeprank2.domain import nodestorage as Nfeat
from deeprank2.domain import targetstorage as targets
from deeprank2.neuralnets.cnn.model3d import CnnClassification, CnnRegression
from deeprank2.neuralnets.gnn.foutnet import FoutNet
from deeprank2.neuralnets.gnn.ginet import GINet
from deeprank2.neuralnets.gnn.naive_gnn import NaiveNetwork
from deeprank2.neuralnets.gnn.sgat import SGAT
from deeprank2.trainer import Trainer, _divide_dataset
from deeprank2.utils.exporters import HDF5OutputExporter, ScatterPlotExporter, TensorboardBinaryClassificationExporter

# ruff: noqa: FBT003

_log = logging.getLogger(__name__)

default_features = [
    Nfeat.RESTYPE,
    Nfeat.POLARITY,
    Nfeat.BSA,
    Nfeat.RESDEPTH,
    Nfeat.HSE,
    Nfeat.INFOCONTENT,
    Nfeat.PSSM,
]


def _model_base_test(
    save_path: str,
    model_class: torch.nn.Module,
    train_hdf5_path: str,
    val_hdf5_path: str,
    test_hdf5_path: str,
    node_features: list[str],
    edge_features: list[str],
    task: str,
    target: str,
    target_transform: bool,
    output_exporters: list[HDF5OutputExporter],
    clustering_method: str,
    use_cuda: bool = False,
) -> None:
    dataset_train = GraphDataset(
        hdf5_path=train_hdf5_path,
        node_features=node_features,
        edge_features=edge_features,
        clustering_method=clustering_method,
        target=target,
        target_transform=target_transform,
        task=task,
    )

    if val_hdf5_path is not None:
        dataset_val = GraphDataset(
            hdf5_path=val_hdf5_path,
            train_source=dataset_train,
            clustering_method=clustering_method,
        )
    else:
        dataset_val = None

    if test_hdf5_path is not None:
        dataset_test = GraphDataset(
            hdf5_path=test_hdf5_path,
            train_source=dataset_train,
            clustering_method=clustering_method,
        )
    else:
        dataset_test = None

    trainer = Trainer(
        model_class,
        dataset_train,
        dataset_val,
        dataset_test,
        cuda=use_cuda,
        output_exporters=output_exporters,
    )

    if use_cuda:
        _log.debug("cuda is available, testing that the model is cuda")
        for parameter in trainer.model.parameters():
            assert parameter.is_cuda, f"{parameter} is not cuda"

    with warnings.catch_warnings(record=UserWarning):
        trainer.train(
            nepoch=3,
            batch_size=64,
            validate=True,
            best_model=False,
            filename=save_path,
        )

        Trainer(
            model_class,
            dataset_train,
            dataset_val,
            dataset_test,
            pretrained_model=save_path,
        )


class TestTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(class_) -> None:
        class_.work_directory = tempfile.mkdtemp()
        class_.save_path = class_.work_directory + "test.tar"

    @classmethod
    def tearDownClass(class_) -> None:
        shutil.rmtree(class_.work_directory)

    def test_grid_regression(self) -> None:
        dataset = GridDataset(
            hdf5_path="tests/data/hdf5/1ATN_ppi.hdf5",
            subset=None,
            features=[Efeat.VDW],
            target=targets.IRMSD,
            task=targets.REGRESS,
        )
        trainer = Trainer(CnnRegression, dataset)
        trainer.train(nepoch=1, batch_size=2, best_model=False, filename=None)

    def test_grid_classification(self) -> None:
        dataset = GridDataset(
            hdf5_path="tests/data/hdf5/1ATN_ppi.hdf5",
            subset=None,
            features=[Efeat.VDW],
            target=targets.BINARY,
            task=targets.CLASSIF,
        )
        trainer = Trainer(CnnClassification, dataset)
        trainer.train(
            nepoch=1,
            batch_size=2,
            best_model=False,
            filename=None,
        )

    def test_ginet_sigmoid(self) -> None:
        files = glob.glob(self.work_directory + "/*")
        for f in files:
            os.remove(f)
        assert len(os.listdir(self.work_directory)) == 0

        _model_base_test(
            self.save_path,
            GINet,
            "tests/data/hdf5/1ATN_ppi.hdf5",
            "tests/data/hdf5/1ATN_ppi.hdf5",
            "tests/data/hdf5/1ATN_ppi.hdf5",
            default_features,
            [Efeat.DISTANCE],
            targets.REGRESS,
            targets.IRMSD,
            True,
            [HDF5OutputExporter(self.work_directory)],
            "mcl",
        )
        assert len(os.listdir(self.work_directory)) > 0

    def test_ginet(self) -> None:
        files = glob.glob(self.work_directory + "/*")
        for f in files:
            os.remove(f)
        assert len(os.listdir(self.work_directory)) == 0

        _model_base_test(
            self.save_path,
            GINet,
            "tests/data/hdf5/1ATN_ppi.hdf5",
            "tests/data/hdf5/1ATN_ppi.hdf5",
            "tests/data/hdf5/1ATN_ppi.hdf5",
            default_features,
            [Efeat.DISTANCE],
            targets.REGRESS,
            targets.IRMSD,
            False,
            [HDF5OutputExporter(self.work_directory)],
            "mcl",
        )
        assert len(os.listdir(self.work_directory)) > 0

    def test_ginet_class(self) -> None:
        files = glob.glob(self.work_directory + "/*")
        for f in files:
            os.remove(f)
        assert len(os.listdir(self.work_directory)) == 0

        _model_base_test(
            self.save_path,
            GINet,
            "tests/data/hdf5/variants.hdf5",
            "tests/data/hdf5/variants.hdf5",
            "tests/data/hdf5/variants.hdf5",
            [Nfeat.POLARITY, Nfeat.INFOCONTENT, Nfeat.PSSM],
            [Efeat.DISTANCE],
            targets.CLASSIF,
            targets.BINARY,
            False,
            [TensorboardBinaryClassificationExporter(self.work_directory)],
            "mcl",
        )

        assert len(os.listdir(self.work_directory)) > 0

    def test_fout(self) -> None:
        files = glob.glob(self.work_directory + "/*")
        for f in files:
            os.remove(f)
        assert len(os.listdir(self.work_directory)) == 0

        _model_base_test(
            self.save_path,
            FoutNet,
            "tests/data/hdf5/test.hdf5",
            "tests/data/hdf5/test.hdf5",
            "tests/data/hdf5/test.hdf5",
            default_features,
            [Efeat.DISTANCE],
            targets.CLASSIF,
            targets.BINARY,
            False,
            [HDF5OutputExporter(self.work_directory)],
            "mcl",
        )
        assert len(os.listdir(self.work_directory)) > 0

    def test_sgat(self) -> None:
        files = glob.glob(self.work_directory + "/*")
        for f in files:
            os.remove(f)
        assert len(os.listdir(self.work_directory)) == 0

        _model_base_test(
            self.save_path,
            SGAT,
            "tests/data/hdf5/1ATN_ppi.hdf5",
            "tests/data/hdf5/1ATN_ppi.hdf5",
            "tests/data/hdf5/1ATN_ppi.hdf5",
            default_features,
            [Efeat.DISTANCE],
            targets.REGRESS,
            targets.IRMSD,
            False,
            [HDF5OutputExporter(self.work_directory)],
            "mcl",
        )
        assert len(os.listdir(self.work_directory)) > 0

    def test_naive(self) -> None:
        files = glob.glob(self.work_directory + "/*")
        for f in files:
            os.remove(f)
        assert len(os.listdir(self.work_directory)) == 0

        _model_base_test(
            self.save_path,
            NaiveNetwork,
            "tests/data/hdf5/test.hdf5",
            "tests/data/hdf5/test.hdf5",
            "tests/data/hdf5/test.hdf5",
            default_features,
            [Efeat.DISTANCE],
            targets.REGRESS,
            "BA",
            False,
            [HDF5OutputExporter(self.work_directory)],
            "mcl",
        )
        assert len(os.listdir(self.work_directory)) > 0

    def test_incompatible_regression(self) -> None:
        with pytest.raises(ValueError):
            _model_base_test(
                self.save_path,
                SGAT,
                "tests/data/hdf5/1ATN_ppi.hdf5",
                "tests/data/hdf5/1ATN_ppi.hdf5",
                "tests/data/hdf5/1ATN_ppi.hdf5",
                default_features,
                [Efeat.DISTANCE],
                targets.REGRESS,
                targets.IRMSD,
                False,
                [TensorboardBinaryClassificationExporter(self.work_directory)],
                "mcl",
            )

    def test_incompatible_classification(self) -> None:
        with pytest.raises(ValueError):
            _model_base_test(
                self.save_path,
                GINet,
                "tests/data/hdf5/variants.hdf5",
                "tests/data/hdf5/variants.hdf5",
                "tests/data/hdf5/variants.hdf5",
                [
                    Nfeat.RESSIZE,
                    Nfeat.POLARITY,
                    Nfeat.SASA,
                    Nfeat.INFOCONTENT,
                    Nfeat.PSSM,
                ],
                [Efeat.DISTANCE],
                targets.CLASSIF,
                targets.BINARY,
                False,
                [ScatterPlotExporter(self.work_directory)],
                "mcl",
            )

    def test_incompatible_no_pretrained_no_train(self) -> None:
        dataset = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            target=targets.BINARY,
        )

        with pytest.raises(ValueError):
            Trainer(
                neuralnet=NaiveNetwork,
                dataset_test=dataset,
            )

    def test_incompatible_no_pretrained_no_Net(self) -> None:
        with pytest.raises(ValueError):
            _ = GraphDataset(
                hdf5_path="tests/data/hdf5/test.hdf5",
            )

    def test_incompatible_no_pretrained_no_target(self) -> None:
        dataset = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            target=targets.BINARY,
        )
        with pytest.raises(ValueError):
            Trainer(
                dataset_train=dataset,
            )

    def test_incompatible_pretrained_no_test(self) -> None:
        dataset = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            clustering_method="mcl",
            target=targets.BINARY,
        )
        trainer = Trainer(
            neuralnet=GINet,
            dataset_train=dataset,
        )

        with warnings.catch_warnings(record=UserWarning):
            trainer.train(nepoch=3, validate=True, best_model=False, filename=self.save_path)
        with pytest.raises(ValueError):
            Trainer(
                neuralnet=GINet,
                dataset_train=dataset,
                pretrained_model=self.save_path,
            )

    def test_incompatible_pretrained_no_Net(self) -> None:
        dataset = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            clustering_method="mcl",
            target=targets.BINARY,
        )
        trainer = Trainer(
            neuralnet=GINet,
            dataset_train=dataset,
        )

        with warnings.catch_warnings(record=UserWarning):
            trainer.train(nepoch=3, validate=True, best_model=False, filename=self.save_path)
        with pytest.raises(ValueError):
            Trainer(dataset_test=dataset, pretrained_model=self.save_path)

    def test_no_training_no_pretrained(self) -> None:
        dataset_train = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            clustering_method="mcl",
            target=targets.BINARY,
        )
        dataset_val = GraphDataset(hdf5_path="tests/data/hdf5/test.hdf5", train_source=dataset_train)
        dataset_test = GraphDataset(hdf5_path="tests/data/hdf5/test.hdf5", train_source=dataset_train)
        trainer = Trainer(
            neuralnet=GINet,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
            dataset_test=dataset_test,
        )
        with pytest.raises(ValueError):
            trainer.test()

    def test_no_valid_provided(self) -> None:
        dataset = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            clustering_method="mcl",
            target=targets.BINARY,
        )
        trainer = Trainer(
            neuralnet=GINet,
            dataset_train=dataset,
        )
        trainer.train(batch_size=1, best_model=False, filename=None)
        assert len(trainer.train_loader) == int(0.75 * len(dataset))
        assert len(trainer.valid_loader) == int(0.25 * len(dataset))

    def test_no_test_provided(self) -> None:
        dataset_train = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            clustering_method="mcl",
            target=targets.BINARY,
        )
        dataset_val = GraphDataset(hdf5_path="tests/data/hdf5/test.hdf5", train_source=dataset_train)
        trainer = Trainer(
            neuralnet=GINet,
            dataset_train=dataset_train,
            dataset_val=dataset_val,
        )
        trainer.train(batch_size=1, best_model=False, filename=None)
        with pytest.raises(ValueError):
            trainer.test()

    def test_no_valid_full_train(self) -> None:
        dataset = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            clustering_method="mcl",
            target=targets.BINARY,
        )
        trainer = Trainer(
            neuralnet=GINet,
            dataset_train=dataset,
            val_size=0,
        )
        trainer.train(batch_size=1, best_model=False, filename=None)
        assert len(trainer.train_loader) == len(dataset)
        assert trainer.valid_loader is None

    def test_optim(self) -> None:
        dataset = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            target=targets.BINARY,
        )
        trainer = Trainer(
            neuralnet=NaiveNetwork,
            dataset_train=dataset,
        )

        optimizer = torch.optim.Adamax
        lr = 0.1
        weight_decay = 1e-04
        trainer.configure_optimizers(optimizer, lr, weight_decay)

        assert isinstance(trainer.optimizer, optimizer)
        assert trainer.lr == lr
        assert trainer.weight_decay == weight_decay

        with warnings.catch_warnings(record=UserWarning):
            trainer.train(nepoch=3, best_model=False, filename=self.save_path)
            trainer_pretrained = Trainer(
                neuralnet=NaiveNetwork,
                dataset_test=dataset,
                pretrained_model=self.save_path,
            )

        assert str(type(trainer_pretrained.optimizer)) == "<class 'torch.optim.adamax.Adamax'>"
        assert trainer_pretrained.lr == lr
        assert trainer_pretrained.weight_decay == weight_decay

    def test_default_optim(self) -> None:
        dataset = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            target=targets.BINARY,
        )
        trainer = Trainer(
            neuralnet=NaiveNetwork,
            dataset_train=dataset,
        )

        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert trainer.lr == 0.001
        assert trainer.weight_decay == 1e-05

    def test_cuda(self) -> None:  # test_ginet, but with cuda
        if torch.cuda.is_available():
            files = glob.glob(self.work_directory + "/*")
            for f in files:
                os.remove(f)
            assert len(os.listdir(self.work_directory)) == 0

            _model_base_test(
                self.save_path,
                GINet,
                "tests/data/hdf5/1ATN_ppi.hdf5",
                "tests/data/hdf5/1ATN_ppi.hdf5",
                "tests/data/hdf5/1ATN_ppi.hdf5",
                default_features,
                [Efeat.DISTANCE],
                targets.REGRESS,
                targets.IRMSD,
                False,
                [HDF5OutputExporter(self.work_directory)],
                "mcl",
                True,
            )
            assert len(os.listdir(self.work_directory)) > 0

        else:
            warnings.warn("CUDA is not available; test_cuda was skipped")
            _log.info("CUDA is not available; test_cuda was skipped")

    def test_dataset_equivalence_no_pretrained(self) -> None:
        # TestCase: dataset_train set (no pretrained model assigned).

        # Raise error when train dataset is neither a GraphDataset or GridDataset.
        dataset_invalid_train = GINet(input_shape=2)
        with pytest.raises(TypeError):
            Trainer(
                neuralnet=GINet,
                dataset_train=dataset_invalid_train,
            )

        dataset_train = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            edge_features=[Efeat.DISTANCE, Efeat.COVALENT],
            target=targets.BINARY,
        )

        # Raise error when train_source parameter in GraphDataset/GridDataset
        # is not equivalent to the dataset_train passed to Trainer.
        dataset_train_other = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            edge_features=[Efeat.SAMECHAIN, Efeat.COVALENT],
            target="BA",
            task="regress",
        )
        dataset_val = GraphDataset(hdf5_path="tests/data/hdf5/test.hdf5", train_source=dataset_train)
        dataset_test = GraphDataset(hdf5_path="tests/data/hdf5/test.hdf5", train_source=dataset_train)
        with pytest.raises(ValueError):
            Trainer(
                neuralnet=GINet,
                dataset_train=dataset_train_other,
                dataset_val=dataset_val,
            )
        with pytest.raises(ValueError):
            Trainer(
                neuralnet=GINet,
                dataset_train=dataset_train_other,
                dataset_test=dataset_test,
            )

    def test_dataset_equivalence_pretrained(self) -> None:
        # TestCase: No dataset_train set (pretrained model assigned).
        # Raise error when no dataset_test is set.

        dataset_train = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            edge_features=[Efeat.DISTANCE, Efeat.COVALENT],
            clustering_method="mcl",
            target=targets.BINARY,
        )

        trainer = Trainer(
            neuralnet=GINet,
            dataset_train=dataset_train,
        )

        with warnings.catch_warnings(record=UserWarning):
            # train pretrained model
            trainer.train(nepoch=3, validate=True, best_model=False, filename=self.save_path)
            # pretrained model assigned(no dataset_train needed)
        with pytest.raises(ValueError):
            Trainer(neuralnet=GINet, pretrained_model=self.save_path)

    def test_trainsize(self) -> None:
        hdf5 = "tests/data/hdf5/train.hdf5"
        hdf5_file = h5py.File(hdf5, "r")  # contains 44 datapoints
        n_val = int(0.25 * len(hdf5_file))
        n_train = len(hdf5_file) - n_val
        test_cases = [None, 0.25, n_val]

        for t in test_cases:
            dataset_train, dataset_val = _divide_dataset(
                dataset=GraphDataset(hdf5_path=hdf5, target=targets.BINARY),
                splitsize=t,
            )
            assert len(dataset_train) == n_train
            assert len(dataset_val) == n_val

        hdf5_file.close()

    def test_invalid_trainsize(self) -> None:
        hdf5 = "tests/data/hdf5/train.hdf5"
        hdf5_file = h5py.File(hdf5, "r")  # contains 44 datapoints
        n = len(hdf5_file)
        test_cases = [
            1.0,
            n,  # fmt: skip; cannot be 100% validation data
            -0.5,
            -1,  # fmt: skip; no negative values
            1.1,
            n + 1,  # fmt: skip; cannot use more than all data as input
        ]

        for t in test_cases:
            print(t)  # noqa: T201, print in case it fails we can see on which one it failed
            with pytest.raises(ValueError):
                _divide_dataset(
                    dataset=GraphDataset(hdf5_path=hdf5),
                    splitsize=t,
                )

        hdf5_file.close()

    def test_invalid_cuda_ngpus(self) -> None:
        dataset_train = GraphDataset(hdf5_path="tests/data/hdf5/test.hdf5", target=targets.BINARY)
        dataset_val = GraphDataset(hdf5_path="tests/data/hdf5/test.hdf5", train_source=dataset_train)

        with pytest.raises(ValueError):
            Trainer(
                neuralnet=GINet,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                ngpu=2,
            )

    def test_invalid_no_cuda_available(self) -> None:
        if not torch.cuda.is_available():
            dataset_train = GraphDataset(hdf5_path="tests/data/hdf5/test.hdf5", target=targets.BINARY)
            dataset_val = GraphDataset(hdf5_path="tests/data/hdf5/test.hdf5", train_source=dataset_train)

            with pytest.raises(ValueError):
                Trainer(
                    neuralnet=GINet,
                    dataset_train=dataset_train,
                    dataset_val=dataset_val,
                    cuda=True,
                )

        else:
            warnings.warn("CUDA is available; test_invalid_no_cuda_available was skipped")
            _log.info("CUDA is available; test_invalid_no_cuda_available was skipped")

    def test_train_method_no_train(self) -> None:
        # Graphs data
        test_data_graph = "tests/data/hdf5/test.hdf5"
        pretrained_model_graph = "tests/data/pretrained/testing_graph_model.pth.tar"

        dataset_test = GraphDataset(hdf5_path=test_data_graph, train_source=pretrained_model_graph)
        trainer = Trainer(
            neuralnet=NaiveNetwork,
            dataset_test=dataset_test,
            pretrained_model=pretrained_model_graph,
        )

        with pytest.raises(ValueError):
            trainer.train()

        # Grids data
        test_data_grid = "tests/data/hdf5/1ATN_ppi.hdf5"
        pretrained_model_grid = "tests/data/pretrained/testing_grid_model.pth.tar"

        dataset_test = GridDataset(hdf5_path=test_data_grid, train_source=pretrained_model_grid)
        trainer = Trainer(
            neuralnet=CnnClassification,
            dataset_test=dataset_test,
            pretrained_model=pretrained_model_grid,
        )

        with pytest.raises(ValueError):
            trainer.train()

    def test_test_method_pretrained_model_on_dataset_with_target(self) -> None:
        # Graphs data
        test_data_graph = "tests/data/hdf5/test.hdf5"
        pretrained_model_graph = "tests/data/pretrained/testing_graph_model.pth.tar"

        dataset_test = GraphDataset(hdf5_path=test_data_graph, train_source=pretrained_model_graph)

        trainer = Trainer(
            neuralnet=NaiveNetwork,
            dataset_test=dataset_test,
            pretrained_model=pretrained_model_graph,
            output_exporters=[HDF5OutputExporter("./")],
        )

        trainer.test()

        output = pd.read_hdf("output_exporter.hdf5", key="testing")
        assert len(output) == len(dataset_test)

        # Grids data
        test_data_grid = "tests/data/hdf5/1ATN_ppi.hdf5"
        pretrained_model_grid = "tests/data/pretrained/testing_grid_model.pth.tar"

        dataset_test = GridDataset(hdf5_path=test_data_grid, train_source=pretrained_model_grid)

        trainer = Trainer(
            neuralnet=CnnClassification,
            dataset_test=dataset_test,
            pretrained_model=pretrained_model_grid,
            output_exporters=[HDF5OutputExporter("./")],
        )

        trainer.test()

        output = pd.read_hdf("output_exporter.hdf5", key="testing")
        assert len(output) == len(dataset_test)

    def test_test_method_pretrained_model_on_dataset_without_target(self) -> None:
        # Graphs data
        test_data_graph = "tests/data/hdf5/test_no_target.hdf5"
        pretrained_model_graph = "tests/data/pretrained/testing_graph_model.pth.tar"

        dataset_test = GraphDataset(hdf5_path=test_data_graph, train_source=pretrained_model_graph)

        trainer = Trainer(
            neuralnet=NaiveNetwork,
            dataset_test=dataset_test,
            pretrained_model=pretrained_model_graph,
            output_exporters=[HDF5OutputExporter("./")],
        )

        trainer.test()

        output = pd.read_hdf("output_exporter.hdf5", key="testing")
        assert len(output) == len(dataset_test)
        assert output.target.unique().tolist()[0] is None
        assert output.loss.unique().tolist()[0] is None

        # Grids data
        test_data_grid = "tests/data/hdf5/test_no_target.hdf5"
        pretrained_model_grid = "tests/data/pretrained/testing_grid_model.pth.tar"

        dataset_test = GridDataset(hdf5_path=test_data_grid, train_source=pretrained_model_grid)

        trainer = Trainer(
            neuralnet=CnnClassification,
            dataset_test=dataset_test,
            pretrained_model=pretrained_model_grid,
            output_exporters=[HDF5OutputExporter("./")],
        )

        trainer.test()

        output = pd.read_hdf("output_exporter.hdf5", key="testing")
        assert len(output) == len(dataset_test)
        assert output.target.unique().tolist()[0] is None
        assert output.loss.unique().tolist()[0] is None

    def test_graph_save_and_load_model(self) -> None:
        test_data_graph = "tests/data/hdf5/test.hdf5"
        n = 10
        features_transform = {
            Nfeat.RESTYPE: {"transform": lambda x: x / 2, "standardize": True},
            Nfeat.BSA: {"transform": None, "standardize": False},
        }

        dataset = GraphDataset(
            hdf5_path=test_data_graph,
            node_features=[Nfeat.RESTYPE, Nfeat.POLARITY, Nfeat.BSA],
            target=targets.BINARY,
            task=targets.CLASSIF,
            features_transform=features_transform,
        )
        trainer = Trainer(NaiveNetwork, dataset)
        # during the training the model is saved
        trainer.train(nepoch=2, batch_size=2, filename=self.save_path)
        assert trainer.features_transform == features_transform

        # load the model into a new GraphDataset instance
        dataset_test = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            train_source=self.save_path,
        )

        # Check if the features_transform is correctly loaded from the saved model
        assert dataset_test.features_transform[Nfeat.RESTYPE]["transform"](n) == n / 2  # the only way to test the transform in this case is to apply it
        assert dataset_test.features_transform[Nfeat.RESTYPE]["standardize"] == features_transform[Nfeat.RESTYPE]["standardize"]
        assert dataset_test.features_transform[Nfeat.BSA] == features_transform[Nfeat.BSA]


if __name__ == "__main__":
    unittest.main()
