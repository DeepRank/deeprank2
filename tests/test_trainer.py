import tempfile
import shutil
import os
import glob
import unittest
import pytest
import logging
import warnings
import torch
import h5py
from deeprankcore.trainer import Trainer, _divide_dataset
from deeprankcore.dataset import GraphDataset, GridDataset
from deeprankcore.neuralnets.gnn.ginet import GINet
from deeprankcore.neuralnets.gnn.foutnet import FoutNet
from deeprankcore.neuralnets.gnn.naive_gnn import NaiveNetwork
from deeprankcore.neuralnets.gnn.sgat import SGAT
from deeprankcore.neuralnets.cnn.model3d import CnnRegression, CnnClassification
from deeprankcore.utils.exporters import (
    HDF5OutputExporter,
    TensorboardBinaryClassificationExporter,
    ScatterPlotExporter
)
from deeprankcore.domain import (edgestorage as Efeat, nodestorage as Nfeat,
                                targetstorage as targets)


_log = logging.getLogger(__name__)

default_features = [Nfeat.RESTYPE, Nfeat.POLARITY, Nfeat.BSA, Nfeat.RESDEPTH, Nfeat.HSE, Nfeat.INFOCONTENT, Nfeat.PSSM]
model_path = './tests/test.pth.tar'

def _model_base_test( # pylint: disable=too-many-arguments, too-many-locals
    model_class,
    train_hdf5_path,
    val_hdf5_path,
    test_hdf5_path,
    node_features,
    edge_features,
    task,
    target,
    target_transform,
    output_exporters,
    clustering_method,
    use_cuda = False
):

    dataset_train = GraphDataset(
        hdf5_path=train_hdf5_path,
        node_features=node_features,
        edge_features=edge_features,
        target=target,
        task = task,
        clustering_method=clustering_method,
        target_transform = target_transform)

    if val_hdf5_path is not None:
        dataset_val = GraphDataset(
            hdf5_path=val_hdf5_path,
            node_features=node_features,
            edge_features=edge_features,
            target=target,
            task = task,
            clustering_method=clustering_method,
            target_transform = target_transform)
    else:
        dataset_val = None

    if test_hdf5_path is not None:
        dataset_test = GraphDataset(
            hdf5_path=test_hdf5_path,
            node_features=node_features,
            edge_features=edge_features,
            target=target,
            task=task,
            clustering_method=clustering_method,
            target_transform = target_transform)
    else:
        dataset_test = None

    trainer = Trainer(
        model_class,
        dataset_train,
        dataset_val,
        dataset_test,
        output_exporters=output_exporters,
    )

    if use_cuda:
        _log.debug("cuda is available, testing that the model is cuda")
        for parameter in trainer.model.parameters():
            assert parameter.is_cuda, f"{parameter} is not cuda"

        data = dataset_train.get(0)

        for name, data_tensor in (("x", data.x), ("y", data.y),
                                  (Efeat.INDEX, data.edge_index),
                                  ("edge_attr", data.edge_attr),
                                  (Nfeat.POSITION, data.pos),
                                  ("cluster0",data.cluster0),
                                  ("cluster1", data.cluster1)):

            if data_tensor is not None:
                assert data_tensor.is_cuda, f"data.{name} is not cuda"

    with warnings.catch_warnings(record=UserWarning):
        trainer.train(nepoch=3, batch_size=64, validate=True, save_best_model=None)
        trainer.save_model(model_path)

        Trainer(
            model_class,
            dataset_train,
            dataset_val,
            dataset_test,
            pretrained_model=model_path)

class TestTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(class_):
        class_.work_directory = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(class_):
        shutil.rmtree(class_.work_directory)

    def test_grid_regression(self):
        dataset = GridDataset(hdf5_path="tests/data/hdf5/1ATN_ppi.hdf5",
                              subset=None,
                              target=targets.IRMSD,
                              task=targets.REGRESS,
                              features=[Efeat.VANDERWAALS])

        trainer = Trainer(CnnRegression, dataset)

        trainer.train(nepoch=1, batch_size=2, save_best_model=None)

    def test_grid_classification(self):
        dataset = GridDataset(hdf5_path="tests/data/hdf5/1ATN_ppi.hdf5",
                              subset=None,
                              target=targets.BINARY,
                              task=targets.CLASSIF,
                              features=[Efeat.VANDERWAALS])

        trainer = Trainer(CnnClassification, dataset)

        trainer.train(nepoch=1, batch_size = 2, save_best_model=None)

    def test_grid_graph_incompatible(self):
        dataset_train = GridDataset(hdf5_path="tests/data/hdf5/1ATN_ppi.hdf5",
                                    subset=None,
                                    target=targets.BINARY,
                                    task=targets.CLASSIF,
                                    features=[Efeat.VANDERWAALS])

        dataset_valid = GraphDataset(
                hdf5_path="tests/data/hdf5/valid.hdf5",
                target=targets.BINARY,
            )

        with pytest.raises(TypeError):
            Trainer(CnnClassification,
                    dataset_train=dataset_train,
                    dataset_val=dataset_valid)

    def test_ginet_sigmoid(self):
        try:
            f = glob.glob(self.work_directory + '/*')[0]
            os.remove(f)
        except IndexError:
            pass
        assert len(os.listdir(self.work_directory)) == 0

        _model_base_test(
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

    def test_ginet(self):
        try:
            f = glob.glob(self.work_directory + '/*')[0]
            os.remove(f)
        except IndexError:
            pass
        assert len(os.listdir(self.work_directory)) == 0
        
        _model_base_test(           
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

    def test_ginet_class(self):
        try:
            f = glob.glob(self.work_directory + '/*')[0]
            os.remove(f)
        except IndexError:
            pass
        assert len(os.listdir(self.work_directory)) == 0

        _model_base_test(
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

    def test_fout(self):
        _model_base_test(
            FoutNet,
            "tests/data/hdf5/test.hdf5",
            "tests/data/hdf5/test.hdf5",
            "tests/data/hdf5/test.hdf5",
            default_features,
            [Efeat.DISTANCE],
            targets.CLASSIF,
            targets.BINARY,
            False,
            None,
            "mcl",
        )

    def test_sgat(self):
        _model_base_test(
            SGAT,
            "tests/data/hdf5/1ATN_ppi.hdf5",
            "tests/data/hdf5/1ATN_ppi.hdf5",
            "tests/data/hdf5/1ATN_ppi.hdf5",
            default_features,
            [Efeat.DISTANCE],
            targets.REGRESS,
            targets.IRMSD,
            False,
            None,
            "mcl",
        )

    def test_naive(self):
        _model_base_test(
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

    def test_incompatible_regression(self):
        with pytest.raises(ValueError):
            _model_base_test(
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

    def test_incompatible_classification(self):
        with pytest.raises(ValueError):
            _model_base_test(
                GINet,
                "tests/data/hdf5/variants.hdf5",
                "tests/data/hdf5/variants.hdf5",
                "tests/data/hdf5/variants.hdf5",
                [Nfeat.RESSIZE, Nfeat.POLARITY, Nfeat.SASA, Nfeat.INFOCONTENT, Nfeat.PSSM],
                [Efeat.DISTANCE],
                targets.CLASSIF,
                targets.BINARY,
                False,
                [ScatterPlotExporter(self.work_directory)],
                "mcl",
            )

    def test_incompatible_no_pretrained_no_train(self):
        with pytest.raises(ValueError):

            dataset = GraphDataset(
                hdf5_path="tests/data/hdf5/test.hdf5",
                target=targets.BINARY,
            )

            Trainer(
                neuralnet = NaiveNetwork,
                dataset_test = dataset,
            )

    def test_incompatible_no_pretrained_no_Net(self):
        with pytest.raises(ValueError):
            dataset = GraphDataset(
                hdf5_path="tests/data/hdf5/test.hdf5",
            )

            Trainer(
                neuralnet = NaiveNetwork,
                dataset_train = dataset,
            )

    def test_incompatible_no_pretrained_no_target(self):
        with pytest.raises(ValueError):
            dataset = GraphDataset(
                hdf5_path="tests/data/hdf5/test.hdf5",
                target=targets.BINARY,
            )

            Trainer(
                dataset_train = dataset,
            )

    def test_incompatible_pretrained_no_test(self):
        with pytest.raises(ValueError):
            dataset = GraphDataset(
                hdf5_path="tests/data/hdf5/test.hdf5",
                target=targets.BINARY,
                clustering_method="mcl"
            )

            trainer = Trainer(
                neuralnet = GINet,
                dataset_train = dataset,
            )

            with warnings.catch_warnings(record=UserWarning):
                trainer.train(nepoch=3, validate=True, save_best_model=None)
                trainer.save_model(model_path)

                Trainer(
                    neuralnet = GINet,
                    dataset_train = dataset,
                    pretrained_model=model_path)

    def test_incompatible_pretrained_no_Net(self):
        with pytest.raises(ValueError):
            dataset = GraphDataset(
                hdf5_path="tests/data/hdf5/test.hdf5",
                target=targets.BINARY,
                clustering_method="mcl"
            )

            trainer = Trainer(
                neuralnet = GINet,
                dataset_train = dataset,
            )

            with warnings.catch_warnings(record=UserWarning):
                trainer.train(nepoch=3, validate=True, save_best_model=None)
                trainer.save_model(model_path)

                Trainer(
                    dataset_test = dataset,
                    pretrained_model=model_path)

    def test_no_valid_provided(self):

        dataset = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            target=targets.BINARY,
            clustering_method="mcl"
        )

        trainer = Trainer(
            neuralnet = GINet,
            dataset_train = dataset,
        )
        trainer.train(batch_size = 1)

        assert len(trainer.train_loader) == int(0.75 * len(dataset))
        assert len(trainer.valid_loader) == int(0.25 * len(dataset))

    def test_no_valid_full_train(self):

        dataset = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            target=targets.BINARY,
            clustering_method = "mcl"
        )

        trainer = Trainer(
            neuralnet = GINet,
            dataset_train = dataset,
            val_size = 0
        )
        trainer.train(batch_size=1)

        assert len(trainer.train_loader) == len(dataset)
        assert trainer.valid_loader is None

    def test_optim(self):

        dataset = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            target=targets.BINARY,
        )

        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )

        optimizer = torch.optim.Adamax
        lr = 0.1
        weight_decay = 1e-04

        trainer.configure_optimizers(optimizer, lr, weight_decay)

        assert isinstance(trainer.optimizer, optimizer)
        assert trainer.lr == lr
        assert trainer.weight_decay == weight_decay

        with warnings.catch_warnings(record=UserWarning):
            trainer.train(nepoch=3, save_best_model=None)
            trainer.save_model(model_path)

            trainer_pretrained = Trainer(
                neuralnet = NaiveNetwork,
                dataset_test=dataset,
                pretrained_model=model_path)

        assert isinstance(trainer_pretrained.optimizer, optimizer)
        assert trainer_pretrained.lr == lr
        assert trainer_pretrained.weight_decay == weight_decay

    def test_default_optim(self):

        dataset = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5",
            target=targets.BINARY,
        )

        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )

        assert isinstance(trainer.optimizer, torch.optim.Adam)
        assert trainer.lr == 0.001
        assert trainer.weight_decay == 1e-05

    def test_cuda(self):    # test_ginet, but with cuda
        if torch.cuda.is_available():

            try:
                f = glob.glob(self.work_directory + '/*')[0]
                os.remove(f)
            except IndexError:
                pass
            assert len(os.listdir(self.work_directory)) == 0

            _model_base_test(           
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
                True
            )

            assert len(os.listdir(self.work_directory)) > 0

        else:
            warnings.warn("CUDA not available. test_cuda skipped")
            _log.debug("cuda is not available, test_cuda skipped")

    def test_dataset_equivalence_no_pretrained(self):
        with pytest.raises(ValueError):
            dataset_train = GraphDataset(
                hdf5_path="tests/data/hdf5/test.hdf5",
                target=targets.BINARY,
                edge_features=[Efeat.DISTANCE, Efeat.COVALENT]
            )
            
            dataset_val = GraphDataset(
                hdf5_path="tests/data/hdf5/test.hdf5",
                target=targets.BINARY,
                edge_features=[Efeat.DISTANCE]
            )

            Trainer(
                neuralnet = GINet,
                dataset_train = dataset_train,
                dataset_val = dataset_val,
            )

    def test_dataset_equivalence_pretrained(self):
        with pytest.raises(ValueError):
            dataset_train = GraphDataset(
                hdf5_path="tests/data/hdf5/test.hdf5",
                target=targets.BINARY,
                edge_features=[Efeat.DISTANCE, Efeat.COVALENT],
                clustering_method="mcl"
            )
            
            dataset_test = GraphDataset(
                hdf5_path="tests/data/hdf5/test.hdf5",
                target=targets.BINARY,
                edge_features=[Efeat.DISTANCE],
                clustering_method="mcl"
            )

            trainer = Trainer(
                neuralnet = GINet,
                dataset_train = dataset_train,
            )

            with warnings.catch_warnings(record=UserWarning):
                trainer.train(nepoch=3, validate=True, save_best_model=None)
                trainer.save_model(model_path)

                Trainer(
                    neuralnet = GINet,
                    dataset_train = dataset_train,
                    dataset_test = dataset_test,
                    pretrained_model=model_path)

    def test_trainsize(self):
        hdf5 = "tests/data/hdf5/train.hdf5"
        hdf5_file = h5py.File(hdf5, 'r')    # contains 44 datapoints
        n_val = int ( 0.25 * len(hdf5_file) )
        n_train = len(hdf5_file) - n_val
        test_cases = [None, 0.25, n_val]
        
        for t in test_cases:
            dataset_train, dataset_val =_divide_dataset(
                dataset = GraphDataset(hdf5_path=hdf5),
                splitsize=t,
            )

            assert len(dataset_train) == n_train
            assert len(dataset_val) == n_val

        hdf5_file.close()
        
    def test_invalid_trainsize(self):
        hdf5 = "tests/data/hdf5/train.hdf5"
        hdf5_file = h5py.File(hdf5, 'r')    # contains 44 datapoints
        n = len(hdf5_file)
        test_cases = [
            1.0, n,     # cannot be 100% validation data
            -0.5, -1,   # no negative values 
            1.1, n + 1, # cannot use more than all data as input
            ]
        
        for t in test_cases:
            print(t)
            with self.assertRaises(ValueError):
                _divide_dataset(
                    dataset = GraphDataset(hdf5_path=hdf5),
                    splitsize=t,
                )
        
        hdf5_file.close()

    def test_invalid_cuda_ngpus(self):
        dataset_train = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5"
        )
        
        dataset_val = GraphDataset(
            hdf5_path="tests/data/hdf5/test.hdf5"
        )

        if not torch.cuda.is_available():
            with pytest.raises(ValueError):
                Trainer(
                    neuralnet = GINet,
                    dataset_train = dataset_train,
                    dataset_val = dataset_val,
                    cuda = True
                )

        with pytest.raises(ValueError):
            Trainer(
                neuralnet = GINet,
                dataset_train = dataset_train,
                dataset_val = dataset_val,
                ngpu = 2
            )


if __name__ == "__main__":
    unittest.main()
