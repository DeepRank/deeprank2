import tempfile
import shutil
import unittest
import pytest
import logging
import warnings
from torch import nn
from deeprankcore.trainer import Trainer
from deeprankcore.dataset import GraphDataset
from deeprankcore.neuralnets.gnn.naive_gnn import NaiveNetwork
from deeprankcore.domain import targetstorage as targets, losstypes as losses

_log = logging.getLogger(__name__)


model_path = './tests/test.pth.tar'
hdf5_path = 'tests/data/hdf5/test.hdf5'

def base_test(trainer: Trainer, lossfunction = None, override = False):

    if lossfunction:
        trainer.set_lossfunction(lossfunction = lossfunction, override_invalid=override)

    # check correct passing to/picking up from pretrained model 
    with warnings.catch_warnings(record=UserWarning):
        trainer.train(nepoch=2, save_best_model=None)
        trainer.save_model(model_path)

        trainer_pretrained = Trainer(
            neuralnet = NaiveNetwork,
            dataset_test=trainer.dataset_train,
            pretrained_model=model_path)
        
        return trainer_pretrained


class TestLosses(unittest.TestCase):

    # Classification tasks
    def test_classif_default(self):
        dataset = GraphDataset(hdf5_path, 
            target=targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )

        trainer_pretrained = base_test(trainer)
        assert isinstance(trainer.lossfunction, nn.CrossEntropyLoss)
        assert isinstance(trainer_pretrained.lossfunction, nn.CrossEntropyLoss)


    def test_classif_all(self):
        dataset = GraphDataset(hdf5_path,
            target=targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )

        # only NLLLoss and CrossEntropyLoss are currently working
        # for lossfunction in losses.classification_losses:
        for lossfunction in [nn.CrossEntropyLoss, nn.NLLLoss]:
            trainer_pretrained = base_test(trainer, lossfunction)
            assert isinstance(trainer.lossfunction, lossfunction)
            assert isinstance(trainer_pretrained.lossfunction, lossfunction)
        

    def test_classif_weighted(self):
        dataset = GraphDataset(hdf5_path, 
            target=targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
            class_weights = True
        )
        lossfunction = nn.NLLLoss

        trainer_pretrained = base_test(trainer, lossfunction)
        assert isinstance(trainer.lossfunction, lossfunction)
        assert isinstance(trainer_pretrained.lossfunction, lossfunction)
        assert trainer_pretrained.class_weights


    # def test_classif_invalid_weighted(self):
    #     dataset = GraphDataset(hdf5_path, 
    #         target=targets.BINARY)
    #     trainer = Trainer(
    #         neuralnet = NaiveNetwork,
    #         dataset_train = dataset,
    #         class_weights = True
    #     )
    #     # use a loss function that does not allow for weighted loss, e.g. MultiLabelMarginLoss
    #     lossfunction = nn.MultiLabelMarginLoss

    #     with pytest.raises(ValueError):
    #         base_test(trainer, lossfunction)


    def test_classif_invalid_lossfunction(self):
        dataset = GraphDataset(hdf5_path, 
            target=targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        lossfunction = nn.MSELoss

        with pytest.raises(ValueError):
            base_test(trainer, lossfunction)


    def test_classif_invalid_lossfunction_override(self):
        dataset = GraphDataset(hdf5_path, 
            target=targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        lossfunction = nn.MSELoss

        with pytest.raises(RuntimeError):
            base_test(trainer, lossfunction, override = True)


    # Regression tasks
    def test_regress_default(self):
        dataset = GraphDataset(hdf5_path,
            target=targets.BA)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )

        trainer_pretrained = base_test(trainer)
        assert isinstance(trainer.lossfunction, nn.MSELoss)
        assert isinstance(trainer_pretrained.lossfunction, nn.MSELoss)


    def test_regress_all(self):
        dataset = GraphDataset(hdf5_path, 
            target=targets.BA)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        for f in losses.regression_losses:
            lossfunction = f

            trainer_pretrained = base_test(trainer, lossfunction)
            assert isinstance(trainer.lossfunction, lossfunction)
            assert isinstance(trainer_pretrained.lossfunction, lossfunction)


    def test_regress_invalid_lossfunction(self):
        dataset = GraphDataset(hdf5_path, 
            target=targets.BA)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        lossfunction = nn.CrossEntropyLoss

        with pytest.raises(ValueError):
            base_test(trainer, lossfunction)


    def test_regress_invalid_lossfunction_override(self):
        dataset = GraphDataset(hdf5_path, 
            target=targets.BA)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        lossfunction = nn.CrossEntropyLoss

        base_test(trainer, lossfunction, override=True)
