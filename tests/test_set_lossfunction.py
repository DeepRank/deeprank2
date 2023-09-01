import shutil
import tempfile
import unittest
import warnings

import pytest
from deeprank2.dataset import GraphDataset
from deeprank2.neuralnets.gnn.naive_gnn import NaiveNetwork
from deeprank2.trainer import Trainer
from torch import nn

from deeprank2.domain import losstypes as losses
from deeprank2.domain import targetstorage as targets

hdf5_path = 'tests/data/hdf5/test.hdf5'

def base_test(model_path, trainer: Trainer, lossfunction = None, override = False):

    if lossfunction:
        trainer.set_lossfunction(lossfunction = lossfunction, override_invalid=override)

    # check correct passing to/picking up from pretrained model
    with warnings.catch_warnings(record=UserWarning):
        trainer.train(nepoch=2, best_model=False, filename=model_path)

        trainer_pretrained = Trainer(
            neuralnet = NaiveNetwork,
            dataset_test=trainer.dataset_train,
            pretrained_model=model_path)

        return trainer_pretrained


class TestLosses(unittest.TestCase):
    @classmethod
    def setUpClass(class_):
        class_.work_directory = tempfile.mkdtemp()
        class_.save_path = class_.work_directory + 'test.tar'

    @classmethod
    def tearDownClass(class_):
        shutil.rmtree(class_.work_directory)

    # Classification tasks
    def test_classif_default(self):
        dataset = GraphDataset(hdf5_path,
            target = targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )

        trainer_pretrained = base_test(self.save_path, trainer)
        assert isinstance(trainer.lossfunction, nn.CrossEntropyLoss)
        assert isinstance(trainer_pretrained.lossfunction, nn.CrossEntropyLoss)


    def test_classif_all(self):
        dataset = GraphDataset(hdf5_path,
            target = targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )

        # only NLLLoss and CrossEntropyLoss are currently working
        # for lossfunction in losses.classification_losses:
        for lossfunction in [nn.CrossEntropyLoss, nn.NLLLoss]:
            trainer_pretrained = base_test(self.save_path, trainer, lossfunction)
            assert isinstance(trainer.lossfunction, lossfunction)
            assert isinstance(trainer_pretrained.lossfunction, lossfunction)


    def test_classif_weighted(self):
        dataset = GraphDataset(hdf5_path,
            target = targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
            class_weights = True
        )
        lossfunction = nn.NLLLoss

        trainer_pretrained = base_test(self.save_path, trainer, lossfunction)
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
    #         base_test(self.save_path, trainer, lossfunction)


    def test_classif_invalid_lossfunction(self):
        dataset = GraphDataset(hdf5_path,
            target = targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        lossfunction = nn.MSELoss

        with pytest.raises(ValueError):
            base_test(self.save_path, trainer, lossfunction)


    def test_classif_invalid_lossfunction_override(self):
        dataset = GraphDataset(hdf5_path,
            target = targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        lossfunction = nn.MSELoss

        with pytest.raises(RuntimeError):
            base_test(self.save_path, trainer, lossfunction, override = True)


    # Regression tasks
    def test_regress_default(self):
        dataset = GraphDataset(hdf5_path,
            target = 'BA',
            task = 'regress')
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )

        trainer_pretrained = base_test(self.save_path, trainer)
        assert isinstance(trainer.lossfunction, nn.MSELoss)
        assert isinstance(trainer_pretrained.lossfunction, nn.MSELoss)


    def test_regress_all(self):
        dataset = GraphDataset(hdf5_path,
            target = 'BA', task = 'regress')
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        for f in losses.regression_losses:
            lossfunction = f

            trainer_pretrained = base_test(self.save_path, trainer, lossfunction)
            assert isinstance(trainer.lossfunction, lossfunction)
            assert isinstance(trainer_pretrained.lossfunction, lossfunction)


    def test_regress_invalid_lossfunction(self):
        dataset = GraphDataset(hdf5_path,
            target = 'BA', task = 'regress')
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        lossfunction = nn.CrossEntropyLoss

        with pytest.raises(ValueError):
            base_test(self.save_path, trainer, lossfunction)


    def test_regress_invalid_lossfunction_override(self):
        dataset = GraphDataset(hdf5_path,
            target = 'BA', task = 'regress')
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        lossfunction = nn.CrossEntropyLoss

        base_test(self.save_path, trainer, lossfunction, override=True)
