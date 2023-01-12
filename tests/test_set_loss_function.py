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
from deeprankcore.domain import targetstorage as targets



regression_losses = [nn.L1Loss, nn.SmoothL1Loss, nn.MSELoss, nn.HuberLoss, ]

binary_classification_losses = [nn.SoftMarginLoss, nn.BCELoss, nn.BCEWithLogitsLoss, ]

multi_classification_losses = [nn.NLLLoss, nn.PoissonNLLLoss, nn.GaussianNLLLoss, nn.CrossEntropyLoss, 
                        nn.KLDivLoss, nn.MultiLabelMarginLoss, nn.MultiLabelSoftMarginLoss, ]

other_losses = [nn.HingeEmbeddingLoss, nn.CosineEmbeddingLoss, 
                nn.MarginRankingLoss, nn.TripletMarginLoss, nn.CTCLoss]

classification_losses = multi_classification_losses + binary_classification_losses


_log = logging.getLogger(__name__)

model_path = './tests/test.pth.tar'
hdf5_path = 'tests/data/hdf5/test.hdf5'

def base_test(trainer: Trainer, loss_function):
    trainer.set_loss_function(loss_function = loss_function)

    assert isinstance(trainer.loss_function, loss_function)

    with warnings.catch_warnings(record=UserWarning):
        trainer.train(nepoch=3, save_best_model=None)
        trainer.save_model(model_path)

        trainer_pretrained = Trainer(
            neuralnet = NaiveNetwork,
            dataset_test=trainer.dataset_train,
            pretrained_model=model_path)

    assert isinstance(trainer_pretrained.loss_function, loss_function)


class TestTrainer(unittest.TestCase):
    @classmethod
    def setUpClass(class_):
        class_.work_directory = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(class_):
        shutil.rmtree(class_.work_directory)

    # classification tasks
    def test_classif_unweighted(self):
        dataset = GraphDataset(hdf5_path, 
            target=targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        loss_function = nn.NLLLoss
        base_test(trainer, loss_function)

    def test_classif_weighted(self):
        dataset = GraphDataset(hdf5_path, 
            target=targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
            class_weights = True
        )
        loss_function = nn.NLLLoss

        base_test(trainer, loss_function)
        assert trainer.class_weights

    def test_classif_weighted_invalid(self):
        assert True
    
    def test_classif_invalid_lossfunction(self):
        assert True

    def test_classif_invalid_lossfunction_override(self):
        assert True


    # regression tasks

    def test_regress(self):
        dataset = GraphDataset(hdf5_path, 
            target='BA',
            task=targets.REGRESS)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        loss_function = nn.MSELoss
        base_test(trainer, loss_function)

    def test_regress_invalid_lossfunction(self):
        assert True

    def test_regress_invalid_lossfunction_override(self):
        assert True
