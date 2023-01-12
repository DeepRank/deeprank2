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

def base_test(trainer: Trainer, loss_function = None):

    trainer.set_loss_function(loss_function = loss_function)

    # check correct passing to/picking up from pretrained model 
    with warnings.catch_warnings(record=UserWarning):
        trainer.train(nepoch=2, save_best_model=None)
        trainer.save_model(model_path)

        trainer_pretrained = Trainer(
            neuralnet = NaiveNetwork,
            dataset_test=trainer.dataset_train,
            pretrained_model=model_path)
        
        return trainer_pretrained


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

        # only NLLLoss and CrossEntropyLoss are working!
        classification_losses = [nn.NLLLoss, nn.CrossEntropyLoss]
        for f in classification_losses:
            loss_function = f

            trainer_pretrained = base_test(trainer, loss_function)
            assert isinstance(trainer.loss_function, loss_function)
            assert isinstance(trainer_pretrained.loss_function, loss_function)
        
        raise AssertionError('Only 2 classification losses are currently working')

    def test_classif_weighted(self):
        dataset = GraphDataset(hdf5_path, 
            target=targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
            class_weights = True
        )
        loss_function = nn.NLLLoss

        trainer_pretrained = base_test(trainer, loss_function)
        assert isinstance(trainer.loss_function, loss_function)
        assert isinstance(trainer_pretrained.loss_function, loss_function)
        assert trainer_pretrained.class_weights

    def test_classif_invalid_weighted(self):
        assert False
    
    def test_classif_invalid_lossfunction(self):
        dataset = GraphDataset(hdf5_path, 
            target=targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        loss_function = nn.MSELoss

        with pytest.raises(ValueError):
            base_test(trainer, loss_function)

    def test_classif_invalid_lossfunction_override(self):
        assert False

    def test_classif_default(self):
        dataset = GraphDataset(hdf5_path, 
            target=targets.BINARY)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )

        trainer_pretrained = base_test(trainer)
        assert isinstance(trainer.loss_function, nn.CrossEntropyLoss)
        assert isinstance(trainer_pretrained.loss_function, nn.CrossEntropyLoss)


    # regression tasks

    def test_regress(self):
        dataset = GraphDataset(hdf5_path, 
            target=targets.BA)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        for f in regression_losses:
            loss_function = f

            trainer_pretrained = base_test(trainer, loss_function)
            assert isinstance(trainer.loss_function, loss_function)
            assert isinstance(trainer_pretrained.loss_function, loss_function)

    def test_regress_invalid_lossfunction(self):
        dataset = GraphDataset(hdf5_path, 
            target=targets.BA)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )
        loss_function = nn.CrossEntropyLoss

        with pytest.raises(ValueError):
            base_test(trainer, loss_function)

    def test_regress_invalid_lossfunction_override(self):
        assert False

    def test_regress_default(self):
        dataset = GraphDataset(hdf5_path,
            target=targets.BA)
        trainer = Trainer(
            neuralnet = NaiveNetwork,
            dataset_train = dataset,
        )

        trainer_pretrained = base_test(trainer)
        assert isinstance(trainer.loss_function, nn.MSELoss)
        assert isinstance(trainer_pretrained.loss_function, nn.MSELoss)
