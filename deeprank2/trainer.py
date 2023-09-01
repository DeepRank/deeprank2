import copy
import logging
from time import time
from typing import List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from deeprank2.dataset import GraphDataset, GridDataset
from deeprank2.domain import losstypes as losses
from deeprank2.domain import targetstorage as targets
from deeprank2.utils.community_pooling import (community_detection,
                                               community_pooling)
from deeprank2.utils.earlystopping import EarlyStopping
from deeprank2.utils.exporters import (HDF5OutputExporter, OutputExporter,
                                       OutputExporterCollection)

_log = logging.getLogger(__name__)


class Trainer():
    def __init__( # pylint: disable=too-many-arguments # noqa: MC0001
                self,
                neuralnet = None,
                dataset_train: Optional[Union[GraphDataset, GridDataset]] = None,
                dataset_val: Optional[Union[GraphDataset, GridDataset]] = None,
                dataset_test: Optional[Union[GraphDataset, GridDataset]] = None,
                val_size: Optional[Union[float, int]] = None,
                test_size: Optional[Union[float, int]] = None,
                class_weights: bool = False,
                pretrained_model: Optional[str] = None,
                cuda: bool = False,
                ngpu: int = 0,
                output_exporters: Optional[List[OutputExporter]] = None,
            ):
        """Class from which the network is trained, evaluated and tested.

        Args:
            neuralnet (child class of :class:`torch.nn.Module`, optional): Neural network class (ex. :class:`GINet`, :class:`Foutnet` etc.).
                It should subclass :class:`torch.nn.Module`, and it shouldn't be specific to regression or classification
                in terms of output shape (:class:`Trainer` class takes care of formatting the output shape according to the task).
                More specifically, in classification task cases, softmax shouldn't be used as the last activation function.
                Defaults to None.
            dataset_train (Optional[Union[:class:`GraphDataset`, :class:`GridDataset`]], optional): Training set used during training.
                Can't be None if pretrained_model is also None. Defaults to None.
            dataset_val (Optional[Union[:class:`GraphDataset`, :class:`GridDataset`]], optional): Evaluation set used during training.
                If None, training set will be split randomly into training set and validation set during training, using val_size parameter.
                Defaults to None.
            dataset_test (Optional[Union[:class:`GraphDataset`, :class:`GridDataset`]], optional): Independent evaluation set. Defaults to None.
            val_size (Optional[Union[float,int]], optional): Fraction of dataset (if float) or number of datapoints (if int) to use for validation.
                Only used if dataset_val is not specified. Can be set to 0 if no validation set is needed. Defaults to None (in _divide_dataset function).
            test_size (Optional[Union[float,int]], optional): Fraction of dataset (if float) or number of datapoints (if int) to use for test dataset.
                Only used if dataset_test is not specified. Can be set to 0 if no test set is needed. Defaults to None.
            class_weights (bool, optional): Assign class weights based on the dataset content. Defaults to False.
            pretrained_model (Optional[str], optional): Path to pre-trained model. Defaults to None.
            cuda (bool, optional): Whether to use CUDA. Defaults to False.
            ngpu (int, optional): Number of GPU to be used. Defaults to 0.
            output_exporters (Optional[List[OutputExporter]], optional): The output exporters to use for saving/exploring/plotting predictions/targets/losses
                over the epochs. If None, defaults to :class:`HDF5OutputExporter`, which saves all the results in an .HDF5 file stored in ./output directory.
                Defaults to None.
        """
        self.batch_size_train = None
        self.batch_size_test = None
        self.shuffle = None

        self._init_output_exporters(output_exporters)

        self.neuralnet = neuralnet

        self._init_datasets(dataset_train, dataset_val, dataset_test,
                            val_size, test_size)

        self.cuda = cuda
        self.ngpu = ngpu

        if self.cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
            if self.ngpu == 0:
                self.ngpu = 1
                _log.info("CUDA detected. Setting number of GPUs to 1.")
        elif self.cuda and not torch.cuda.is_available():
            _log.error(
                """
                --> CUDA not detected: Make sure that CUDA is installed
                    and that you are running on GPUs.\n
                --> To turn CUDA off set cuda=False in Trainer.\n
                --> Aborting the experiment \n\n'
                """)
            raise ValueError(
                """
                --> CUDA not detected: Make sure that CUDA is installed
                    and that you are running on GPUs.\n
                --> To turn CUDA off set cuda=False in Trainer.\n
                --> Aborting the experiment \n\n'
                """)
        else:
            self.device = torch.device("cpu")
            if self.ngpu > 0:
                _log.error(
                    """
                    --> CUDA not detected.
                        Set cuda=True in Trainer to turn CUDA on.\n
                    --> Aborting the experiment \n\n
                    """)
                raise ValueError(
                    """
                    --> CUDA not detected.
                        Set cuda=True in Trainer to turn CUDA on.\n
                    --> Aborting the experiment \n\n
                    """)

        _log.info(f"Device set to {self.device}.")
        if self.device.type == 'cuda':
            _log.info(f"CUDA device name is {torch.cuda.get_device_name(0)}.")
            _log.info(f"Number of GPUs set to {self.ngpu}.")

        if pretrained_model is None:
            if self.dataset_train is None:
                raise ValueError("No training data specified. Training data is required if there is no pretrained model.")
            if self.neuralnet is None:
                raise ValueError("No neural network specified. Specifying a model framework is required if there is no pretrained model.")

            self.classes = self.dataset_train.classes
            self.classes_to_index = self.dataset_train.classes_to_index
            self.optimizer = None
            self.class_weights = class_weights
            self.subset = self.dataset_train.subset
            self.epoch_saved_model = None

            if self.target is None:
                raise ValueError("No target set. You need to choose a target (set in the dataset) for training.")

            self._load_model()

            # clustering the datasets
            if self.clustering_method is not None:
                if self.clustering_method in ('mcl', 'louvain'):
                    _log.info("Loading clusters")
                    self._precluster(self.dataset_train)

                    if self.dataset_val is not None:
                        self._precluster(self.dataset_val)
                    else:
                        _log.warning("No validation dataset given. Randomly splitting training set in training set and validation set.")
                        self.dataset_train, self.dataset_val = _divide_dataset(
                            self.dataset_train, splitsize=self.val_size)

                    if self.dataset_test is not None:
                        self._precluster(self.dataset_test)
                else:
                    raise ValueError(
                        f"Invalid node clustering method: {self.clustering_method}\n\t"
                        "Please set clustering_method to 'mcl', 'louvain' or None. Default to 'mcl' \n\t")

        else:
            if self.dataset_train is not None:
                _log.warning("Pretrained model loaded: dataset_train will be ignored.")
            if self.dataset_val is not None:
                _log.warning("Pretrained model loaded: dataset_val will be ignored.")
            if self.neuralnet is None:
                raise ValueError("No neural network class found. Please add it to complete loading the pretrained model.")
            if self.dataset_test is None:
                raise ValueError("No dataset_test found. Please add it to evaluate the pretrained model.")
            if self.target is None:
                raise ValueError("No target set. Make sure the pretrained model explicitly defines the target to train against.")

            self.pretrained_model_path = pretrained_model
            self.classes_to_index = self.dataset_test.classes_to_index

            self._load_params()
            self._load_pretrained_model()

    def _init_output_exporters(self, output_exporters: Optional[List[OutputExporter]]):

        if output_exporters is not None:
            self._output_exporters = OutputExporterCollection(*output_exporters)
        else:
            self._output_exporters = OutputExporterCollection(HDF5OutputExporter('./output'))

    def _init_datasets(self,  # pylint: disable=too-many-arguments
                       dataset_train: Union[GraphDataset, GridDataset],
                       dataset_val: Optional[Union[GraphDataset, GridDataset]],
                       dataset_test: Optional[Union[GraphDataset, GridDataset]],
                       val_size: Optional[Union[int, float]],
                       test_size: Optional[Union[int, float]]):

        self._check_dataset_equivalence(dataset_train, dataset_val, dataset_test)

        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_val = dataset_val
        self.val_size = val_size
        self.test_size = test_size

        # Divide datasets where necessary.
        if test_size is not None:
            if dataset_test is None:
                self.dataset_train, self.dataset_test = _divide_dataset(dataset_train, test_size)
            else:
                _log.warning("Test dataset was provided to Trainer; test_size parameter is ignored.")

        if val_size is not None:
            if dataset_val is None:
                self.dataset_train, self.dataset_val = _divide_dataset(dataset_train, val_size)
            else:
                _log.warning("Validation dataset was provided to Trainer; val_size parameter is ignored.")

        # Copy settings from the dataset that we will use.
        if self.dataset_train is not None:
            self._init_from_dataset(self.dataset_train)
        else:
            self._init_from_dataset(self.dataset_test)

    def _init_from_dataset(self, dataset: Union[GraphDataset, GridDataset]):

        if isinstance(dataset, GraphDataset):
            self.clustering_method = dataset.clustering_method
            self.node_features = dataset.node_features
            self.edge_features = dataset.edge_features
            self.features = None

        elif isinstance(dataset, GridDataset):
            self.clustering_method = None
            self.node_features = None
            self.edge_features = None
            self.features = dataset.features
        else:
            raise TypeError(type(dataset))

        self.target = dataset.target
        self.task = dataset.task

    def _load_model(self):
        """Loads the neural network model."""

        self._put_model_to_device(self.dataset_train)
        self.configure_optimizers()
        self.set_lossfunction()

    def _check_dataset_equivalence(self, dataset_train, dataset_val, dataset_test):
        """Check train_dataset type, train parameter and dataset_train parameter settings."""

        # dataset_train is None when pretrained_model is set
        if dataset_train is None:
            # only check the test dataset
            if dataset_test is None:
                raise ValueError("Please provide at least a train or test dataset")
        else:
            # Make sure train dataset has valid type
            if not isinstance(dataset_train, GraphDataset) and not isinstance(dataset_train, GridDataset):
                raise TypeError(f"""train dataset is not the right type {type(dataset_train)}
                                Make sure it's either GraphDataset or GridDataset""")

            if dataset_val is not None:
                self._check_dataset_value(dataset_train, dataset_val, type_dataset = "valid")

            if dataset_test is not None:
                self._check_dataset_value(dataset_train, dataset_test, type_dataset = "test")

    def _check_dataset_value(self, dataset_train, dataset_check, type_dataset):
        """Check valid/test dataset settings."""

        # Check train parameter in valid/test is set as False.
        if dataset_check.train is not False:
            raise ValueError(f"""{type_dataset} dataset has train parameter {dataset_check.train}
                        Make sure to set it as False""")
        # Check dataset_train parameter in valid/test is equivalent to train which passed to Trainer.
        if dataset_check.dataset_train != dataset_train:
            raise ValueError(f"""{type_dataset} dataset has different dataset_train parameter compared to the one given in Trainer.
                        Make sure to assign equivalent dataset_train in Trainer""")

    def _load_pretrained_model(self):
        """
        Loads pretrained model
        """

        self.test_loader = DataLoader(
            self.dataset_test,
            pin_memory=self.cuda)
        _log.info("Testing set loaded\n")
        self._put_model_to_device(self.dataset_test)

        # load the model and the optimizer state
        self.optimizer.load_state_dict(self.opt_loaded_state_dict)
        self.model.load_state_dict(self.model_load_state_dict)

    def _precluster(self, dataset: GraphDataset):
        """Pre-clusters nodes of the graphs

        Args:
            dataset (:class:`GraphDataset`)
        """
        for fname, mol in tqdm(dataset.index_entries):
            data = dataset.load_one_graph(fname, mol)

            if data is None:
                f5 = h5py.File(fname, "a")
                try:
                    _log.info(f"deleting {mol}")
                    del f5[mol]
                except BaseException:
                    _log.info(f"{mol} not found")
                f5.close()
                continue

            f5 = h5py.File(fname, "a")
            grp = f5[mol]
            clust_grp = grp.require_group("clustering")

            if self.clustering_method.lower() in clust_grp:
                del clust_grp[self.clustering_method.lower()]

            method_grp = clust_grp.create_group(self.clustering_method.lower())
            cluster = community_detection(
                data.edge_index, data.num_nodes, method=self.clustering_method
            )
            method_grp.create_dataset("depth_0", data=cluster.cpu())
            data = community_pooling(cluster, data)
            cluster = community_detection(
                data.edge_index, data.num_nodes, method=self.clustering_method
            )
            method_grp.create_dataset("depth_1", data=cluster.cpu())

            f5.close()

    def _put_model_to_device(self, dataset: Union[GraphDataset, GridDataset]):
        """
        Puts the model on the available device

        Args:
            dataset (Union[:class:`GraphDataset`, :class:`GridDataset`]): GraphDataset object.

        Raises:
            ValueError: Incorrect output shape
        """

        # regression mode
        if self.task == targets.REGRESS:
            self.output_shape = 1

        # classification mode
        elif self.task == targets.CLASSIF:
            self.output_shape = len(self.classes)

        # the target values are optional
        if dataset.get(0).y is not None:
            target_shape = dataset.get(0).y.shape[0]
        else:
            target_shape = None

        if isinstance(dataset, GraphDataset):

            num_node_features = dataset.get(0).num_features
            num_edge_features = len(dataset.edge_features)

            self.model = self.neuralnet(
                num_node_features,
                self.output_shape,
                num_edge_features
            ).to(self.device)

        elif isinstance(dataset, GridDataset):
            _, num_features, box_width, box_height, box_depth = dataset.get(0).x.shape

            self.model = self.neuralnet(num_features,
                                        (box_width, box_height, box_depth)
            ).to(self.device)
        else:
            raise TypeError(type(dataset))

        # multi-gpu
        if self.ngpu > 1:
            ids = list(range(self.ngpu))
            self.model = nn.DataParallel(self.model, device_ids=ids).to(self.device)

        # check for compatibility
        for output_exporter in self._output_exporters:
            if not output_exporter.is_compatible_with(self.output_shape, target_shape):
                raise ValueError(f"""output exporter of type {type(output_exporter)}\n
                                 is not compatible with output shape {self.output_shape}\n
                                 and target shape {target_shape}""")

    def configure_optimizers(self, optimizer = None, lr: float = 0.001, weight_decay: float = 1e-05):

        """
        Configure optimizer and its main parameters.

        Args:
            optimizer (:class:`torch.optim`, optional): PyTorch optimizer object. If none, defaults to :class:`torch.optim.Adam`.
                Defaults to None.

            lr (float, optional): Learning rate. Defaults to 0.001.

            weight_decay (float, optional): Weight decay (L2 penalty).
                Weight decay is fundamental for GNNs, otherwise, parameters can become too big and the gradient may explode. Defaults to 1e-05.
        """

        self.lr = lr
        self.weight_decay = weight_decay

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            try:
                self.optimizer = optimizer(self.model.parameters(), lr = lr, weight_decay = weight_decay)
            except Exception as e:
                _log.error(e)
                _log.info("Invalid optimizer. Please use only optimizers classes from torch.optim package.")
                raise e

    def set_lossfunction(self, lossfunction = None, override_invalid: bool = False): #pylint: disable=too-many-locals # noqa: MC0001

        """
        Set the loss function.

        Args:
            lossfunction (optional): Make sure to use a loss function that is appropriate for
                your task (classification or regression). All loss functions
                from torch.nn.modules.loss are listed as belonging to either
                category (or to neither) and an exception is raised if an invalid
                loss function is chosen for the set task.
                Default for regression: MSELoss.
                Default for classification: CrossEntropyLoss.
                Defaults to None.
            override_invalid (bool, optional): If True, loss functions that are considered
                invalid for the task do no longer automaticallt raise an exception.
                Defaults to False.
        """

        default_regression_loss = nn.MSELoss
        default_classification_loss = nn.CrossEntropyLoss

        def _invalid_loss():
            if override_invalid:
                _log.warning(f'The provided loss function ({lossfunction}) is not appropriate for {self.task} tasks.\n\t' +
                            'You have set override_invalid to True, so the training will run with this loss function nonetheless.\n\t' +
                            'This will likely cause other errors or exceptions down the line.')
            else:
                invalid_loss_error = (f'The provided loss function ({lossfunction}) is not appropriate for {self.task} tasks.\n\t' +
                                    'If you want to use this loss function anyway, set override_invalid to True.')
                _log.error(invalid_loss_error)
                raise ValueError(invalid_loss_error)

        # check for custom/invalid loss functions
        if lossfunction in losses.other_losses:
            _invalid_loss()
        elif lossfunction not in (losses.regression_losses + losses.classification_losses):
            custom_loss = True
        else:
            custom_loss = False

        # set regression loss
        if self.task == targets.REGRESS:
            if lossfunction is None:
                lossfunction = default_regression_loss
                _log.info(f'No loss function provided, the default loss function for {self.task} tasks is used: {lossfunction}')
            else:
                if custom_loss:
                    custom_loss_warning = ( f'The provided loss function ({lossfunction}) is not part of the default list.\n\t' +
                                            f'Please ensure that this loss function is appropriate for {self.task} tasks.\n\t')
                    _log.warning(custom_loss_warning)
                elif lossfunction not in losses.regression_losses:
                    _invalid_loss()
            self.lossfunction = lossfunction()

        # Set classification loss
        elif self.task == targets.CLASSIF:
            if lossfunction is None:
                lossfunction = default_classification_loss
                _log.info(f'No loss function provided, the default loss function for {self.task} tasks is used: {lossfunction}')
            else:
                if custom_loss:
                    custom_loss_warning = ( f'The provided loss function ({lossfunction}) is not part of the default list.\n\t' +
                                            f'Please ensure that this loss function is appropriate for {self.task} tasks.\n\t')
                    _log.warning(custom_loss_warning)
                elif lossfunction not in losses.classification_losses:
                    _invalid_loss()
            if not self.class_weights:
                self.lossfunction = lossfunction()
            else:
                self.lossfunction = lossfunction # weights will be set in the train() method

    def train( # pylint: disable=too-many-arguments, too-many-branches, too-many-locals # noqa: MC0001
        self,
        nepoch: int = 1,
        batch_size: int = 32,
        shuffle: bool = True,
        earlystop_patience: Optional[int] = None,
        earlystop_maxgap: Optional[float] = None,
        min_epoch: int = 10,
        validate: bool = False,
        num_workers: int = 0,
        best_model: bool = True,
        filename: Optional[str] = 'model.pth.tar'
    ):
        """
        Performs the training of the model.

        Args:
            nepoch (int, optional): Maximum number of epochs to run.
                        Defaults to 1.
            batch_size (int, optional): Sets the size of the batch.
                        Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the training dataloaders data (train set and validation set).
                        Default: True.
            earlystop_patience (Optional[int], optional): Training ends if the model has run for this number of epochs without improving the validation loss.
                        Defaults to None.
            earlystop_maxgap (Optional[float], optional): Training ends if the difference between validation and training loss exceeds this value.
                        Defaults to None.
            min_epoch (float, optional): Minimum epoch to be reached before looking at maxgap.
                        Defaults to 10.
            validate (bool, optional): Perform validation on independent data set (requires a validation data set).
                        Defaults to False.
            num_workers (int, optional): How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
                        Defaults to 0.
            best_model (bool, optional):
                        If True, the best model (in terms of validation loss) is selected for later testing or saving.
                        If False, the last model tried is selected.
                        Defaults to True.
            filename (str, optional): Name of the file where to save the selected model. If not None, the model is saved to `filename`.
                If None, the model is not saved. Defaults to 'model.pth.tar'.
        """
        self.batch_size_train = batch_size
        self.shuffle = shuffle

        self.train_loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size_train,
            shuffle=self.shuffle,
            num_workers=num_workers,
            pin_memory=self.cuda
        )
        _log.info("Training set loaded\n")

        if self.dataset_val is not None:
            self.valid_loader = DataLoader(
                self.dataset_val,
                batch_size=self.batch_size_train,
                shuffle=self.shuffle,
                num_workers=num_workers,
                pin_memory=self.cuda
            )
            _log.info("Validation set loaded\n")
        else:
            self.valid_loader = None
            _log.info("No validation set provided\n")

        # Assign weights to each class
        if self.task == targets.CLASSIF and self.class_weights:
            targets_all = []
            for batch in self.train_loader:
                targets_all.append(batch.y)

            targets_all = torch.cat(targets_all).squeeze().tolist()
            self.weights = torch.tensor(
                [targets_all.count(i) for i in self.classes], dtype=torch.float32
            )
            _log.info(f"class occurences: {self.weights}")
            self.weights = 1.0 / self.weights
            self.weights = self.weights / self.weights.sum()
            _log.info(f"class weights: {self.weights}")

            try:
                self.lossfunction = self.lossfunction(weight=self.weights.to(self.device))  # Check whether loss allows for weighted classes
            except TypeError as e:
                weight_error = (f"Loss function {self.lossfunction} does not allow for weighted classes.\n\t" +
                                "Please use a different loss function or set class_weights to False.\n")
                _log.error(weight_error)
                raise ValueError(weight_error) from e
        else:
            self.weights = None

        train_losses = []
        valid_losses = []

        if earlystop_patience or earlystop_maxgap:
            early_stopping = EarlyStopping(patience=earlystop_patience, maxgap=earlystop_maxgap, min_epoch=min_epoch, trace_func=_log.info)
        else:
            early_stopping = None

        with self._output_exporters:
            # Number of epochs
            self.nepoch = nepoch
            _log.info('Epoch 0:')
            self._eval(self.train_loader, 0, "training")
            if validate:
                if self.valid_loader is None:
                    raise ValueError("No validation dataset provided.")
                self._eval(self.valid_loader, 0, "validation")

            # Loop over epochs
            for epoch in range(1, nepoch + 1):
                _log.info(f'Epoch {epoch}:')

                # Set the module in training mode
                self.model.train()
                loss_ = self._epoch(epoch, "training")
                train_losses.append(loss_)

                # Validate the model
                if validate:
                    loss_ = self._eval(self.valid_loader, epoch, "validation")
                    valid_losses.append(loss_)
                    if best_model:
                        if min(valid_losses) == loss_:
                            checkpoint_model = self._save_model()
                            self.epoch_saved_model = epoch
                            _log.info(f'Best model saved at epoch # {self.epoch_saved_model}.')
                    # check early stopping criteria (in validation case only)
                    if early_stopping:
                        # compare last validation and training loss
                        early_stopping(epoch, valid_losses[-1], train_losses[-1])
                        if early_stopping.early_stop:
                            break

                else:
                    # if no validation set, save the best performing model on the training set
                    if best_model:
                        if min(train_losses) == loss_:
                            _log.warning(
                                "Training data is used both for learning and model selection, which will to overfitting." +
                                "\n\tIt is preferable to use an independent training and validation data sets.")
                            checkpoint_model = self._save_model()
                            self.epoch_saved_model = epoch
                            _log.info(f'Best model saved at epoch # {self.epoch_saved_model}.')

            # Save the last model
            if best_model is False:
                checkpoint_model = self._save_model()
                self.epoch_saved_model = epoch
                _log.info(f'Last model saved at epoch # {self.epoch_saved_model}.')

        # Now that the training loop is over, save the model
        if filename:
            torch.save(checkpoint_model, filename)
        self.opt_loaded_state_dict = checkpoint_model["optimizer_state"]
        self.model_load_state_dict = checkpoint_model["model_state"]
        self.optimizer.load_state_dict(self.opt_loaded_state_dict)
        self.model.load_state_dict(self.model_load_state_dict)

    def _epoch(self, epoch_number: int, pass_name: str) -> float:
        """
        Runs a single epoch

        Args:
            epoch_number (int)
            pass_name (str): 'training', 'validation' or 'testing'

        Returns:
            Running loss.
        """

        sum_of_losses = 0
        count_predictions = 0
        target_vals = []
        outputs = []
        entry_names = []
        t0 = time()
        for data_batch in self.train_loader:
            if self.cuda:
                data_batch = data_batch.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            pred = self.model(data_batch)
            pred, data_batch.y = self._format_output(pred, data_batch.y)
            loss_ = self.lossfunction(pred, data_batch.y)
            loss_.backward()
            self.optimizer.step()
            count_predictions += pred.shape[0]

            # convert mean back to sum
            sum_of_losses += loss_.detach().item() * pred.detach().shape[0]
            target_vals += data_batch.y.detach().cpu().numpy().tolist()

            # Get the outputs for export
            # Remember that non-linear activation is automatically applied in CrossEntropyLoss
            if self.task == targets.CLASSIF:
                pred = F.softmax(pred.detach(), dim=1)
            else:
                pred = pred.detach().reshape(-1)
            outputs += pred.cpu().numpy().tolist()

            # Get the name
            entry_names += data_batch.entry_names

        dt = time() - t0
        if count_predictions > 0:
            epoch_loss = sum_of_losses / count_predictions
        else:
            epoch_loss = 0.0

        self._output_exporters.process(
            pass_name, epoch_number, entry_names, outputs, target_vals, epoch_loss)
        self._log_epoch_data(pass_name, epoch_loss, dt)

        return epoch_loss

    def _eval( # pylint: disable=too-many-locals
            self,
            loader: DataLoader,
            epoch_number: int,
            pass_name: str
        ) -> float:

        """
        Evaluates the model

        Args:
            loader (Dataloader): Data to evaluate on.
            epoch_number (int): Number for this epoch, used for storing the outputs.
            pass_name (str): 'training', 'validation' or 'testing'

        Returns:
            Running loss.
        """

        # Sets the module in evaluation mode
        self.model.eval()
        loss_func = self.lossfunction
        target_vals = []
        outputs = []
        entry_names = []
        sum_of_losses = 0
        count_predictions = 0
        t0 = time()
        for data_batch in loader:
            if self.cuda:
                data_batch = data_batch.to(self.device, non_blocking=True)
            pred = self.model(data_batch)
            pred, y = self._format_output(pred, data_batch.y)

            # Check if a target value was provided (i.e. benchmarck scenario)
            if y is not None:
                target_vals += y.cpu().numpy().tolist()
                loss_ = loss_func(pred, y)
                count_predictions += pred.shape[0]
                sum_of_losses += loss_.detach().item() * pred.shape[0]

            # Get the outputs for export
            # Remember that non-linear activation is automatically applied in CrossEntropyLoss
            if self.task == targets.CLASSIF:
                pred = F.softmax(pred.detach(), dim=1)
            else:
                pred = pred.detach().reshape(-1)
            outputs += pred.cpu().numpy().tolist()

            # get the name
            entry_names += data_batch.entry_names

        dt = time() - t0
        if count_predictions > 0:
            eval_loss = sum_of_losses / count_predictions
        else:
            eval_loss = 0.0

        self._output_exporters.process(
            pass_name, epoch_number, entry_names, outputs, target_vals, eval_loss)
        self._log_epoch_data(pass_name, eval_loss, dt)

        return eval_loss

    @staticmethod
    def _log_epoch_data(stage: str, loss: float, time: float):
        """
        Prints the data of each epoch

        Args:
            stage (str): Train or valid.
            loss (float): Loss during that epoch.
            time (float): Timing of the epoch.
        """
        _log.info(f'{stage} loss {loss} | time {time}')

    def _format_output(self, pred, target=None):

        """
        Format the network output depending on the task (classification/regression).
        """

        if (self.task == targets.CLASSIF) and (target is not None):
            # For categorical cross entropy, the target must be a one-dimensional tensor
            # of class indices with type long and the output should have raw, unnormalized values
            target = torch.tensor(
                [self.classes_to_index[x] if isinstance(x, str) else self.classes_to_index[int(x)] for x in target]
            )
            if isinstance(self.lossfunction, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                # # pred must be in (0,1) range and target must be float with same shape as pred
                # pred = F.softmax(pred)
                # target = torch.tensor(
                #     [[0,1] if x == [1] else [1,0] for x in target]
                # ).float()
                raise ValueError('BCELoss and BCEWithLogitsLoss are currently not supported.\n\t' +
                                'For further details see: https://github.com/DeepRank/deeprank2/issues/318')

            if isinstance(self.lossfunction, losses.classification_losses) and not isinstance(self.lossfunction, losses.classification_tested):
                raise ValueError(f'{self.lossfunction} is currently not supported.\n\t' +
                                f'Supported loss functions for classification: {losses.classification_tested}.\n\t' +
                                'Implementation of other loss functions requires adaptation of Trainer._format_output.')

        elif self.task == targets.REGRESS:
            pred = pred.reshape(-1)

        if target is not None:
            target = target.to(self.device)

        return pred, target

    def test(
        self,
        batch_size: int = 32,
        num_workers: int = 0):
        """
        Performs the testing of the model.

        Args:
            batch_size (int, optional): Sets the size of the batch.
                        Defaults to 32.
            num_workers (int, optional): How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
                        Defaults to 0.
        """
        self.batch_size_test = batch_size

        if self.dataset_test is not None:
            _log.info("Loading independent testing dataset...")

            self.test_loader = DataLoader(
                self.dataset_test,
                batch_size=self.batch_size_test,
                num_workers=num_workers,
                pin_memory=self.cuda
            )
            _log.info("Testing set loaded\n")
        else:
            _log.error("No test dataset provided.")
            raise ValueError("No test dataset provided.")

        with self._output_exporters:

            # Run test
            self._eval(self.test_loader, self.epoch_saved_model, "testing")

    def _load_params(self):
        """
        Loads the parameters of a pretrained model
        """

        state = torch.load(self.pretrained_model_path)

        self.target = state["target"]
        self.batch_size_train = state["batch_size_train"]
        self.batch_size_test = state["batch_size_test"]
        self.val_size = state["val_size"]
        self.test_size = state["test_size"]
        self.lr = state["lr"]
        self.weight_decay = state["weight_decay"]
        self.subset = state["subset"]
        self.class_weights = state["class_weights"]
        self.task = state["task"]
        self.classes = state["classes"]
        self.shuffle = state["shuffle"]
        self.optimizer = state["optimizer"]
        self.opt_loaded_state_dict = state["optimizer_state"]
        self.lossfunction = state["lossfunction"]
        self.model_load_state_dict = state["model_state"]
        self.clustering_method = state["clustering_method"]
        self.node_features = state["node_features"]
        self.edge_features = state["edge_features"]
        self.features = state["features"]
        self.cuda = state["cuda"]
        self.ngpu = state["ngpu"]

    def _save_model(self):
        """
        Saves the model to a file.

        Args:
            filename (str, optional): Name of the file. Defaults to None.
        """
        state = {
            "model_state": self.model.state_dict(),
            "optimizer": self.optimizer,
            "optimizer_state": self.optimizer.state_dict(),
            "lossfunction": self.lossfunction,
            "target": self.target,
            "task": self.task,
            "classes": self.classes,
            "class_weights": self.class_weights,
            "batch_size_train": self.batch_size_train,
            "batch_size_test": self.batch_size_test,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "subset": self.subset,
            "shuffle": self.shuffle,
            "clustering_method": self.clustering_method,
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "features": self.features,
            "cuda": self.cuda,
            "ngpu": self.ngpu
        }

        return state


def _divide_dataset(dataset: Union[GraphDataset, GridDataset], splitsize: Optional[Union[float, int]] = None) -> \
        Union[Tuple[GraphDataset, GraphDataset], Tuple[GridDataset, GridDataset]]:

    """Divides the dataset into a training set and an evaluation set

    Args:
        dataset (Union[:class:`GraphDataset`, :class:`GridDataset`]): Input dataset to be split into training and validation data.
        splitsize (Optional[Union[float, int]], optional): Fraction of dataset (if float) or number of datapoints (if int) to use for validation.
            Defaults to None.
    """

    if splitsize is None:
        splitsize = 0.25
    full_size = len(dataset)

    # find number of datapoints to include in training dataset
    if isinstance (splitsize, float):
        n_split = int(splitsize * full_size)
    elif isinstance (splitsize, int):
        n_split = splitsize
    else:
        raise TypeError (f"type(splitsize) must be float, int or None ({type(splitsize)} detected.)")

    # raise exception if no training data or negative validation size
    if n_split >= full_size or n_split < 0:
        raise ValueError (f"invalid splitsize: {n_split}\n\t" +
            f"splitsize must be a float between 0 and 1 OR an int smaller than the size of the dataset ({full_size} datapoints)")

    if splitsize == 0:  # i.e. the fraction of splitsize was so small that it rounded to <1 datapoint
        dataset_main = dataset
        dataset_split = None
    else:
        indices = np.arange(full_size)
        np.random.shuffle(indices)

        dataset_main = copy.deepcopy(dataset)
        dataset_main.index_entries = [dataset.index_entries[i] for i in indices[n_split:]]

        dataset_split = copy.deepcopy(dataset)
        dataset_split.index_entries = [dataset.index_entries[i] for i in indices[:n_split]]

    return dataset_main, dataset_split
