from time import time
from typing import List, Optional, Union, Tuple
import logging
from tqdm import tqdm
import h5py
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import MSELoss
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from deeprankcore.utils.exporters import OutputExporterCollection, OutputExporter, HDF5OutputExporter
from deeprankcore.utils.community_pooling import community_detection, community_pooling
from deeprankcore.utils.earlystopping import EarlyStopping
from deeprankcore.domain import targetstorage as targets
from deeprankcore.dataset import GraphDataset, GridDataset

_log = logging.getLogger(__name__)


class Trainer():
    def __init__( # pylint: disable=too-many-arguments
                self,
                neuralnet = None,
                dataset_train: Union[GraphDataset, GridDataset] = None,
                dataset_val: Union[GraphDataset, GridDataset] = None,
                dataset_test: Union[GraphDataset, GridDataset] = None,
                val_size: Union[float, int] = None,
                test_size: Union[float, int] = None,
                class_weights: bool = False,
                pretrained_model: Optional[str] = None,
                batch_size: int = 32,
                shuffle: bool = True,
                output_exporters: Optional[List[OutputExporter]] = None,
            ):
        """Class from which the network is trained, evaluated and tested.

        Args:
            neuralnet (function, optional): Neural network class (ex. :class:`GINet`, :class:`Foutnet` etc.).
                It should subclass :class:`torch.nn.Module`, and it shouldn't be specific to regression or classification
                in terms of output shape (:class:`Trainer` class takes care of formatting the output shape according to the task).
                More specifically, in classification task cases, softmax shouldn't be used as the last activation function.
                Defaults to None.

            dataset_train (:class:`GraphDataset`, optional): Training set used during training.
                Can't be None if pretrained_model is also None. Defaults to None.

            dataset_val (:class:`GraphDataset`, optional): Evaluation set used during training.
                If None, training set will be split randomly into training set and validation set during training, using val_size parameter.
                Defaults to None.

            dataset_test (:class:`GraphDataset`, optional): Independent evaluation set. Defaults to None.

            val_size (Union[float,int], optional): Fraction of dataset (if float) or number of datapoints (if int) to use for validation.
                Only used if dataset_val is not specified. Can be set to 0 if no validation set is needed. Defaults to to 0.25 (in _divide_dataset function).

            test_size (Union[float,int], optional): Fraction of dataset (if float) or number of datapoints (if int) to use for test dataset.
                Only used if dataset_test is not specified. Can be set to 0 if no test set is needed. Defaults to 0 (i.e., no test data).

            class_weights (bool, optional): Assign class weights based on the dataset content. Defaults to False.

            pretrained_model (str, optional): Path to pre-trained model. Defaults to None.

            batch_size (int, optional): Sets the size of the batch. Defaults to 32.

            shuffle (bool, optional): whether to shuffle the dataloaders data. Defaults to True.

            output_exporters (List[OutputExporter], optional): The output exporters to use for saving/exploring/plotting predictions/targets/losses over the
                epochs. If None, defaults to :class:`HDF5OutputExporter`, which saves all the results in an .HDF5 file stored in ./output directory.
                Defaults to None.
        """

        self._init_output_exporters(output_exporters)

        self.neuralnet = neuralnet

        self._init_datasets(dataset_train, dataset_val, dataset_test,
                            val_size, test_size)

        if pretrained_model is None:
            if self.dataset_train is None:
                raise ValueError("No training data specified. Training data is required if there is no pretrained model.")
            if self.neuralnet is None:
                raise ValueError("No neural network specified. Specifying a model framework is required if there is no pretrained model.")

            self.classes = self.dataset_train.classes
            self.classes_to_index = self.dataset_train.classes_to_index
            self.optimizer = None
            self.batch_size = batch_size
            self.class_weights = class_weights
            self.shuffle = shuffle
            self.subset = self.dataset_train.subset
            self.epoch_saved_model = None

            if self.target is None:
                raise ValueError("No target set. You need to choose a target (set in the dataset) for training.")

            self._load_model()
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
        """
        Loads model

        Raises:
            ValueError: Invalid node clustering method.
        """

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

        # dataloader
        self.train_loader = DataLoader(
            self.dataset_train, batch_size=self.batch_size, shuffle=self.shuffle
        )
        _log.info("Training set loaded\n")

        if self.dataset_val is not None:
            self.valid_loader = DataLoader(
                self.dataset_val, batch_size=self.batch_size, shuffle=self.shuffle
            )
            _log.info("Validation set loaded\n")
        else:
            self.valid_loader = None

        # independent validation dataset
        if self.dataset_test is not None:
            _log.info("Loading independent testing dataset...")

            self.test_loader = DataLoader(
                self.dataset_test, batch_size=self.batch_size, shuffle=self.shuffle
            )
            _log.info("Testing set loaded\n")
        else:
            _log.info("No independent testing set loaded")
            self.test_loader = None

        self._put_model_to_device(self.dataset_train)
        self.configure_optimizers()
        self.set_loss()

    def _check_dataset_equivalence(self, dataset_train, dataset_val, dataset_test):

        if dataset_train is None:
            # only check the test dataset
            if dataset_test is None:
                raise ValueError("Please provide at least a train or test dataset")

            if not isinstance(dataset_test, GraphDataset) and not isinstance(dataset_test, GridDataset):
                raise TypeError(f"""test dataset is not the right type {type(dataset_test)}
                                Make sure it's either GraphDataset or GridDataset""")
            return

        # Compare the datasets to each other
        for dataset_other_name, dataset_other in [("validation", dataset_val),
                                                  ("testing", dataset_test)]:
            if dataset_other is not None:

                if dataset_other.target != dataset_train.target:
                    raise ValueError(f"training dataset has target {dataset_train.target} while "
                                     f"{dataset_other_name} dataset has target {dataset_other.target}")

                if dataset_other.task != dataset_other.task:
                    raise ValueError(f"training dataset has task {dataset_train.task} while "
                                     f"{dataset_other_name} dataset has task {dataset_other.task}")

                if dataset_other.classes != dataset_other.classes:
                    raise ValueError(f"training dataset has classes {dataset_train.classes} while "
                                     f"{dataset_other_name} dataset has classes {dataset_other.classes}")

                if isinstance(dataset_train, GraphDataset) and isinstance(dataset_other, GraphDataset):

                    if dataset_other.node_features != dataset_train.node_features:
                        raise ValueError(f"training dataset has node_features {dataset_train.node_features} while "
                                         f"{dataset_other_name} dataset has node_features {dataset_other.node_features}")

                    if dataset_other.edge_features != dataset_train.edge_features:
                        raise ValueError(f"training dataset has edge_features {dataset_train.edge_features} while "
                                         f"{dataset_other_name} dataset has edge_features {dataset_other.edge_features}")

                    if dataset_other.clustering_method != dataset_other.clustering_method:
                        raise ValueError(f"training dataset has clustering method {dataset_train.clustering_method} while "
                                         f"{dataset_other_name} dataset has clustering method {dataset_other.clustering_method}")

                elif isinstance(dataset_train, GridDataset) and isinstance(dataset_other, GridDataset):

                    if dataset_other.features != dataset_train.features:
                        raise ValueError(f"training dataset has features {dataset_train.features} while "
                                         f"{dataset_other_name} dataset has features {dataset_other.features}")

                else:
                    raise TypeError(f"Training and {dataset_other_name} datasets are not the same type.\n"
                                     "Make sure to use only graph or only grid datasets")

    def _load_pretrained_model(self):
        """
        Loads pretrained model
        """

        if self.clustering_method is not None: 
            self._precluster(self.dataset_test)
        self.test_loader = DataLoader(self.dataset_test)
        _log.info("Testing set loaded\n")
        self._put_model_to_device(self.dataset_test)
        self.set_loss()

        # load the model and the optimizer state
        self.optimizer.load_state_dict(self.opt_loaded_state_dict)
        self.model.load_state_dict(self.model_load_state_dict)

    def _precluster(self, dataset: GraphDataset):
        """Pre-clusters nodes of the graphs

        Args:
            dataset (GraphDataset object)
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
            dataset (str): GraphDataset object

        Raises:
            ValueError: Incorrect output shape
        """
        # get the device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        _log.info("Device set to %s.", self.device)
        if self.device.type == 'cuda':
            _log.info("cuda device name is %s", torch.cuda.get_device_name(0))

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

    def set_loss(self):

        """
        Sets the loss function: MSE loss for regression and CrossEntropy loss for classification.
        """

        if self.task == targets.REGRESS:
            self.loss = MSELoss()

        elif self.task == targets.CLASSIF:
            # Assign weights to each class
            self.weights = None
            if self.class_weights:
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

            # Note that non-linear activation is automatically applied in CrossEntropyLoss
            self.loss = nn.CrossEntropyLoss(
                weight=self.weights, reduction="mean")


    def train(
        self,
        nepoch: int = 1,
        patience: Optional[int] = None,
        validate: bool = False,
        save_best_model: Optional[bool] = True,
        output_prefix: Optional[str] = None,
    ):
        """
        Performs the training of the model.

        Args:
            nepoch (int): Maximum number of epochs to run.
                        Default: 1.
            patience (int): Early stopping patience.
                        Training ends if the model has run for this number of epochs without improving the validation loss.
                        Set to None to disable early stopping.
                        Default: None.
            validate (bool): Perform validation on independent data set (requires a validation data set).
                        Default: False.
            save_best_model (bool, optional): 
                        True (default): save the best model (in terms of validation loss).
                        False: save the last model tried.
                        None: do not save at all.
            output_prefix (str, optional): Name under which the model is saved. A description of the model settings is appended to the prefix.
                        Default: 'model'.
        """

        train_losses = []
        valid_losses = []
        early_stopping = EarlyStopping(patience=patience, trace_func=_log.info)

        if output_prefix is None:
            output_prefix = 'model'
        output_file = output_prefix + f'_t{self.task}_y{self.target}_b{str(self.batch_size)}_e{str(nepoch)}_lr{str(self.lr)}_{str(nepoch)}.pth.tar'

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
                    if save_best_model:
                        if min(valid_losses) == loss_:
                            self.save_model(output_file)
                            self.epoch_saved_model = epoch
                            _log.info(f'Best model saved at epoch # {self.epoch_saved_model}')
                else:
                    # if no validation set, save the best performing model on the training set
                    if save_best_model:
                        if min(train_losses) == loss_: # noqa
                            _log.warning(
                                "Training data is used both for learning and model selection, which will to overfitting." +
                                "\n\tIt is preferable to use an independent training and validation data sets.")

                            self.save_model(output_file)
                            self.epoch_saved_model = epoch
                            _log.info(f'Best model saved at epoch # {self.epoch_saved_model}')
                
                if patience:
                    early_stopping(loss_, min(train_losses))
                    if early_stopping.early_stop:
                        _log.info(f"Early stopping at epoch # {epoch}")
                        break

            # Save the last model
            if save_best_model is False:
                self.save_model(output_file)
                self.epoch_saved_model = epoch
                _log.info(f'Last model saved at epoch # {self.epoch_saved_model}')

    def _epoch(self, epoch_number: int, pass_name: str) -> float:
        """
        Runs a single epoch

        Args:
            epoch_number (int)
            pass_name (str): 'training', 'validation' or 'testing'

        Returns:
            running loss
        """

        sum_of_losses = 0
        count_predictions = 0
        target_vals = []
        outputs = []
        entry_names = []
        t0 = time()
        for data_batch in self.train_loader:
            data_batch = data_batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(data_batch)
            pred, data_batch.y = self._format_output(pred, data_batch.y)
            loss_ = self.loss(pred, data_batch.y)
            loss_.backward()
            self.optimizer.step()
            count_predictions += pred.shape[0]

            # convert mean back to sum
            sum_of_losses += loss_.detach().item() * pred.shape[0]
            target_vals += data_batch.y.tolist()

            # Get the outputs for export
            # Remember that non-linear activation is automatically applied in CrossEntropyLoss
            if self.task == targets.CLASSIF:
                pred = F.softmax(pred.detach(), dim=1)
            else:
                pred = pred.detach().reshape(-1)
            outputs += pred.tolist()

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
            loader (Dataloader): data to evaluate on
            epoch_number (int): number for this epoch, used for storing the outputs
            pass_name (str): 'training', 'validation' or 'testing'

        Returns:
            running loss
        """

        # Sets the module in evaluation mode
        self.model.eval()
        loss_func = self.loss
        target_vals = []
        outputs = []
        entry_names = []
        sum_of_losses = 0
        count_predictions = 0
        t0 = time()
        for data_batch in loader:
            data_batch = data_batch.to(self.device)
            pred = self.model(data_batch)
            pred, y = self._format_output(pred, data_batch.y)

            # Check if a target value was provided (i.e. benchmarck scenario)
            if y is not None:
                target_vals += y.tolist()
                loss_ = loss_func(pred, y)
                count_predictions += pred.shape[0]
                sum_of_losses += loss_.detach().item() * pred.shape[0]

            # Get the outputs for export
            # Remember that non-linear activation is automatically applied in CrossEntropyLoss
            if self.task == targets.CLASSIF:
                pred = F.softmax(pred.detach(), dim=1)
            else:
                pred = pred.detach().reshape(-1)
            outputs += pred.tolist()

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
            stage (str): train or valid
            loss (float): loss during that epoch
            time (float): timing of the epoch
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
            ).to(self.device)

        elif self.task == targets.REGRESS:
            pred = pred.reshape(-1)

        if target is not None:
            target = target.to(self.device)

        return pred, target


    def test(self):
        """
        Performs the testing of the model.
        """

        with self._output_exporters:
            # Loads the test dataset if provided
            if self.dataset_test is not None:
                if self.clustering_method in ('mcl', 'louvain'):
                    self._precluster(self.dataset_test)
                self.test_loader = DataLoader(
                    self.dataset_test, batch_size=self.batch_size, shuffle=self.shuffle
                )
            elif self.test_loader is None:
                raise ValueError("No test dataset provided.")

            # Run test
            self._eval(self.test_loader, 0, "testing")

    def _load_params(self):
        """
        Loads the parameters of a pretrained model
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        state = torch.load(self.pretrained_model_path,
                           map_location=torch.device(self.device))

        self.target = state["target"]
        self.batch_size = state["batch_size"]
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
        self.model_load_state_dict = state["model_state"]
        self.clustering_method = state["clustering_method"]
        self.node_features = state["node_features"]
        self.edge_features = state["edge_features"]
        self.features = state["features"]


    def save_model(self, filename='model.pth.tar'):
        """
        Saves the model to a file.

        Args:
            filename (str, optional): Name of the file. Defaults to 'model.pth.tar'.
        """
        state = {
            "model_state": self.model.state_dict(),
            "optimizer": self.optimizer,
            "optimizer_state": self.optimizer.state_dict(),
            "target": self.target,
            "task": self.task,
            "classes": self.classes,
            "class_weights": self.class_weights,
            "batch_size": self.batch_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "subset": self.subset,
            "shuffle": self.shuffle,
            "clustering_method": self.clustering_method,
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "features": self.features
        }

        torch.save(state, filename)


def _divide_dataset(dataset: Union[GraphDataset, GridDataset], splitsize: Union[float, int] = None) -> \
        Union[Tuple[GraphDataset, GraphDataset], Tuple[GridDataset, GridDataset]]:

    """Divides the dataset into a training set and an evaluation set

    Args:
        dataset (deeprank-core dataset object): input dataset to be split into training and validation data

        val_size (float or int, optional): fraction of dataset (if float) or number of datapoints (if int) to use for validation. 
            Defaults to 0.25.
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
