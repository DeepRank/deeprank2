from time import time
from typing import List, Optional
import logging
import warnings
from tqdm import tqdm
import h5py
import copy
import numpy as np
import os
import torch
from torch import nn
from torch.nn import MSELoss
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from deeprankcore.models.metrics import MetricsExporterCollection, MetricsExporter, ConciseOutputExporter
from deeprankcore.community_pooling import community_detection, community_pooling
from deeprankcore.domain import targettypes as targets

_log = logging.getLogger(__name__)


class Trainer():

    def __init__(self, # pylint: disable=too-many-arguments
                 dataset_train = None,
                 dataset_val = None,
                 dataset_test = None,
                 Net = None,
                 val_size = None,
                 class_weights = None,
                 pretrained_model = None,
                 batch_size = 32,
                 shuffle = True,
                 transform_sigmoid: Optional[bool] = False,
                 metrics_exporters: Optional[List[MetricsExporter]] = None,
                 output_dir = './metrics'):
        """Class from which the network is trained, evaluated and tested

        Args:
            dataset_train (HDF5DataSet object, required): training set used during training.
                Can't be None if pretrained_model is also None. Defaults to None.
            dataset_val (HDF5DataSet object, optional): evaluation set used during training.
                Defaults to None. If None, training set will be split randomly into training set and
                validation set during training, using val_size parameter
            dataset_test (HDF5DataSet object, optional): independent evaluation set. Defaults to None.
            Net (function, required): neural network class (ex. GINet, Foutnet etc.).
                It should subclass torch.nn.Module, and it shouldn't be specific to regression or classification
                in terms of output shape (Trainer class takes care of formatting the output shape according to the task).
                More specifically, in classification task cases, softmax shouldn't be used as the last activation function.
            val_size (float or int, optional): fraction of dataset (if float) or number of datapoints (if int)
                to use for validation.
                - Should be set to 0 if no validation set is needed.
                - Should be not set (None) if dataset_val is not None.
                Defaults to None, and it is set to 0.25 in _DivideDataSet function if no dataset_val is provided.
            class_weights ([list or bool], optional): weights provided to the cross entropy loss function.
                    The user can either input a list of weights or let DeepRanl-GNN (True) define weights
                    based on the dataset content. Defaults to None.
            pretrained_model (str, optional): path to pre-trained model. Defaults to None.
            batch_size (int, optional): defaults to 32.
            shuffle (bool, optional): shuffle the dataloaders data. Defaults to True.
            transform_sigmoid: whether or not to apply a sigmoid transformation to the output (for regression only). 
                This can speed up the optimization and puts the value between 0 and 1.
            metrics_exporters: the metrics exporters to use for generating metrics output
            output_dir: location for metrics file (see ConciseOutputExporter class)
        """
        if metrics_exporters is not None:
            self._metrics_exporters = MetricsExporterCollection(
                *metrics_exporters)
                
        else:
            self._metrics_exporters = MetricsExporterCollection()

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.complete_exporter = ConciseOutputExporter(self.output_dir)

        if (val_size is not None) and (dataset_val is not None):
            raise ValueError("Because a validation dataset has been assigned to dataset_val, val_size should not be used.")

        if pretrained_model is None:

            if dataset_train is None:
                raise ValueError("No pretrained model uploaded. You need to upload a training dataset.")
            
            if Net is None:
                raise ValueError("No pretrained model uploaded. You need to upload a neural network class to be trained.")

            self.target = dataset_train.target  # already defined in HDF5DatSet object
            self.task = dataset_train.task
            self.classes = dataset_train.classes
            self.classes_to_idx = dataset_train.classes_to_idx
            self.optimizer = None
            self.batch_size = batch_size
            self.val_size = val_size            # if None, will be set to 0.25 in _DivideDataSet function
            self.class_weights = class_weights

            self.shuffle = shuffle
            self.transform_sigmoid = transform_sigmoid

            self.subset = dataset_train.subset
            self.node_feature = dataset_train.node_feature
            self.edge_feature = dataset_train.edge_feature
            self.cluster_nodes = dataset_train.clustering_method
            self.epoch_saved_model = None

            self._load_model(dataset_train, dataset_val, dataset_test, Net)

        else:
            self._load_params(pretrained_model)

            if dataset_train is not None:
                warnings.warn("Pretrained model loaded. dataset_train will be ignored.")
            if dataset_val is not None:
                warnings.warn("Pretrained model loaded. dataset_val will be ignored.")
            if Net is None:
                raise ValueError("No neural network class found. Please add it for \
                    completing the loading of the pretrained model.")

            if dataset_test is not None:
                self._load_pretrained_model(dataset_test, Net)
            else:
                raise ValueError("No dataset_test found. Please add it for evaluating the pretrained model.")

    def configure_optimizers(self, optimizer = None, lr = 0.001, weight_decay = 1e-05):

        """Configure optimizer and its main parameters.
        Parameters
        ----------
        optimizer (optional) : object from torch.optim
            PyTorch optimizer. Defaults to Adam.
        lr (optional) : float
            Learning rate. Defaults to 0.01.
        weight_decay (optional) : float
            Weight decay (L2 penalty). Weight decay is fundamental for GNNs, otherwise, parameters can become too big and
            the gradient may explode. Defaults to 1e-05.
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

    def _load_pretrained_model(self, dataset_test, Net):
        """
        Loads pretrained model

        Args:
            dataset_test: HDF5DataSet object to be tested with the model
            Net (function): neural network
        """

        if self.cluster_nodes is not None: 
            self._PreCluster(dataset_test, method=self.cluster_nodes)

        self.test_loader = DataLoader(dataset_test)

        _log.info("Testing set loaded\n")
        
        self._put_model_to_device(dataset_test, Net)

        self.set_loss()

        # load the model and the optimizer state
        self.optimizer.load_state_dict(self.opt_loaded_state_dict)
        self.model.load_state_dict(self.model_load_state_dict)

    def _load_model(self, dataset_train, dataset_val, dataset_test, Net):
        
        """
        Loads model

        Args:
            dataset_train (str): HDF5DataSet object, training set used during training phase.
            dataset_val (str): HDF5DataSet object, evaluation set used during training phase.
            dataset_eval (str): HDF5DataSet object, the independent evaluation set used after
                training phase. 
            Net (function): neural network.

        Raises:
            ValueError: Invalid node clustering method.
        """

        if self.cluster_nodes is not None:
            if self.cluster_nodes in ('mcl', 'louvain'):
                _log.info("Loading clusters")
                self._PreCluster(dataset_train, method=self.cluster_nodes)

                if dataset_val is not None:
                    self._PreCluster(dataset_val, method=self.cluster_nodes)
                else:
                    _log.warning("No validation dataset given. Randomly splitting training set in training set and validation set.")
                    dataset_train, dataset_val = _DivideDataSet(
                        dataset_train, val_size=self.val_size)
            else:
                raise ValueError(
                    "Invalid node clustering method. \n\t"
                    "Please set cluster_nodes to 'mcl', 'louvain' or None. Default to 'mcl' \n\t")

        # dataloader
        self.train_loader = DataLoader(
            dataset_train, batch_size=self.batch_size, shuffle=self.shuffle
        )
        _log.info("Training set loaded\n")

        if dataset_val is not None:
            self.valid_loader = DataLoader(
                dataset_val, batch_size=self.batch_size, shuffle=self.shuffle
            )
            _log.info("Validation set loaded\n")
        else:
            self.valid_loader = None

        # independent validation dataset
        if dataset_test is not None:
            _log.info("Loading independent testing dataset...")

            if self.cluster_nodes in ('mcl', 'louvain'):
                self._PreCluster(dataset_test, method=self.cluster_nodes)

            self.test_loader = DataLoader(
                dataset_test, batch_size=self.batch_size, shuffle=self.shuffle
            )
            _log.info("Testing set loaded\n")

        else:
            _log.info("No independent testing set loaded")
            self.test_loader = None

        self._put_model_to_device(dataset_train, Net)

        # optimizer
        self.configure_optimizers()

        self.set_loss()

    def _put_model_to_device(self, dataset, Net):
        """
        Puts the model on the available device

        Args:
            dataset (str): HDF5DataSet object
            Net (function): Neural Network

        Raises:
            ValueError: Incorrect output shape
        """
        # get the device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        _log.info("Device set to %s.", self.device)
        if self.device.type == 'cuda':
            _log.info("cuda device name is %s", torch.cuda.get_device_name(0))

        self.num_edge_features = len(self.edge_feature)

        # the target values are optional
        if dataset.get(0).y is not None:

            target_shape = dataset.get(0).y.shape[0]
        else:
            target_shape = None

        # regression mode
        if self.task == targets.REGRESS:

            self.output_shape = 1

            self.model = Net(
                dataset.get(0).num_features,
                self.output_shape,
                self.num_edge_features).to(
                self.device)

        # classification mode
        elif self.task == targets.CLASSIF:

            self.output_shape = len(self.classes)

            self.model = Net(
                dataset.get(0).num_features,
                self.output_shape,
                self.num_edge_features).to(
                self.device)

        # check for compatibility
        for metrics_exporter in self._metrics_exporters:
            if not metrics_exporter.is_compatible_with(self.output_shape, target_shape):
                raise ValueError(f"metrics exporter of type {type(metrics_exporter)} "
                                 f"is not compatible with output shape {self.output_shape} "
                                 f"and target shape {target_shape}")

    def set_loss(self):
        """Sets the loss function (MSE loss for regression/ CrossEntropy loss for classification)."""
        if self.task == targets.REGRESS:
            self.loss = MSELoss()

        elif self.task == targets.CLASSIF:

            # assign weights to each class in case of unbalanced dataset
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
        nepoch: Optional[int] = 1,
        validate: Optional[bool] = False,
        save_model: Optional[str] = 'last',
        model_path: Optional[str] = None,
    ):
        """
        Trains the model

        Args:
            nepoch (int, optional): number of epochs. Defaults to 1.
            validate (bool, optional): perform validation. If True, there must be
                a validation set. Defaults to False.
            save_model (last, best, optional): save the model. Defaults to 'last'
            hdf5 (str, optional): hdf5 output file
        """

        train_losses = []
        valid_losses = []

        if model_path is None:
            model_path = f't{self.task}_y{self.target}_b{str(self.batch_size)}_e{str(nepoch)}_lr{str(self.lr)}_{str(nepoch)}.pth.tar'

        with self._metrics_exporters:

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

                # Sets the module in training mode
                self.model.train()

                loss_ = self._epoch(epoch, "training")

                train_losses.append(loss_)

                # Validate the model
                if validate:

                    loss_ = self._eval(self.valid_loader, epoch, "validation")

                    valid_losses.append(loss_)

                    # save the best model (i.e. lowest loss value on validation
                    # data)
                    if save_model == 'best':

                        if min(valid_losses) == loss_:
                            self.save_model(model_path)
                            self.epoch_saved_model = epoch
                else:
                    # if no validation set, saves the best performing model on
                    # the training set
                    if save_model == 'best':
                        if min(train_losses) == loss_: # noqa
                            _log.warning(
                                """The training set is used both for learning and model selection.
                                            This may lead to training set data overfitting.
                                            We advice you to use an external validation set.""")

                            self.save_model(model_path)
                            self.epoch_saved_model = epoch
                            _log.info(f'Best model saved at epoch # {self.epoch_saved_model}')

            # Save the last model
            if save_model == 'last':
                self.save_model(model_path)
                self.epoch_saved_model = epoch
                _log.info(f'Last model saved at epoch # {self.epoch_saved_model}')

            self.complete_exporter.save_all_metrics()

    def test(self, dataset_test=None):
        """
        Tests the model

        Args:
            dataset_test (HDF5Dataset object, required): dataset for testing the model
        """

        with self._metrics_exporters:
            # Loads the test dataset if provided
            if dataset_test is not None:

                if self.cluster_nodes in ('mcl', 'louvain'):
                    self._PreCluster(dataset_test, method=self.cluster_nodes)

                self.test_loader = DataLoader(
                    dataset_test, batch_size=self.batch_size, shuffle=self.shuffle
                )

            elif (dataset_test is None) and (self.test_loader is None):
                raise ValueError("No test dataset provided.")
                
            # Run test
            self._eval(self.test_loader, 0, "testing")

            self.complete_exporter.save_all_metrics()

    def _eval( # pylint: disable=too-many-locals
            self,
            loader: DataLoader,
            epoch_number: int,
            pass_name: str) -> float:
        """
        Evaluates the model

        Args:
            loader: data to evaluate on
            epoch_number: number for this epoch, used for storing the metrics
            pass_name: 'training', 'validation' or 'testing'

        Returns:
            running loss:
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
        for _, data_batch in enumerate(loader):

            data_batch = data_batch.to(self.device)
            pred = self.model(data_batch)
            pred, data_batch.y = self._format_output(pred, data_batch.y)

            # Check if a target value was provided (i.e. benchmarck scenario)
            if data_batch.y is not None:
                target_vals += data_batch.y.tolist()
                loss_ = loss_func(pred, data_batch.y)

                count_predictions += pred.shape[0]
                sum_of_losses += loss_.detach().item() * pred.shape[0]

            # Get the outputs for export
            # Remember that non-linear activation is automatically applied in CrossEntropyLoss
            if self.task == targets.CLASSIF:
                pred = F.softmax(pred.detach(), dim=1)
            else:
                pred = pred.detach().reshape(-1)

            outputs += pred.tolist()

            # get the data
            entry_names += data_batch['mol']

        dt = time() - t0

        if count_predictions > 0:
            eval_loss = sum_of_losses / count_predictions
        else:
            eval_loss = 0.0

        self._metrics_exporters.process(
            pass_name, epoch_number, entry_names, outputs, target_vals)

        self.complete_exporter.epoch_process(
            pass_name,
            epoch_number,
            entry_names,
            outputs,
            target_vals,
            eval_loss)

        self._log_epoch_data(pass_name, eval_loss, dt)

        return eval_loss

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
        for _, data_batch in enumerate(self.train_loader):

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

            # get the data
            entry_names += data_batch['mol']

        dt = time() - t0

        if count_predictions > 0:
            epoch_loss = sum_of_losses / count_predictions
        else:
            epoch_loss = 0.0

        self._metrics_exporters.process(
            pass_name, epoch_number, entry_names, outputs, target_vals)

        self.complete_exporter.epoch_process(
            pass_name,
            epoch_number,
            entry_names,
            outputs,
            target_vals,
            epoch_loss)

        self._log_epoch_data(pass_name, epoch_loss, dt)

        return epoch_loss

    @staticmethod
    def _log_epoch_data(stage, loss, time):
        """
        Prints the data of each epoch

        Args:
            stage (str): train or valid
            epoch (int): epoch number
            loss (float): loss during that epoch
            time (float): timing of the epoch
        """
        _log.info(f'{stage} loss {loss} | time {time}')

    def _format_output(self, pred, target=None):
        """Format the network output depending on the task (classification/regression)."""

        if (self.task == targets.CLASSIF) and (target is not None):
            # For categorical cross entropy, the target must be a one-dimensional tensor
            # of class indices with type long and the output should have raw, unnormalized values
            target = torch.tensor(
                [self.classes_to_idx[int(x)] for x in target]
            ).to(self.device)

        elif self.task == targets.REGRESS:
            if self.transform_sigmoid is True:

                # Sigmoid(x) = 1 / (1 + exp(-x))
                pred = torch.sigmoid(pred.reshape(-1))

            else:
                pred = pred.reshape(-1)

        if target is not None:
            target = target.to(self.device)

        return pred, target


    def save_model(self, filename='model.pth.tar'):
        """
        Saves the model to a file

        Args:
            filename (str, optional): name of the file. Defaults to 'model.pth.tar'.
        """
        state = {
            "model_state": self.model.state_dict(),
            "optimizer": self.optimizer,
            "optimizer_state": self.optimizer.state_dict(),
            "node": self.node_feature,
            "edge": self.edge_feature,
            "target": self.target,
            "task": self.task,
            "classes": self.classes,
            "class_weight": self.class_weights,
            "batch_size": self.batch_size,
            "val_size": self.val_size,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "subset": self.subset,
            "shuffle": self.shuffle,
            "cluster_nodes": self.cluster_nodes,
            "transform_sigmoid": self.transform_sigmoid,
        }

        torch.save(state, filename)

    def _load_params(self, filename):
        """
        Loads the parameters of a pretrained model

        Args:
            filename ([type]): [description]

        Returns:
            [type]: [description]
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        state = torch.load(filename, map_location=torch.device(self.device))

        self.node_feature = state["node"]
        self.edge_feature = state["edge"]
        self.target = state["target"]
        self.batch_size = state["batch_size"]
        self.val_size = state["val_size"]
        self.lr = state["lr"]
        self.weight_decay = state["weight_decay"]
        self.subset = state["subset"]
        self.class_weights = state["class_weight"]
        self.task = state["task"]
        self.classes = state["classes"]
        self.shuffle = state["shuffle"]
        self.cluster_nodes = state["cluster_nodes"]
        self.transform_sigmoid = state["transform_sigmoid"]
        self.optimizer = state["optimizer"]
        self.opt_loaded_state_dict = state["optimizer_state"]
        self.model_load_state_dict = state["model_state"]

    def _PreCluster(self, dataset, method):
        """Pre-clusters nodes of the graphs

        Args:
            dataset (HDF5DataSet object)
            method (srt): 'mcl' (Markov Clustering) or 'louvain'
        """
        for fname, mol in tqdm(dataset.index_complexes):

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

            if method.lower() in clust_grp:
                #_log.info(f"Deleting previous data for mol {mol} method {method}")
                del clust_grp[method.lower()]

            method_grp = clust_grp.create_group(method.lower())

            cluster = community_detection(
                data.edge_index, data.num_nodes, method=method
            )
            method_grp.create_dataset("depth_0", data=cluster.cpu())

            data = community_pooling(cluster, data)

            cluster = community_detection(
                data.edge_index, data.num_nodes, method=method
            )
            method_grp.create_dataset("depth_1", data=cluster.cpu())

            f5.close()

def _DivideDataSet(dataset, val_size=None):
    """Divides the dataset into a training set and an evaluation set

    Args:
        dataset (HDF5DataSet): input dataset to be split into training and validation data
        val_size (float or int, optional): fraction of dataset (if float) or number of datapoints (if int) to use for validation. 
            Defaults to 0.25.

    Returns:
        HDF5DataSet: [description]
    """

    if val_size is None:
        val_size = 0.25
    full_size = len(dataset)

    # find number of datapoints to include in training dataset
    if isinstance (val_size, float):
        n_val = int(val_size * full_size)
    elif isinstance (val_size, int):
        n_val = val_size
    else:
        raise TypeError (f"type(val_size) must be float, int or None ({type(val_size)} detected.)")
    
    # raise exception if no training data or negative validation size
    if n_val >= full_size or n_val < 0:
        raise ValueError ("invalid val_size. \n\t" +
            f"val_size must be a float between 0 and 1 OR an int smaller than the size of the dataset used ({full_size})")

    if val_size == 0:
        dataset_train = dataset
        dataset_val = None
    else:
        index = np.arange(full_size)
        np.random.shuffle(index)

        index_train, index_val = index[n_val:], index[:n_val]

        dataset_train = copy.deepcopy(dataset)
        dataset_train.index_complexes = [dataset.index_complexes[i] for i in index_train]

        dataset_val = copy.deepcopy(dataset)
        dataset_val.index_complexes = [dataset.index_complexes[i] for i in index_val]

    return dataset_train, dataset_val