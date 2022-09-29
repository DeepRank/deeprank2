from time import time
from typing import List, Optional, Union
import os
import logging

# torch import
import torch
from torch import nn
from torch.nn import MSELoss
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

# deeprankcore import
from deeprankcore.models.metrics import MetricsExporterCollection, MetricsExporter
from deeprankcore.DataSet import _DivideDataSet, PreCluster, HDF5DataSet

_log = logging.getLogger(__name__)


class NeuralNet():

    def __init__(self, # pylint: disable=too-many-arguments
                 Net,
                 dataset_train: HDF5DataSet,
                 dataset_val: Optional[HDF5DataSet] = None,
                 dataset_test: Optional[HDF5DataSet] = None,
                 val_size: Optional[Union[float, int]] = None,
                 lr: float = 0.01,
                 weight_decay: int = 1e-05,
                 batch_size: int = 32,
                 class_weights: Optional[Union[list,bool]] = None,
                 classes: Optional[list] = None,
                 pretrained_model: Optional[str] = None,
                 shuffle: bool = True,
                 transform_sigmoid: Optional[bool] = False,
                 metrics_exporters: Optional[List[MetricsExporter]] = None):
        """Class from which the network is trained, evaluated and tested

        Args:
            Net (function, required): neural network function (ex. GINet, Foutnet etc.)
            
            dataset_train (HDF5DataSet object, required): training set used during training.
            dataset_val (HDF5DataSet object, optional): evaluation set used during training.
                Defaults to None. If None, training set will be split randomly into training set and
                validation set during training, using val_size parameter
            dataset_test (HDF5DataSet object, optional): independent evaluation set. Defaults to None.
            val_size (float or int, optional): fraction of dataset (if float) or number of datapoints (if int) to use for validation. 
                Defaults to 0.25 in _DivideDataSet function.
            
            lr (float, optional): learning rate. Defaults to 0.01.
            weight_decay (float, optional): weight decay (L2 penalty). Weight decay is 
                    fundamental for GNNs, otherwise, parameters can become too big and
                    the gradient may explode. Defaults to 1e-05.
            batch_size (int, optional): defaults to 32.
            
            class_weights ([list or bool], optional): weights provided to the cross entropy loss function.
                    The user can either input a list of weights or let DeepRanl-GNN (True) define weights
                    based on the dataset content. Defaults to None.
            classes (list, optional): define the dataset target classes in classification mode. Defaults to [0, 1].

            pretrained_model (str, optional): path to pre-trained model. Defaults to None.
            shuffle (bool, optional): shuffle the dataloaders data. Defaults to True.
            transform_sigmoid: whether or not to apply a sigmoid transformation to the output (for regression only). 
            This can speed up the optimization and puts the value between 0 and 1.
            metrics_exporters: the metrics exporters to use for generating metrics output
        """

        if metrics_exporters is not None:
            self._metrics_exporters = MetricsExporterCollection(
                *metrics_exporters)
        else:
            self._metrics_exporters = MetricsExporterCollection()

        if pretrained_model is None:
            self.target = dataset_train.target
            self.task = dataset_train.task
            self.lr = lr
            self.weight_decay = weight_decay
            self.batch_size = batch_size
            self.val_size = val_size    # if None, will be set to 0.25 in _DivideDataSet function

            self.class_weights = class_weights
            if classes is None:
                self.classes = [0, 1]
            else:
                self.classes = classes

            self.shuffle = shuffle
            self.transform_sigmoid = transform_sigmoid

            self.subset = dataset_train.subset
            self.node_feature = dataset_train.node_feature
            self.edge_feature = dataset_train.edge_feature
            self.cluster_nodes = dataset_train.clustering_method

            self.load_model(dataset_train, dataset_val, dataset_test, Net)

        else:
            self.load_params(pretrained_model)
            if dataset_test is not None:
                self.load_pretrained_model(dataset_test, Net)
            else:
                raise ValueError("A HDF5DataSet object needs to be passed as a test set for evaluating the pre-trained model.")

    def load_pretrained_model(self, dataset_test, Net):
        """
        Loads pretrained model

        Args:
            dataset_test: HDF5DataSet object to be tested with the model
            Net (function): neural network
        """
        self.test_loader = DataLoader(dataset_test)

        if self.cluster_nodes is not None: 
            PreCluster(dataset_test, method=self.cluster_nodes)

        print("Test set loaded")
        
        self.put_model_to_device(dataset_test, Net)

        self.set_loss()

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # load the model and the optimizer state if we have one
        self.optimizer.load_state_dict(self.opt_loaded_state_dict)
        self.model.load_state_dict(self.model_load_state_dict)

    def load_model(self, dataset_train, dataset_val, dataset_test, Net):
        
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
                print("Loading clusters")
                PreCluster(dataset_train, method=self.cluster_nodes)

                if dataset_val is not None:
                    PreCluster(dataset_val, method=self.cluster_nodes)
                else:
                    print("No validation dataset given. Randomly splitting training set in training set and validation set.")
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
        print("Training set loaded")

        self.valid_loader = DataLoader(
            dataset_val, batch_size=self.batch_size, shuffle=self.shuffle
        )
        print("Validation set loaded")

        # independent validation dataset
        if dataset_test is not None:
            print("Loading independent evaluation dataset...")

            if self.cluster_nodes in ('mcl', 'louvain'):
                print("Loading clusters for the evaluation set.")
                PreCluster(dataset_test, method=self.cluster_nodes)

            self.valid_loader = DataLoader(
                dataset_test, batch_size=self.batch_size, shuffle=self.shuffle
            )
            print("Independent validation set loaded !")

        else:
            print("No independent validation set loaded")

        self.put_model_to_device(dataset_train, Net)

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.set_loss()

    def put_model_to_device(self, dataset, Net):
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

        _log.info("device set to : %s", self.device)
        if self.device.type == 'cuda':
            _log.info("cuda device name is %s", torch.cuda.get_device_name(0))

        self.num_edge_features = len(self.edge_feature)

        # the target values are optional
        if dataset.get(0).y is not None:

            target_shape = dataset.get(0).y.shape[0]
        else:
            target_shape = None

        # regression mode
        if self.task == "regress":

            self.output_shape = 1

            self.model = Net(
                dataset.get(0).num_features,
                self.output_shape,
                self.num_edge_features).to(
                self.device)

        # classification mode
        elif self.task == "classif":

            self.classes_to_idx = {
                i: idx for idx, i in enumerate(
                    self.classes)}
            self.idx_to_classes = dict(enumerate(self.classes))
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
        if self.task == "regress":
            self.loss = MSELoss()

        elif self.task == "classif":

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
                print(f"class occurences: {self.weights}")
                self.weights = 1.0 / self.weights
                self.weights = self.weights / self.weights.sum()
                print(f"class weights: {self.weights}")

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
            validate (bool, optional): perform validation. Defaults to False.
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

            self.eval(self.train_loader, 0, "training")
            if validate:
                self.eval(self.valid_loader, 0, "validation")

            # Loop over epochs
            self.data = {}
            for epoch in range(1, nepoch + 1):

                # Train the model
                self.model.train()

                loss_ = self._epoch(epoch, "training")

                train_losses.append(loss_)

                # Validate the model
                if validate:

                    loss_ = self.eval(self.valid_loader, epoch, "validation")

                    valid_losses.append(loss_)

                    # save the best model (i.e. lowest loss value on validation
                    # data)
                    if save_model == 'best':

                        if min(valid_losses) == loss_:
                            self.save_model(model_path)
                else:
                    # if no validation set, saves the best performing model on
                    # the traing set
                    if save_model == 'best':
                        if min(train_losses) == loss_: # noqa
                            _log.warning(
                                """The training set is used both for learning and model selection.
                                            This may lead to training set data overfitting.
                                            We advice you to use an external validation set.""")

                            self.save_model(model_path)

            # Save the last model
            if save_model == 'last':
                self.save_model(model_path)

    def test(self, dataset_test=None):
        """
        Tests the model

        Args:
            dataset_test ([type], optional): HDF5DataSet object for testing
            hdf5 (str, optional): output hdf5 file. Defaults to 'test_data.hdf5'.
        """

        with self._metrics_exporters:

            # Loads the test dataset if provided
            if dataset_test is not None:

                PreCluster(dataset_test, method='mcl')

                self.test_loader = DataLoader(dataset_test)

            else:
                if self.load_pretrained_model is None:
                    raise ValueError(
                        "You need to upload a test dataset \n\t"
                        "\n\t"
                        ">> model.test(test_dataset)\n\t"
                        "if a pretrained network is loaded, you can directly test the model on the loaded dataset :\n\t"
                        ">> model = NeuralNet(dataset_test, gnn, pretrained_model = model_saved, target=None)\n\t"
                        ">> model.test()\n\t")
            self.data = {}

            # Run test
            self.eval(self.test_loader, 0, "testing")

    def eval( # pylint: disable=too-many-locals
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
        self.model.eval()

        loss_func = self.loss

        targets = []
        outputs = []
        entry_names = []

        sum_of_losses = 0
        count_predictions = 0

        t0 = time()
        for _, data_batch in enumerate(loader):

            data_batch = data_batch.to(self.device)
            pred = self.model(data_batch)
            pred, data_batch.y = self.format_output(pred, data_batch.y)

            # Check if a target value was provided (i.e. benchmarck scenario)
            if data_batch.y is not None:
                targets += data_batch.y.tolist()
                loss_ = loss_func(pred, data_batch.y)

                count_predictions += pred.shape[0]
                sum_of_losses += loss_.detach().item() * pred.shape[0]

            # get the outputs for export
            if self.task == 'classif':
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
            pass_name, epoch_number, entry_names, outputs, targets)

        self.log_epoch_data(pass_name, epoch_number, eval_loss, dt)

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

        targets = []
        outputs = []
        entry_names = []

        t0 = time()
        for _, data_batch in enumerate(self.train_loader):

            data_batch = data_batch.to(self.device)
            self.optimizer.zero_grad()
            pred = self.model(data_batch)
            pred, data_batch.y = self.format_output(pred, data_batch.y)

            loss_ = self.loss(pred, data_batch.y)
            loss_.backward()
            self.optimizer.step()

            count_predictions += pred.shape[0]

            # convert mean back to sum
            sum_of_losses += loss_.detach().item() * pred.shape[0]

            targets += data_batch.y.tolist()

            # get the outputs for export
            if self.task == 'classif':
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
            pass_name, epoch_number, entry_names, outputs, targets)
        self.log_epoch_data(pass_name, epoch_number, epoch_loss, dt)

        return epoch_loss

    def compute_class_weights(self):

        targets_all = []
        for batch in self.train_loader:
            targets_all.append(batch.y)

        targets_all = torch.cat(targets_all).squeeze().tolist()
        weights = torch.tensor(
            [targets_all.count(i) for i in self.classes], dtype=torch.float32
        )
        print(f"class occurences: {weights}")
        weights = 1.0 / weights
        weights = weights / weights.sum()
        print(f"class weights: {weights}")
        return weights

    @staticmethod
    def log_epoch_data(stage, epoch, loss, time):
        """
        Prints the data of each epoch

        Args:
            stage (str): train or valid
            epoch (int): epoch number
            loss (float): loss during that epoch
            time (float): timing of the epoch
        """

        _log.info( # pylint: disable=logging-not-lazy
            'Epoch [%04d] : %s loss %e | time %1.2e sec.' % # pylint: disable=consider-using-f-string
            (epoch, stage, loss, time))

    def format_output(self, pred, target=None):
        """Format the network output depending on the task (classification/regression)."""
        if self.task == "classif":
            # pred = F.softmax(pred, dim=1)

            if target is not None:
                target = torch.tensor([self.classes_to_idx[int(x)]
                                       for x in target]).to(self.device)

        elif self.transform_sigmoid is True:

            # Sigmoid(x) = 1 / (1 + exp(-x))
            pred = torch.sigmoid(pred.reshape(-1))

        else:
            pred = pred.reshape(-1)

        return pred, target

    @staticmethod
    def update_name(hdf5, outdir):
        """
        Checks if the file already exists, if so, update the name

        Args:
            hdf5 (str): hdf5 file
            outdir (str): output directory

        Returns:
            str: update hdf5 name
        """
        fname = os.path.join(outdir, hdf5)

        count = 0
        hdf5_name = hdf5.split(".")[0]

        # If file exists, change its name with a number
        while os.path.exists(fname):
            count += 1
            hdf5 = f"{hdf5_name}_{count:03d}.hdf5"
            fname = os.path.join(outdir, hdf5)

        return fname

    def save_model(self, filename='model.pth.tar'):
        """
        Saves the model to a file

        Args:
            filename (str, optional): name of the file. Defaults to 'model.pth.tar'.
        """
        state = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
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

    def load_params(self, filename):
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
        self.task = state["task"]
        self.batch_size = state["batch_size"]
        self.val_size = state["val_size"]
        self.lr = state["lr"]
        self.weight_decay = state["weight_decay"]
        self.subset = state["subset"]
        self.class_weights = state["class_weight"]
        self.classes = state["classes"]
        self.shuffle = state["shuffle"]
        self.cluster_nodes = state["cluster_nodes"]
        self.transform_sigmoid = state["transform_sigmoid"]

        self.opt_loaded_state_dict = state["optimizer"]
        self.model_load_state_dict = state["model"]
