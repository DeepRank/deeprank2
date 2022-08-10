from time import time
from typing import List, Optional
import os
import logging

# torch import
import torch
from torch import nn
from torch.nn import MSELoss
import torch.nn.functional as F
from torch_geometric.data import DataLoader

# deeprankcore import
from deeprankcore.models.metrics import MetricsExporterCollection, MetricsExporter
from deeprankcore.DataSet import DivideDataSet, PreCluster

_log = logging.getLogger(__name__)


class NeuralNet():

    def __init__(self, # pylint: disable=too-many-arguments, too-many-locals
                 dataset,
                 Net,
                 lr=0.01,
                 batch_size=32,
                 percent=None,
                 dataset_eval=None,
                 class_weights=None,
                 task=None,
                 classes=None,
                 threshold=None,
                 pretrained_model=None,
                 shuffle=True,
                 train_valid_split=DivideDataSet,
                 transform_sigmoid: Optional[bool] = False,
                 metrics_exporters: Optional[List[MetricsExporter]] = None):
        """Class from which the network is trained, evaluated and tested

        Args:
            dataset (HDF5DataSet object, required)
            Net (function, required): neural network function (ex. GINet, Foutnet etc.)
            lr (float, optional): learning rate. Defaults to 0.01.
            batch_size (int, optional): defaults to 32.
            percent (list, optional): divides the input dataset into a training and an evaluation set.
                    Defaults to [1.0, 0.0].
            dataset_eval ([type], optional): independent evaluation HDF5DataSet object, optional. Defaults to None.
            class_weights ([list or bool], optional): weights provided to the cross entropy loss function.
                    The user can either input a list of weights or let DeepRanl-GNN (True) define weights
                    based on the dataset content. Defaults to None.
            task (str, optional): 'reg' for regression or 'class' for classification . Defaults to None.
            classes (list, optional): define the dataset target classes in classification mode. Defaults to [0, 1].
            threshold (int, optional): threshold to compute binary classification metrics. Defaults to 4.0.
            pretrained_model (str, optional): path to pre-trained model. Defaults to None.
            shuffle (bool, optional): shuffle the training set. Defaults to True.
            train_valid_split (func, optional): split the dataset in training and validation set. If you want to implement
            your own function to split the dataset, assign it to this parameter. Note that it has to take three parameters
            as input (dataset, percent, and shuffle). Defaults to DivideDataSet func (splitting is done according to percent,
            after having shuffled the dataset).
            transform_sigmoid: whether or not to apply a sigmoid transformation to the output (for regression only). 
            This can speed up the optimization and puts the value between 0 and 1.
            metrics_exporters: the metrics exporters to use for generating metrics output

        Notes:
          - 'task' will default to 'class' if the target is 'bin_class' or 'capri_classes'.
          - 'task' will default to 'reg' if the target is 'irmsd', 'lrmsd', 'fnat' or 'dockQ'.
          - 'task' must be set to either 'class' or 'reg' if 'target' is not one of the conventional names.
          - 'target' must be set to a target value that was actually given to the Query class as input. See: deeprankcore.models.query
        """

        if metrics_exporters is not None:
            self._metrics_exporters = MetricsExporterCollection(
                *metrics_exporters)
        else:
            self._metrics_exporters = MetricsExporterCollection()

        if pretrained_model is None:
            self.target = dataset.target # already defined in HDF5DatSet object
            self.lr = lr
            self.batch_size = batch_size

            if percent is None:
                self.percent = [1.0, 0.0]
            else:
                self.percent = percent

            self.class_weights = class_weights
            self.task = task

            if classes is None:
                self.classes = [0, 1]
            else:
                self.classes = classes

            self.threshold = threshold
            self.shuffle = shuffle
            self.train_valid_split = train_valid_split
            self.transform_sigmoid = transform_sigmoid

            self.index = dataset.index
            self.node_feature = dataset.node_feature
            self.edge_feature = dataset.edge_feature
            self.cluster_nodes = dataset.clustering_method

            if self.task is None:
                if self.target in ["irmsd", "lrmsd", "fnat", "dockQ"]:
                    self.task = "reg"
                elif self.target in ["bin_class", "capri_classes"]:
                    self.task = "class"
                else:
                    raise ValueError(
                        "User target detected -> The task argument is required ('class' or 'reg'). \n\t"
                        "Example: \n\t"
                        ""
                        "model = NeuralNet(dataset, GINet,"
                        "                  target='physiological_assembly',"
                        "                  task='class',"
                        "                  shuffle=True,"
                        "                  percent=[0.8, 0.2])")

            if self.task == "class" and self.threshold is None:
                print(
                    f"the threshold for accuracy computation is set to {self.classes[1]}"
                )
                self.threshold = self.classes[1]
            if self.task == "reg" and self.threshold is None:
                print("the threshold for accuracy computation is set to 0.3")
                self.threshold = 0.3
            self.load_model(dataset, Net, dataset_eval)

        else:
            self.load_params(pretrained_model)
            self.load_pretrained_model(dataset, Net)

    def load_pretrained_model(self, test_dataset, Net):
        """
        Loads pretrained model

        Args:
            dataset: HDF5DataSet object
            Net (function): neural network
        """
        self.test_loader = DataLoader(test_dataset)

        if self.cluster_nodes is not None: 
            PreCluster(test_dataset, method=self.cluster_nodes)

        print("Test set loaded")
        
        self.put_model_to_device(test_dataset, Net)

        self.set_loss()

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # load the model and the optimizer state if we have one
        self.optimizer.load_state_dict(self.opt_loaded_state_dict)
        self.model.load_state_dict(self.model_load_state_dict)

    def load_model(self, dataset, Net, dataset_eval):
        """
        Loads model

        Args:
            dataset (str): HDF5DataSet object corresponding to the training set
            Net (function): neural network
            dataset_eval (str): HDF5DataSet object corresponding to the evaluation set

        Raises:
            ValueError: Invalid node clustering method.
        """

        if self.cluster_nodes is not None:
            if self.cluster_nodes in ('mcl', 'louvain'):
                print("Loading clusters")
                PreCluster(dataset, method=self.cluster_nodes)
            else:
                raise ValueError(
                    "Invalid node clustering method. \n\t"
                    "Please set cluster_nodes to 'mcl', 'louvain' or None. Default to 'mcl' \n\t")

        # divide the dataset
        train_dataset, valid_dataset = self.train_valid_split(
            dataset, percent=self.percent)

        # dataloader
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=self.shuffle
        )
        print("Training set loaded")

        if self.percent[1] > 0.0:
            self.valid_loader = DataLoader(
                valid_dataset, batch_size=self.batch_size, shuffle=self.shuffle
            )
            print("Evaluation set loaded")

        # independent validation dataset
        if dataset_eval is not None:
            print("Loading independent evaluation dataset...")

            if self.cluster_nodes in ('mcl', 'louvain'):
                print("Loading clusters for the evaluation set.")
                PreCluster(dataset_eval, method=self.cluster_nodes)

            self.valid_loader = DataLoader(
                dataset_eval, batch_size=self.batch_size, shuffle=self.shuffle
            )
            print("Independent validation set loaded !")

        else:
            print("No independent validation set loaded")

        self.put_model_to_device(dataset, Net)

        # optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.set_loss()

    def put_model_to_device(self, dataset, Net):
        """
        Puts the model on the available device

        Args:
            dataset (str): path to the hdf5 file(s)
            Net (function): neural network

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
        if self.task == "reg":

            self.output_shape = 1

            self.model = Net(
                dataset.get(0).num_features,
                self.output_shape,
                self.num_edge_features).to(
                self.device)

        # classification mode
        elif self.task == "class":

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
        if self.task == "reg":
            self.loss = MSELoss()

        elif self.task == "class":

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
            threshold (int, optional): threshold use to tranform data into binary values. Defaults to 4.
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
            if self.task == 'class':
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
            if self.task == 'class':
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
        if self.task == "class":
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
            "percent": self.percent,
            "lr": self.lr,
            "index": self.index,
            "shuffle": self.shuffle,
            "threshold": self.threshold,
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
        self.batch_size = state["batch_size"]
        self.percent = state["percent"]
        self.lr = state["lr"]
        self.index = state["index"]
        self.class_weights = state["class_weight"]
        self.task = state["task"]
        self.classes = state["classes"]
        self.threshold = state["threshold"]
        self.shuffle = state["shuffle"]
        self.cluster_nodes = state["cluster_nodes"]
        self.transform_sigmoid = state["transform_sigmoid"]

        self.opt_loaded_state_dict = state["optimizer"]
        self.model_load_state_dict = state["model"]
