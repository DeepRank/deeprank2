# Quick start
## Data generation

The process of generating graphs takes as input `.pdb` files representing protein-protein structural complexes and the correspondent Position-Specific Scoring Matrices (PSSMs) in the form of `.pssm` files. Query objects describe how the graphs should be built.

```python
from deeprankcore.preprocess import preprocess
from deeprankcore.models.query import ProteinProteinInterfaceResidueQuery
from deeprankcore.feature import bsa, pssm, amino_acid, biopython
from deeprankcore.domain import targettypes as targets

feature_modules = [bsa, pssm, biopython, atomic_contact]

queries = []

# Append data points
queries.append(ProteinProteinInterfaceResidueQuery(
    pdb_path = "1ATN_1w.pdb",
    chain_id1 = "A",
    chain_id2 = "B",
    targets = {
        targets.BINARY: 0
    },
    pssm_paths = {
        "A": "1ATN.A.pdb.pssm",
        "B": "1ATN.B.pdb.pssm"
    }
))
queries.append(ProteinProteinInterfaceResidueQuery(
    pdb_path = "1ATN_2w.pdb",
    chain_id1 = "A",
    chain_id2 = "B",
    targets = {
        targets.BINARY: 1
    },
    pssm_paths = {
        "A": "1ATN.A.pdb.pssm",
        "B": "1ATN.B.pdb.pssm"
    }
))
queries.append(ProteinProteinInterfaceResidueQuery(
    pdb_path = "1ATN_3w.pdb",
    chain_id1 = "A",
    chain_id2 = "B",
    targets = {
        targets.BINARY: 0
    },
    pssm_paths = {
        "A": "1ATN.A.pdb.pssm",
        "B": "1ATN.B.pdb.pssm"
    }
))

# Generate graphs and save them in hdf5 files
# The default creates a number of hdf5 files equals to the cpu cores available
# See deeprankcore.preprocess.preprocess for more details
output_paths = preprocess(feature_modules, queries, "<output_folder>/<prefix_for_outputs>")

```

The user is free to implement his/her own query class. Each implementation requires the `build_graph` method to be present.


## Dataset(s)

Data can be split in sets implementing custom splits according to the specific application. Utility splitting functions are currently under development.

Assuming that the training, validation and testing ids have been chosen (keys of the hdf5 file), then the corresponding graphs can be saved in hdf5 files containing only references (external links) to the original one. For example:

```python

from deeprankcore.DataSet import save_hdf5_keys

save_hdf5_keys("<original_hdf5_path.hdf5>", train_ids, "<train_hdf5_path.hdf5>")
save_hdf5_keys("<original_hdf5_path.hdf5>", valid_ids, "<val_hdf5_path.hdf5>")
save_hdf5_keys("<original_hdf5_path.hdf5>", test_ids, "<test_hdf5_path.hdf5>")
```

Now the HDF5DataSet objects can be defined:

```python
from deeprankcore.DataSet import HDF5DataSet
from deeprankcore.domain.features import nodefeats as Nfeat
from deeprankcore.domain.features import edgefeats as Efeat

node_features = [Nfeat.BSA, Nfeat.RESDEPTH, Nfeat.HSE, Nfeat.INFOCONTENT, Nfeat.PSSM]
edge_features = [Efeat.DISTANCE]

# Creating HDF5DataSet objects
dataset_train = HDF5DataSet(
    hdf5_path = "<train_hdf5_path.hdf5>",
    node_feature = node_features,
    edge_feature = edge_features,
    target = targets.BINARY
)
dataset_val = HDF5DataSet(
    hdf5_path = "<val_hdf5_path.hdf5>",
    node_feature = node_features,
    edge_feature = edge_features,
    target = targets.BINARY
)
dataset_test = HDF5DataSet(
    hdf5_path = "<test_hdf5_path.hdf5>",
    node_feature = node_features,
    edge_feature = edge_features,
    target = targets.BINARY
)
```
## Training

Training can be performed using one of the already existing GNNs, for example GINet:

```python
from deeprankcore.Trainer import Trainer
from deeprankcore.ginet import GINet
from deeprankcore.models.metrics import OutputExporter, ScatterPlotExporter

metrics_output_directory = "./metrics"
metrics_exporters = [OutputExporter(metrics_output_directory)]

trainer = Trainer(
    dataset_train,
    dataset_val,
    dataset_test,
    GINet,
    batch_size = 64,
    metrics_exporters = metrics_exporters
)

trainer.train(nepoch = 50, validate = True)
trainer.test()
trainer.save_model(filename = "<output_model_path.pth.tar>")
```


### Custom GNN

It is also possible to define new network architecture and to specify the optimizer (`torch.optim.Adam` by default) to be used during the training.

```python
import torch 

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class CustomNet(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = SplineConv(d.num_features, 32, dim=2, kernel_size=5)
        self.conv2 = SplineConv(32, 64, dim=2, kernel_size=5)
        self.fc1 = torch.nn.Linear(64, 128)
        self.fc2 = torch.nn.Linear(128, 1)

    def forward(self, data):
        data.x = F.elu(self.conv1(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight)
        data = max_pool(cluster, data)

        data.x = F.elu(self.conv2(data.x, data.edge_index, data.edge_attr))
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight)
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = scatter_mean(x, batch, dim=0)
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


trainer = Trainer(
    dataset_train,
    dataset_val,
    dataset_test,
    CustomNet,
    batch_size = 64,
    metrics_exporters = metrics_exporters
)

trainer.configure_optimizers(torch.optim.Adamax, lr = 0.001, weight_decay = 1e-04)
trainer.train(nepoch=50)
```

## h5x support

After installing  `h5xplorer`  (https://github.com/DeepRank/h5xplorer), you can execute the python file `deeprankcore/h5x/h5x.py` to explorer the connection graph used by deeprank-core. The context menu (right click on the name of the structure) allows to automatically plot the graphs using `plotly` as shown below.

![alt-text](./h5_deeprankcore.png)
