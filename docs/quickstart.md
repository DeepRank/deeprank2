# Quick start
## Data generation

The process of generating graphs takes as input `.pdb` files representing protein-protein structural complexes and the correspondent Position-Specific Scoring Matrices (PSSMs) in the form of `.pssm` files. Query objects describe how the graphs should be built.

```python
from deeprankcore.query import QueryCollection, ProteinProteinInterfaceResidueQuery

queries = QueryCollection()

# Append data points
queries.add(ProteinProteinInterfaceResidueQuery(
    pdb_path = "1ATN_1w.pdb",
    chain_id1 = "A",
    chain_id2 = "B",
    targets = {
        "binary": 0
    },
    pssm_paths = {
        "A": "1ATN.A.pdb.pssm",
        "B": "1ATN.B.pdb.pssm"
    }
))
queries.add(ProteinProteinInterfaceResidueQuery(
    pdb_path = "1ATN_2w.pdb",
    chain_id1 = "A",
    chain_id2 = "B",
    targets = {
        "binary": 1
    },
    pssm_paths = {
        "A": "1ATN.A.pdb.pssm",
        "B": "1ATN.B.pdb.pssm"
    }
))
queries.add(ProteinProteinInterfaceResidueQuery(
    pdb_path = "1ATN_3w.pdb",
    chain_id1 = "A",
    chain_id2 = "B",
    targets = {
        "binary": 0
    },
    pssm_paths = {
        "A": "1ATN.A.pdb.pssm",
        "B": "1ATN.B.pdb.pssm"
    }
))

# Generate graphs and save them in hdf5 files
output_paths = queries.process("<output_folder>/<prefix_for_outputs>")

```

The user is free to implement his/her own query class. Each implementation requires the `build` method to be present.


## Dataset(s)

Data can be split in sets implementing custom splits according to the specific application. Utility splitting functions are currently under development.

Assuming that the training, validation and testing ids have been chosen (keys of the hdf5 file), then the corresponding graphs can be saved in hdf5 files containing only references (external links) to the original one. For example:

```python

from deeprankcore.dataset import save_hdf5_keys

save_hdf5_keys("<original_hdf5_path.hdf5>", train_ids, "<train_hdf5_path.hdf5>")
save_hdf5_keys("<original_hdf5_path.hdf5>", valid_ids, "<val_hdf5_path.hdf5>")
save_hdf5_keys("<original_hdf5_path.hdf5>", test_ids, "<test_hdf5_path.hdf5>")
```

Now the GraphDataset objects can be defined:

```python
from deeprankcore.dataset import GraphDataset

node_features = ["bsa", "res_depth", "hse", "info_content", "pssm"]
edge_features = ["distance"]
target = "binary"

# Creating GraphDataset objects
dataset_train = GraphDataset(
    hdf5_path = "<train_hdf5_path.hdf5>",
    node_features = node_features,
    edge_features = edge_features,
    target = target
)
dataset_val = GraphDataset(
    hdf5_path = "<val_hdf5_path.hdf5>",
    node_features = node_features,
    edge_features = edge_features,
    target = target

)
dataset_test = GraphDataset(
    hdf5_path = "<test_hdf5_path.hdf5>",
    node_features = node_features,
    edge_features = edge_features,
    target = target
)
```
## Training

Let's define a Trainer instance, using for example of the already existing GNNs, GINet:

```python
from deeprankcore.trainer import Trainer
from deeprankcore.ginet import GINet

trainer = Trainer(
    GINet,
    dataset_train,
    dataset_val,
    dataset_test,
    batch_size = 64
)
```

By default, the Trainer class creates the folder `./output` for storing predictions information collected later on during training and testing. `HDF5OutputExporter` is the exporter used by default, but the user can specify any other implemented exporter or implement a custom one.

Optimizer (`torch.optim.Adam` by default) and loss function can be defined by using dedicated functions:

```python
import torch

trainer.configure_optimizers(torch.optim.Adamax, lr = 0.001, weight_decay = 1e-04)

```

Then the Trainer can be trained and tested, and the model can be saved:

```python
trainer.train(nepoch = 50, validate = True)
trainer.test()
trainer.save_model(filename = "<output_model_path.pth.tar>")

```

### Custom GNN

It is also possible to define new network architectures:

```python
import torch 

def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class CustomNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
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
    CustomNet,
    dataset_train,
    dataset_val,
    dataset_test,
    batch_size = 64
)

trainer.train(nepoch=50)
```

## h5x support

After installing  `h5xplorer`  (https://github.com/DeepRank/h5xplorer), you can execute the python file `deeprankcore/h5x/h5x.py` to explorer the connection graph used by deeprankcore. The context menu (right click on the name of the structure) allows to automatically plot the graphs using `plotly` as shown below.

![alt-text](./h5_deeprankcore.png)
