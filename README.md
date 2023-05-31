# DeeprankCore

| Badges | |
|:----:|----|
| **fairness** |  [![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu) [![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6403/badge)](https://bestpractices.coreinfrastructure.org/projects/6403) |
| **package** |  [![PyPI version](https://badge.fury.io/py/deeprankcore.svg)](https://badge.fury.io/py/deeprankcore) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/f3f98b2d1883493ead50e3acaa23f2cc)](https://app.codacy.com/gh/DeepRank/deeprank-core?utm_source=github.com&utm_medium=referral&utm_content=DeepRank/deeprank-core&utm_campaign=Badge_Grade) |
| **docs** | [![Documentation Status](https://readthedocs.org/projects/deeprankcore/badge/?version=latest)](https://deeprankcore.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/450496579.svg)](https://zenodo.org/badge/latestdoi/450496579) |
| **tests** | [![Build Status](https://github.com/DeepRank/deeprank-core/actions/workflows/build.yml/badge.svg)](https://github.com/DeepRank/deeprank-core/actions) ![Linting status](https://github.com/DeepRank/deeprank-core/actions/workflows/linting.yml/badge.svg?branch=main) [![Coverage Status](https://coveralls.io/repos/github/DeepRank/deeprank-core/badge.svg?branch=main)](https://coveralls.io/github/DeepRank/deeprank-core?branch=main) |
| **license** |  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/license/apache-2-0/)  |


## Overview

![alt-text](./deeprankcore.png)


DeeprankCore is a Deep Learning (DL) framework for data mining Protein-Protein Interactions (PPIs) using either Graph Neural Networks (GNNs) or Convolutional Neural Networks (CNNs). It is an improved and unified version of the previously developed [deeprank](https://github.com/DeepRank/deeprank) and [Deeprank-GNN](https://github.com/DeepRank/Deeprank-GNN).

DeeprankCore contains useful APIs for pre-processing PPI data, computing features and targets, as well as training and testing GNN and CNN models.

Main features:
- Predefined atom-level and residue-level PPI feature types
  - e.g. atomic density, vdw energy, residue contacts, PSSM, etc.
- Predefined target type
  - e.g. binary class, CAPRI categories, DockQ, RMSD, FNAT, etc.
- Flexible definition of both new features and targets
- Graphs and grids feature mapping
- Efficient data storage in HDF5 format
- Support both classification and regression (based on [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/))

DeeprankCore extensive documentation can be found [here](https://deeprankcore.rtfd.io/).

## Table of contents

- [DeeprankCore](#deeprankcore)
  - [Overview](#overview)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
    - [Dependencies](#dependencies)
    - [Deeprank-Core Package](#deeprank-core-package)
  - [Documentation](#documentation)
  - [Quick start](#quick-start)
    - [Data mapping](#data-mapping)
    - [Datasets](#datasets)
      - [GraphDataset](#graphdataset)
      - [GridDataset](#griddataset)
    - [Training](#training)
  - [h5x support](#h5x-support)
  - [Package development](#package-development)

## Installation

### Dependencies

Before installing deeprankcore you need to install:

 * [reduce](https://github.com/rlabduke/reduce): follow the instructions in the README of the reduce repository.
    * **How to build it without sudo privileges on a Linux machine**. After having run `make` in the reduce/ root directory, go to reduce/reduce_src/Makefile and modify `/usr/local/` to a folder in your home directory, such as `/home/user_name/apps`. Note that such a folder needs to be added to the PATH in the `.bashrc` file. Then run `make install` from reduce/. 
 * [msms](https://ssbio.readthedocs.io/en/latest/instructions/msms.html): `conda install -c bioconda msms`. *For MacOS with M1 chip users*: you can follow [these instructions](https://ssbio.readthedocs.io/en/latest/instructions/msms.html).
 * [DSSP 4](https://swift.cmbi.umcn.nl/gv/dssp/): 
    * on ubuntu 22.04 or newer: `sudo apt-get install dssp`
    * on older versions of ubuntu or on mac, or lacking sudo sudo priviliges: install from [here](https://github.com/pdb-redo/dssp), following the instructions listed.
   * CPU only: `conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch`
   * if using GPU: `conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia`
 * [pytorch-geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html): `conda install pyg -c pyg`
 * [Dependencies for pytorch geometric from wheels](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-wheels): `pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html`. 
    - Here, `${TORCH}` and `${CUDA}` should be replaced by the pytorch and CUDA versions installed. You can find these using:
      - `python -c "import torch; print(torch.__version__)"` and
      - `python -c "import torch; print(torch.version.cuda)"`
    - For example: `https://data.pyg.org/whl/torch-2.0.0+cpu.html`
 * Only if you have a MacOS with M1 chip, additional steps are needed:
    * `conda install pytables`
    * See [this](https://stackoverflow.com/questions/30145751/python3-cant-find-and-import-pyqt5) solution to install PyQt5 or run `conda install pyqt`

### Deeprank-Core Package

Once the dependencies installed, you can install the latest release of deeprankcore using the PyPi package manager:

```
pip install deeprankcore
```

You can get all the new developments by cloning the repo and installing the code with

```
git clone https://github.com/DeepRank/deeprank-core
cd deeprank-core
pip install -e ./
```

## Documentation

More extensive and detailed documentation can be found [here](https://deeprankcore.rtfd.io/).

## Quick start

### Data mapping

For each protein-protein complex, a query can be created and added to the `QueryCollection` object, to be processed later on. Different types of queries exist, based on the molecular resolution needed:
- In a `ProteinProteinInterfaceResidueQuery` each node represents one amino acid residue
- In a `ProteinProteinInterfaceAtomicQuery` each node represents one atom within the amino acid residues.
A query takes as inputs:
- a `.pdb` file, representing the protein-protein structural complex
- the ids of the two chains composing the complex, and 
- the correspondent Position-Specific Scoring Matrices (PSSMs), in the form of `.pssm` files.

```python
from deeprankcore.query import QueryCollection, ProteinProteinInterfaceResidueQuery

queries = QueryCollection()

# Append data points
queries.add(ProteinProteinInterfaceResidueQuery(
    pdb_path = "tests/data/pdb/1ATN/1ATN_1w.pdb",
    chain_id1 = "A",
    chain_id2 = "B",
    targets = {
        "binary": 0
    },
    pssm_paths = {
        "A": "tests/data/pssm/1ATN/1ATN.A.pdb.pssm",
        "B": "tests/data/pssm/1ATN/1ATN.B.pdb.pssm"
    }
))
queries.add(ProteinProteinInterfaceResidueQuery(
    pdb_path = "tests/data/pdb/1ATN/1ATN_2w.pdb",
    chain_id1 = "A",
    chain_id2 = "B",
    targets = {
        "binary": 1
    },
    pssm_paths = {
        "A": "tests/data/pssm/1ATN/1ATN.A.pdb.pssm",
        "B": "tests/data/pssm/1ATN/1ATN.B.pdb.pssm"
    }
))
queries.add(ProteinProteinInterfaceResidueQuery(
    pdb_path = "tests/data/pdb/1ATN/1ATN_3w.pdb",
    chain_id1 = "A",
    chain_id2 = "B",
    targets = {
        "binary": 0
    },
    pssm_paths = {
        "A": "tests/data/pssm/1ATN/1ATN.A.pdb.pssm",
        "B": "tests/data/pssm/1ATN/1ATN.B.pdb.pssm"
    }
))

```

The user is free to implement a custom query class. Each implementation requires the `build` method to be present.

The queries can then be processed into 3D-graphs only or both 3D-graphs and 3D-grids, depending on which kind of network will be used later for training. 

```python
from deeprankcore.features import components, conservation, contact, exposure, irc, surfacearea
from deeprankcore.utils.grid import GridSettings, MapMethod

feature_modules = [components, conservation, contact, exposure, irc, surfacearea]

# Save data into 3D-graphs only
hdf5_paths = queries.process(
    "<output_folder>/<prefix_for_outputs>",
    feature_modules = feature_modules)

# Save data into 3D-graphs and 3D-grids
hdf5_paths = queries.process(
    "<output_folder>/<prefix_for_outputs>",
    feature_modules = feature_modules,
    grid_settings = GridSettings(
        # the number of points on the x, y, z edges of the cube
        points_counts = [20, 20, 20],
        # x, y, z sizes of the box in Å
        sizes = [1.0, 1.0, 1.0]),
    grid_map_method = MapMethod.GAUSSIAN)
```

### Datasets

Data can be split in sets implementing custom splits according to the specific application. Assuming that the training, validation and testing ids have been chosen (keys of the HDF5 file/s), then the `DeeprankDataset` objects can be defined.

#### GraphDataset

For training GNNs the user can create a `GraphDataset` instance:

```python
from deeprankcore.dataset import GraphDataset

node_features = ["bsa", "res_depth", "hse", "info_content", "pssm"]
edge_features = ["distance"]
target = "binary"

# Creating GraphDataset objects
dataset_train = GraphDataset(
    hdf5_path = hdf5_paths,
    subset = train_ids, 
    node_features = node_features,
    edge_features = edge_features,
    target = target
)
dataset_val = GraphDataset(
    hdf5_path = hdf5_paths,
    subset = valid_ids, 
    node_features = node_features,
    edge_features = edge_features,
    target = target

)
dataset_test = GraphDataset(
    hdf5_path = hdf5_paths,
    subset = test_ids, 
    node_features = node_features,
    edge_features = edge_features,
    target = target
)
```

#### GridDataset

For training CNNs the user can create a `GridDataset` instance:

```python
from deeprankcore.dataset import GridDataset

features = ["bsa", "res_depth", "hse", "info_content", "pssm", "distance"]
target = "binary"

# Creating GraphDataset objects
dataset_train = GridDataset(
    hdf5_path = hdf5_paths,
    subset = train_ids, 
    features = features,
    target = target
)
dataset_val = GridDataset(
    hdf5_path = hdf5_paths,
    subset = valid_ids, 
    features = features,
    target = target

)
dataset_test = GridDataset(
    hdf5_path = hdf5_paths,
    subset = test_ids, 
    features = features,
    target = target
)
```

### Training

Let's define a `Trainer` instance, using for example of the already existing `GINet`. Because `GINet` is a GNN, it requires a dataset instance of type `GraphDataset`.

```python
from deeprankcore.trainer import Trainer
from deeprankcore.neuralnets.gnn.naive_gnn import NaiveNetwork

trainer = Trainer(
    NaiveNetwork,
    dataset_train,
    dataset_val,
    dataset_test
)

```

The same can be done using a CNN, for example `CnnClassification`. Here a dataset instance of type `GridDataset` is required.

```python
from deeprankcore.trainer import Trainer
from deeprankcore.neuralnets.cnn.model3d import CnnClassification

trainer = Trainer(
    CnnClassification,
    dataset_train,
    dataset_val,
    dataset_test
)
```

By default, the `Trainer` class creates the folder `./output` for storing predictions information collected later on during training and testing. `HDF5OutputExporter` is the exporter used by default, but the user can specify any other implemented exporter or implement a custom one.

Optimizer (`torch.optim.Adam` by default) and loss function can be defined by using dedicated functions:

```python
import torch

trainer.configure_optimizers(torch.optim.Adamax, lr = 0.001, weight_decay = 1e-04)

```

Then the `Trainer` can be trained and tested; the best model in terms of validation loss is saved by default, and the user can modify so or indicate where to save it using the `train()` method parameter `filename`.

```python
trainer.train(
    nepoch = 50,
    batch_size = 64,
    validate = True,
    filename = "<my_folder/model.pth.tar>")
trainer.test()

```

## h5x support

After installing  `h5xplorer`  (https://github.com/DeepRank/h5xplorer), you can execute the python file `deeprankcore/h5x/h5x.py` to explorer the connection graph used by deeprankcore. The context menu (right click on the name of the structure) allows to automatically plot the graphs using `plotly`.

## Package development

- Branching
  - When creating a new branch, please use the following convention: `<issue_number>_<description>_<author_name>`.
- Pull Requests
  - When creating a pull request, please use the following convention: `<type>: <description>`. Example _types_ are `fix:`, `feat:`, `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, and others based on the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines).
- Software release
  - Before creating a new package release, make sure to have updated all version strings in the source code. An easy way to do it is to run `bump2version [part]` from command line after having installed [bump2version](https://pypi.org/project/bump2version/) on your local environment. Instead of `[part]`, type the part of the version to increase, e.g. minor. The settings in `.bumpversion.cfg` will take care of updating all the files containing version strings. 
