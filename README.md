# Deeprank2

| Badges | |
|:----:|----|
| **fairness** |  [![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu) [![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/6403/badge)](https://bestpractices.coreinfrastructure.org/projects/6403) |
| **package** |  [![PyPI version](https://badge.fury.io/py/deeprank2.svg)](https://badge.fury.io/py/deeprank2) [![Codacy Badge](https://api.codacy.com/project/badge/Grade/f3f98b2d1883493ead50e3acaa23f2cc)](https://app.codacy.com/gh/DeepRank/deeprank2?utm_source=github.com&utm_medium=referral&utm_content=DeepRank/deeprank2&utm_campaign=Badge_Grade) |
| **docs** | [![Documentation Status](https://readthedocs.org/projects/deeprank2/badge/?version=latest)](https://deeprank2.readthedocs.io/en/latest/?badge=latest) [![DOI](https://zenodo.org/badge/450496579.svg)](https://zenodo.org/badge/latestdoi/450496579) |
| **tests** | [![Build Status](https://github.com/DeepRank/deeprank2/actions/workflows/build.yml/badge.svg)](https://github.com/DeepRank/deeprank2/actions) ![Linting status](https://github.com/DeepRank/deeprank2/actions/workflows/linting.yml/badge.svg?branch=main) [![Coverage Status](https://coveralls.io/repos/github/DeepRank/deeprank2/badge.svg?branch=main)](https://coveralls.io/github/DeepRank/deeprank2?branch=main) ![Python](https://img.shields.io/badge/python-3.10-blue.svg)  ![Python](https://img.shields.io/badge/python-3.11-blue.svg) |
| **running on** | ![Ubuntu](https://img.shields.io/badge/Ubuntu-E95420?style=for-the-badge&logo=ubuntu&logoColor=white) |
| **license** |  [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/license/apache-2-0/)  |

## Overview

![alt-text](./deeprank2.png)

DeepRank2 is an open-source deep learning (DL) framework for data mining of protein-protein interfaces (PPIs) or single-residue variants (SRVs). This package is an improved and unified version of three previously developed packages: [DeepRank](https://github.com/DeepRank/deeprank), [DeepRank-GNN](https://github.com/DeepRank/Deeprank-GNN), and [DeepRank-Mut](https://github.com/DeepRank/DeepRank-Mut).

DeepRank2 allows for transformation of (pdb formatted) molecular data into 3D representations (either grids or graphs) containing structural and physico-chemical information, which can be used for training neural networks. DeepRank2 also offers a pre-implemented training pipeline, using either [CNNs](https://en.wikipedia.org/wiki/Convolutional_neural_network) (for grids) or [GNNs](https://en.wikipedia.org/wiki/Graph_neural_network) (for graphs), as well as output exporters for evaluating performances.

Main features:
- Predefined atom-level and residue-level feature types
  - e.g. atom/residue type, charge, size, potential energy
  - All features' documentation is available [here](https://deeprank2.readthedocs.io/en/latest/features.html)
- Predefined target types
  - binary class, CAPRI categories, DockQ, RMSD, and FNAT
  - Detailed docking scores documentation is available [here](https://deeprank2.readthedocs.io/en/latest/docking.html)
- Flexible definition of both new features and targets
- Features generation for both graphs and grids
- Efficient data storage in HDF5 format
- Support for both classification and regression (based on [PyTorch](https://pytorch.org/) and [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/))

DeepRank2 extensive documentation can be found [here](https://deeprank2.rtfd.io/).

## Table of contents

- [Deeprank2](#deeprank2)
  - [Overview](#overview)
  - [Table of contents](#table-of-contents)
  - [Installation](#installation)
    - [Dependencies](#dependencies)
    - [Deeprank2 Package](#deeprank2-package)
    - [Test installation](#test-installation)
    - [Contributing](#contributing)
    - [Data generation](#data-generation)
    - [Datasets](#datasets)
      - [GraphDataset](#graphdataset)
      - [GridDataset](#griddataset)
    - [Training](#training)
  - [Computational performances](#computational-performances)
  - [Package development](#package-development)

## Installation

The package officially supports ubuntu-latest OS only, whose functioning is widely tested through the continuous integration workflows. 

### Dependencies

Before installing deeprank2 you need to install some dependencies. We advise to use a [conda environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) with Python >= 3.10 installed. The following dependency installation instructions are updated as of 14/09/2023, but in case of issues during installation always refer to the official documentation which is linked below:

*  [MSMS](https://anaconda.org/bioconda/msms): `conda install -c bioconda msms`.
    * [Here](https://ssbio.readthedocs.io/en/latest/instructions/msms.html) for MacOS with M1 chip users.
*  [PyTorch](https://pytorch.org/get-started/locally/)
    * We support torch's CPU library as well as CUDA.
*  [PyG](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and its optional dependencies: `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv`.
*  [DSSP 4](https://swift.cmbi.umcn.nl/gv/dssp/)
    * Check if `dssp` is installed: `dssp --version`. If this gives an error or shows a version lower than 4:
      * on ubuntu 22.04 or newer: `sudo apt-get install dssp`. If the package cannot be located, first run `sudo apt-get update`.
      * on older versions of ubuntu or on mac or lacking sudo priviliges: install from [here](https://github.com/pdb-redo/dssp), following the instructions listed. Alternatively, follow [this](https://github.com/PDB-REDO/libcifpp/issues/49) thread. 
*  [GCC](https://gcc.gnu.org/install/)
    * Check if gcc is installed: `gcc --version`. If this gives an error, run `sudo apt-get install gcc`.  
*  For MacOS with M1 chip users only install [the conda version of PyTables](https://www.pytables.org/usersguide/installation.html).

### Deeprank2 Package

Once the dependencies are installed, you can install the latest stable release of deeprank2 using the PyPi package manager:

```bash
pip install deeprank2
```

Alternatively, get all the new developments by cloning the repo and installing the editable version of the package with:

```bash
git clone https://github.com/DeepRank/deeprank2
cd deeprank2
pip install -e .'[test]'
```

The `test` extra is optional, and can be used to install test-related dependencies useful during the development.

### Test installation

If you have installed the package from a cloned repository (second option above), you can check that all components were installed correctly, using pytest.
The quick test should be sufficient to ensure that the software works, while the full test (a few minutes) will cover a much broader range of settings to ensure everything is correct.

Run `pytest tests/test_integration.py` for the quick test or just `pytest` for the full test (expect a few minutes to run).

### Contributing

If you would like to contribute to the package in any way, please see [our guidelines](CONTRIBUTING.rst).

The following section serves as a first guide to start using the package, using protein-protein Interface (PPI) queries
as example. For an enhanced learning experience, we provide in-depth [tutorial notebooks](https://github.com/DeepRank/deeprank2/tree/main/tutorials) for generating PPI data, generating SVR data, and for the training pipeline.
For more details, see the [extended documentation](https://deeprank2.rtfd.io/).

### Data generation

For each protein-protein complex (or protein structure containing a SRV), a query can be created and added to the `QueryCollection` object, to be processed later on. Different types of queries exist:
- In a `ProteinProteinInterfaceResidueQuery` and `SingleResidueVariantResidueQuery`, each node represents one amino acid residue.
- In a `ProteinProteinInterfaceAtomicQuery` and `SingleResidueVariantAtomicQuery`, each node represents one atom within the amino acid residues.

A query takes as inputs:
- a `.pdb` file, representing the protein-protein structure
- the ids of the chains composing the structure, and
- optionally, the correspondent position-specific scoring matrices (PSSMs), in the form of `.pssm` files.

```python
from deeprank2.query import QueryCollection, ProteinProteinInterfaceResidueQuery

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

The queries can then be processed into graphs only or both graphs and 3D grids, depending on which kind of network will be used later for training.

```python
from deeprank2.features import components, conservation, contact, exposure, irc, surfacearea
from deeprank2.utils.grid import GridSettings, MapMethod

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
from deeprank2.dataset import GraphDataset

node_features = ["bsa", "res_depth", "hse", "info_content", "pssm"]
edge_features = ["distance"]
target = "binary"
train_ids = [<ids>]
valid_ids = [<ids>]
test_ids = [<ids>]

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
    train = False,
    dataset_train = dataset_train
)
dataset_test = GraphDataset(
    hdf5_path = hdf5_paths,
    subset = test_ids,
    train = False,
    dataset_train = dataset_train
)
```

#### GridDataset

For training CNNs the user can create a `GridDataset` instance:

```python
from deeprank2.dataset import GridDataset

features = ["bsa", "res_depth", "hse", "info_content", "pssm", "distance"]
target = "binary"
train_ids = [<ids>]
valid_ids = [<ids>]
test_ids = [<ids>]

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
    train = False,
    dataset_train = dataset_train,
)
dataset_test = GridDataset(
    hdf5_path = hdf5_paths,
    subset = test_ids,
    train = False,
    dataset_train = dataset_train,
)
```

### Training

Let's define a `Trainer` instance, using for example of the already existing `GINet`. Because `GINet` is a GNN, it requires a dataset instance of type `GraphDataset`.

```python
from deeprank2.trainer import Trainer
from deeprank2.neuralnets.gnn.naive_gnn import NaiveNetwork

trainer = Trainer(
    NaiveNetwork,
    dataset_train,
    dataset_val,
    dataset_test
)

```

The same can be done using a CNN, for example `CnnClassification`. Here a dataset instance of type `GridDataset` is required.

```python
from deeprank2.trainer import Trainer
from deeprank2.neuralnets.cnn.model3d import CnnClassification

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

## Computational performances

We measured the efficiency of data generation in DeepRank2 using the tutorials' [PDB files](https://zenodo.org/record/8187806) (~100 data points per data set), averaging the results run on Apple M1 Pro, using a single CPU.
Parameter settings were: atomic resolution, `distance_cutoff` of 5.5 Å, radius (for SRV only) of 10 Å. The [features modules](https://deeprank2.readthedocs.io/en/latest/features.html) used were `components`, `contact`, `exposure`, `irc`, `secondary_structure`, `surfacearea`, for a total of 33 features for PPIs and 26 for SRVs (the latter do not use `irc` features).

|      |         Data processing speed <br />[seconds/structure]        |                Memory <br />[megabyte/structure]               |
|------|:--------------------------------------------------------:|:--------------------------------------------------------:|
| PPIs | graph only: **2.99** (std 0.23) <br />graph+grid: **11.35** (std 1.30) | graph only: **0.54** (std 0.07) <br />graph+grid: **16.09** (std 0.44) |
| SRVs | graph only: **2.20** (std 0.08)  <br />graph+grid: **2.85** (std 0.10) | graph only: **0.05** (std 0.01) <br />graph+grid: **17.52** (std 0.59) |

## Package development

- Branching
  - When creating a new branch, please use the following convention: `<issue_number>_<description>_<author_name>`.
- Pull Requests
  - When creating a pull request, please use the following convention: `<type>: <description>`. Example _types_ are `fix:`, `feat:`, `build:`, `chore:`, `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, and others based on the [Angular convention](https://github.com/angular/angular/blob/22b96b9/CONTRIBUTING.md#-commit-message-guidelines).
- Software release
  - Before creating a new package release, make sure to have updated all version strings in the source code. An easy way to do it is to run `bump2version [part]` from command line after having installed [bump2version](https://pypi.org/project/bump2version/) on your local environment. Instead of `[part]`, type the part of the version to increase, e.g. minor. The settings in `.bumpversion.cfg` will take care of updating all the files containing version strings.
