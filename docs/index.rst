.. DeepRank-GNN documentation master file, created by
   sphinx-quickstart on Wed May 12 11:56:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DeepRank-Core |version| documentation
========================================

DeepRank-Core is an open-source deep learning (DL) framework for data mining of protein-protein interfaces (PPIs) or single-residue missense variants. This package is an improved and unified version of two previously developed packages: [deeprank](https://github.com/DeepRank/deeprank) and [Deeprank-GNN](https://github.com/DeepRank/Deeprank-GNN).

DeepRank-Core allows for transformation of (pdb formatted) molecular data into 3D representations (either grids or graphs) containing structural and physico-chemical information, which can be used for training neural networks. DeepRank-Core also offers a pre-implemented training pipeline, using either [CNNs](https://en.wikipedia.org/wiki/Convolutional_neural_network) (for grids) or [GNNs](https://en.wikipedia.org/wiki/Graph_neural_network) (for graphs), as well as output exporters for evaluating performances. 

Main features:

* Predefined atom-level and residue-level feature types (e.g. atom/residue type, charge, size, potential energy, all features' documentation is available [here](https://deeprankcore.readthedocs.io/en/latest/features.html))
* Predefined target types (binary class, CAPRI categories, DockQ, RMSD, and FNAT)
* Flexible definition of both new features and targets
* Features generation for both graphs and grids
* Efficient data storage in HDF5 format
* Support both classification and regression (based on `PyTorch`_ and `PyTorch Geometric`_)

.. _PyTorch: https://pytorch.org/docs/stable/index.html
.. _PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/latest/

Getting started
===========

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:
   
   installation
   getstarted

:doc:`installation`
    Get DeepRank-Core installed on your computer.

:doc:`getstarted`
    Understand how to use DeepRank-Core and how it can help you.

Notes
===========

.. toctree::
   :caption: Notes
   :hidden:

   features

:doc:`features`
    Get a detailed overview about nodes' and edges' features implemented in the package.

Package reference
===========
   
.. toctree::
   :caption: API
   :hidden:

   reference/deeprankcore

:doc:`reference/deeprankcore`
    This section documents the DeepRank-Core API.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
