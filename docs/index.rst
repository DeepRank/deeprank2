.. DeepRank-GNN documentation master file, created by
   sphinx-quickstart on Wed May 12 11:56:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deeprank-Core |version| documentation
========================================

DeepRank-Core is an open-source deep learning (DL) framework for data mining 3D representations of both protein-protein interfaces (PPIs) and individual proteins' variants using graph neural networks (GNNs) or convolutional neural networks (CNNs). It is an improved and unified version of the previously developed [deeprank](https://github.com/DeepRank/deeprank) and [Deeprank-GNN](https://github.com/DeepRank/Deeprank-GNN).

DeepRank-Core allows to transform and store 3D representations of both PPIs and individual proteins' variants into grids or graphs containing structural and physico-chemical information, which can then be used for training neural networks for whatever specific pattern of interest for the user. DeepRank-Core also offers a pre-implemented training pipeline which can use either CNNs or GNNs, as well as handy output exporters for evaluating performances. 

Main features:

* Predefined atom-level and residue-level feature types (e.g. atomic density, vdw energy, residue contacts, PSSM, etc.)
* Predefined target types (e.g. binary class, CAPRI categories, DockQ, RMSD, FNAT, etc.)
* Flexible definition of both new features and targets
* Graphs and grids features mapping
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
    Get Deeprank-Core installed on your computer.

:doc:`getstarted`
    Understand how to use Deeprank-Core and how it can help you.

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
    This section documents the Deeprank-Core API.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
