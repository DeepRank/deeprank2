DeepRank2 |version| documentation
========================================

DeepRank2 is an open-source deep learning (DL) framework for data mining of protein-protein interfaces (PPIs) or single-residue variants (SRVs).
This package is an improved and unified version of three previously developed packages: `DeepRank`_, `DeepRank-GNN`_, and `DeepRank-Mut`_.

DeepRank2 allows for transformation of (pdb formatted) molecular data into 3D representations (either grids or graphs) containing structural and physico-chemical information, which can be used for training neural networks. DeepRank2 also offers a pre-implemented training pipeline, using either `convolutional neural networks`_ (for grids) or `graph neural networks`_ (for graphs), as well as output exporters for evaluating performances. 

Main features:

* Predefined atom-level and residue-level feature types (e.g. atom/residue type, charge, size, potential energy, all features' documentation is available under `Features`_ notes)
* Predefined target types (binary class, CAPRI categories, DockQ, RMSD, and FNAT, detailed docking scores documentation is available under `Docking scores`_ notes)
* Flexible definition of both new features and targets
* Features generation for both graphs and grids
* Efficient data storage in HDF5 format
* Support both classification and regression (based on `PyTorch`_ and `PyTorch Geometric`_)

.. _DeepRank: https://github.com/DeepRank/deeprank
.. _DeepRank-GNN: https://github.com/DeepRank/Deeprank-GNN
.. _DeepRank-Mut: https://github.com/DeepRank/DeepRank-Mut
.. _convolutional neural networks: https://en.wikipedia.org/wiki/Convolutional_neural_network
.. _graph neural networks: https://en.wikipedia.org/wiki/Graph_neural_network
.. _Features: https://deeprank2.readthedocs.io/en/latest/features.html
.. _Docking scores: https://deeprank2.readthedocs.io/en/latest/docking.html
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
    Get DeepRank2 installed on your computer.

:doc:`getstarted`
    Understand how to use DeepRank2 and how it can help you.

Notes
===========

.. toctree::
   :caption: Notes
   :hidden:

   features
   docking

:doc:`features`
    Get a detailed overview about nodes' and edges' features implemented in the package.

:doc:`docking`
    Get a detailed overview about PPIs' docking metrics implemented in the package.

Package reference
===========

.. toctree::
   :caption: API
   :hidden:

   reference/deeprank2

:doc:`reference/deeprank2`
    This section documents the DeepRank2 API.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
