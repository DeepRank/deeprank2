DeepRank2 |version| documentation
========================================

DeepRank2 is a deep learning framework for data mining Protein-Protein Interactions (PPIs) using Graph Neural Networks. 

DeepRank2 contains useful APIs for pre-processing PPIs data, computing features and targets, as well as training and testing GNN models.

Main features:

* Predefined atom-level and residue-level PPI feature types (e.g. atomic density, vdw energy, residue contacts, PSSM, etc.)
* Predefined target type (e.g. binary class, CAPRI categories, DockQ, RMSD, FNAT, etc.)
* Flexible definition of both new features and targets
* Graphs feature mapping
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
    Get DeepRank2 installed on your computer.

:doc:`getstarted`
    Understand how to use DeepRank2 and how it can help you.

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
    This section documents the DeepRank2 API.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
