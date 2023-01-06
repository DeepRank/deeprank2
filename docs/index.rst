.. DeepRank-GNN documentation master file, created by
   sphinx-quickstart on Wed May 12 11:56:41 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Deeprank-Core |version| documentation
========================================

Deeprank-Core is a deep learning framework for data mining Protein-Protein Interactions (PPIs) using Graph Neural Networks. 

Deeprank-Core contains useful APIs for pre-processing PPIs data, computing features and targets, as well as training and testing GNN models.

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
* :ref:`search`
