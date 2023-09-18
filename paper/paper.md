---
title: 'DeepRank2: Mining 3D Protein Structures with Geometric Deep Learning'
tags:
  - Python
  - PyTorch
  - structural biology
  - geometric deep learning
  - 3D protein structures
  - protein-protein interfaces
  - single-residue variants
authors:
  - name: Giulia Crocioni
    orcid: 0000-0002-0823-0121
    corresponding: true
    affiliation: 1
    equal-contrib: true
  - name: Dani L. Bodor
    orcid: 0000-0003-2109-2349
    affiliation: 1
    equal-contrib: true
  - name: Coos Baakman
    orcid: 0000-0003-4317-1566
    affiliation: 2
    equal-contrib: true
  - name: Daniel Rademaker
    orcid: 0000-0003-1959-1317
    affiliation: 2
  - name: Dario Marzella
    orcid: 0000-0002-0043-3055
    affiliation: 2
  - name: Gayatri Ramakrishnan
    orcid: 0000-0001-8203-2783
    affiliation: 2
  - name: Sven van der Burg
    orcid: 0000-0003-1250-6968
    affiliation: 1
  - name: Farzaneh Meimandi Parizi
    orcid: 0000-0003-4230-7492
    affiliation: 2
  - name: Li C. Xue
    orcid: 0000-0002-2613-538X
    affiliation: 2
affiliations:
 - name: Netherlands eScience Center, Amsterdam, The Netherlands
   index: 1
 - name: Radboud University Medical Center, Nijmegen, The Netherlands
   index: 2
date: 08 August 2023
bibliography: paper.bib

---

# Summary
[comment]: <> (CHECK FOR AUTHORS: Do the summary describe the high-level functionality and purpose of the software for a diverse, non-specialist audience?)

We present DeepRank2, a deep learning (DL) framework geared towards making predictions on 3D protein structures for variety of biologically relevant applications. Our software can be used for predicting structural properties in drug design, immunotherapy, or designing novel proteins, among other fields. DeepRank2 allows for transformation and storage of 3D representations of both protein-protein interfaces (PPIs) and protein single-residue variants (SRVs) into either graphs or volumetric grids containing structural and physico-chemical information. These can be used for training neural networks for a variety of patterns of interest, using either our pre-implemented training pipeline for graph neural networks (GNNs) or convolutional neural networks (CNNs) or external pipelines. The entire framework flowchart is visualized in \autoref{fig:flowchart}. The package is fully open-source, follows the community-endorsed FAIR principles for research software, provides user-friendly APIs, publicily available [documentation](https://deeprank2.readthedocs.io/en/latest/), and in-depth [tutorials](https://github.com/DeepRank/deeprank2/blob/main/tutorials/TUTORIAL.md).

[comment]: <> (CHECK FOR AUTHORS: Do the authors clearly state what problems the software is designed to solve and who the target audience is?)
[comment]: <> (CHECK FOR AUTHORS: Do the authors describe how this software compares to other commonly-used packages?)

![DeepRank2 framework overview. 3D coordinates of protein structures are extracted from PDB files and converted into graphs and grids, using either an atomic or a residue level, depending on the user’s requirements. The data are enriched with geometrical and physicochemical information and are stored into HDF5 files, and can then be used in the pre-implemented DL pipeline for training PyTorch networks and computing predictions.\label{fig:flowchart}](deeprank2.png){ width=100% }

# State of the field

[comment]: <> (Motivation for using 3D protein structures)
The 3D structure of proteins and protein complexes provides fundamental information to understand biological processes at the molecular scale. Exploiting or engineering these molecules is key for many biomedical applications such as drug design [@GANE2000401], immunotherapy [@sadelain_basic_2013], or designing novel proteins [@nonnaturalppi]. For example, PPI data can be harnessed to address critical challenges in the computational prediction of peptides presented on the major histocompatibility complex (MHC) protein, which play a key role in T-cell immunity. Protein structures can also be exploited in molecular diagnostics for the identification of SRVs, that can be pathogenic sequence alterations in patients with inherited diseases [@mut_cnn; @shroff].

[comment]: <> (What makes using 3D protein structures with DL possible)
In the past decades, a variety of experimental methods (e.g., X-ray crystallography, nuclear magnetic resonance, cryogenic electron microscopy) have determined and accumulated a large number of atomic-resolution 3D structures of proteins and protein-protein complexes [@schwede_protein_2013]. Because experimental determination of structures is a tedious and expensive process, several computational prediction methods have been developed over the past decades, exploiting classical molecular modelling [@rosetta; @modeller; @haddock], and, more recently, DL [@alphafold_2021; @alphafold_multi]. The large amount of data available makes it possible to use DL to leverage 3D structures and learn their complex patterns. Unlike other machine learning (ML) techniques, deep neural networks hold the promise of learning from millions of data points without reaching a performance plateau quickly, which is made computationally feasible by hardware accelerators (i.e., GPUs, TPUs) and parallel file system technologies.

[comment]: <> (Examples of DL with PPIs and SRVs)
3D CNNs have been trained on 3D grids for the classification of biological vs. crystallographic PPIs [@renaud_deeprank_2021], and for the scoring of models of protein-protein complexes generated by computational docking [@renaud_deeprank_2021; @dove]. Gaiza et al. have applied geodesic CNNs to extract protein interaction fingerprints by applying 2D CNNs on spread-out protein surface patches [@masif]. 3D CNNs have been used for exploiting protein structure data for predicting mutation-induced changes in protein stability [@mut_cnn] and identifying novel gain-of-function mutations [@shroff]. Contrary to CNNs, in GNNs the convolution operations on graphs can rely on the relative local connectivity between nodes and not on the data orientation, making graphs rotational invariant. Additionally, GNNs can accept any size of graph, while in a CNN the size of the 3D grid for all input data needs to be the same, which may be problematic for datasets containing highly variable in size structures. Based on these arguments, different GNN-based tools have been designed to predict patterns from PPIs [@dove_gnn; @fout_protein_nodate; @reau_deeprank-gnn_2022]. Eisman et al. developed a rotation-equivariant neural network trained on point-based representation of the protein atomic structure to classify PPIs [@rot_eq_gnn].

# Statement of need

[comment]: <> (Motivation for a flexible framework)
Data mining 3D structures of proteins presents several challenges. These include complex physico-chemical rules governing structural features, the possibility of characterizartion at different scales (e.g., atom-level, residue level, and secondary structure level), and the large diversity in shape and size. Furthermore, because a structures can easily comprise of hundreds to thousands of residues (and ~15 times as many atoms), efficient processing and featurization of many structures is critical to handle the computational cost and file storage requirements. Existing software solutions are often highly specialized and not developed as reusable and flexible frameworks, and cannot be easily adapted to diverse applications and predictive tasks. Examples include DeepAtom [@deepatom] for protein-ligand binding affinity prediction only, and MaSIF [@masif] for deciphering patterns in protein surfaces. While some frameworks, such as TorchProtein and TorchDrug [@torchdrug], configure themselves as general-purpose ML libraries for both molecular sequences and 3D structures, they only implement geometric-related features and do not incorporate fundamental physico-chemical information in the 3D representation of molecules.

These limitations create a growing demand for a generic and flexible DL framework that researchers can readily utilize for their specific research questions while cutting down the tedious data preprocessing stages. Generic DL frameworks have already emerged in diverse scientific fields, such as computational chemistry (e.g., DeepChem [@deepchem]) and condensed matter physics (e.g., NetKet [@netket]), which have promoted collaborative efforts, facilitated novel insights, and benefited from continuous improvements and maintenance by engaged user communities.

# Key features

DeepRank2 allows to transform and store 3D representations of both PPIs and SRVs into 3D grids or graphs containing both geometric and physico-chemical information, and provides a DL pipeline which can be used for training pre-implemented neural networks for a given pattern of interest to the user.

As input, DeepRank2 takes [PDB-formatted](https://www.cgl.ucsf.edu/chimera/docs/UsersGuide/tutorials/pdbintro.html) atomic structures, which is one of the standard and most widely used formats in the field of structural biology. These are mapped to graphs, where nodes can represent either residues or atoms, as chosen by the user, and edges represent the interactions between them. The user can configure two types of 3D structures as input for the featurization phase:
-  PPIs, for mining interaction patterns within protein-protein complexes;
-  SRVs, for mining mutation phenotypes within protein structures.

Graphs can either be used directly or mapped to volumetric grids (i.e., 3D image-like representations). Then the physico-chemical and geometrical features for the grids and/or graphs are computed and assigned to each node and edge. The user can choose which features to generate from several pre-existing options defined in the package, or define custom features modules, as explained in the documentation. Examples of pre-defined node features are the type of the amino acid, its size and polarity, as well as more complex features such as its buried surface area and secondary structure features. Examples of pre-defined edge features are distance, covalency, and potential energy. A detailed list of predefined features can be found in the [documentation's features page](https://deeprank2.readthedocs.io/en/latest/features.html). Multiple CPUs can be used to parallelize and speed up the featurization process. The processed data are saved into HDF5 files, designed to efficiently store and organize big data. Users can then use the data for any ML or DL framework suited for the application. Specifically, graphs can be used for the training of GNNs, and 3D grids can be used for the training of CNNs.

DeepRank2 also provides convenient pre-implemented modules for training simple [PyTorch](https://pytorch.org/)-based GNNs and CNNs using the data generated in the previous step. Alternatively, users can implement custom PyTorch networks in the DeepRank package (or export the data to external software). Data can be loaded across multiple CPUs, and the training can be run on GPUs. The data stored within the HDF5 files are read into customized datasets, and the user-friendly API allows for selection of individual features (from those generated above), definition of the targets, and the predictive task (classfication or regression), among other settings. Then the datasets can be used for training, validating, and testing the chosen neural network. The final model and results can be saved using built-in data exporter modules.

DeepRank2 embraces the best practices of open-source development by utilizing platforms like GitHub and Git, unit testing (as of August 2023 coverage is 83%), continuous integration, automatic documentation, and Findable, Accessible, Interoperable, and Reusable (FAIR) principles. Detailed [documentation](https://deeprank2.readthedocs.io/en/latest/?badge=latest) and [tutorials](https://github.com/DeepRank/deeprank2/blob/main/tutorials/TUTORIAL.md) for getting started with the package are publicly available. The project aims to create high-quality software that can be easily accessed, used, and contributed to by a wide range of researchers.

This project is expected to have an impact across the all of structural bioinformatics, enabling advancements that rely on molecular complex analysis, such as structural biology, protein engineering, and rational drug design. The target community includes researchers working with molecular complexes data, such as computational biologists, immunologists, and structural bioinformatics scientists. The existing features, as well as the sustainable package formatting and its modular design make DeepRank2 an excellent framework to build upon. Taken together, DeepRank2 provides all the requirements to become the all-purpose DL tool that is currently lacking in the field of biomolecular interactions.

# Acknowledgements

This work was supported by the [Netherlands eScience Center](https://www.esciencecenter.nl/) under grant number NLESC.OEC.2021.008, and [SURF](https://www.surf.nl/en) infrastructure, and was developed in collaboration with the [Department of Medical BioSciences](https://www.radboudumc.nl/en/research/departments/medical-biosciences) at RadboudUMC.

# References
