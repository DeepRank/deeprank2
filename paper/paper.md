---
title: 'DeepRank2: Mining 3D Protein Structures with Geometric Deep Learning'
tags:
  - Python
  - PyTorch
  - structural biology
  - geometric deep learning
  - 3D protein structures
  - protein protein interfaces
  - missense variants
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

We present DeepRank2, a deep learning (DL) framework geared towards making predictions on 3D protein structures for variety of biologically relevant applications. Our software can be used for predicting structural properties in drug design, immunotherapy, or designing novel proteins, among other fields. DeepRank2 allows for transformation and storage of 3D representations of both protein-protein interfaces (PPIs) and protein single residue variants (SRVs) into either graphs or volumetric grids containing structural and physico-chemical information. These can be used for training neural networks for a variety of patterns of interest, using either our pre-implemented training pipeline for graph neural networks (GNNs) and convolutional neural networks (CNNs) or external pipelines. The package is fully open source, follows the community-endorsed FAIR principles for research software, provides user-friendly APIs, publicily available [documentation](https://deeprank2.readthedocs.io/en/latest/), and in depth [tutorials](https://github.com/DeepRank/deeprank2/blob/main/tutorials/TUTORIAL.md).

[comment]: <> (saving below sentences in case they are useful down the line)
[comment]: <> (The entire framework flowchart is visualized in \autoref{fig:flowchart}. DeepRank2 software aims at unifying previously developed DL frameworks for data mining PPIs (DeepRank [@renaud_deeprank_2021], DeepRank-GNN [@reau_deeprank-gnn_2022]), and proteins' variants (DeepRank-Mut [@]). )
[comment]: <> (Additionally, the software allows for much greater flexibility, allowing users to easily tailor the framework to specific patterns of interest and features, and select the pipeline's steps that best suits their requirements.)


[comment]: <> (CHECK FOR AUTHORS: Do the authors clearly state what problems the software is designed to solve and who the target audience is?)
[comment]: <> (CHECK FOR AUTHORS: Do the authors describe how this software compares to other commonly-used packages?)

# State of the field

[comment]: <> (Motivation for using 3D protein structures)
Individual proteins' and protein complexes' 3D structures provide fundamental information to decipher biological processes at the molecular scale. Gaining knowledge on how those biomolecules interact in 3D space is key for understanding their functions and exploiting or engineering these molecules for a wide variety of purposes such as drug design [@GANE2000401], immunotherapy [@sadelain_basic_2013], or designing novel proteins [@nonnaturalppi]. For example, PPI data can be harnessed to address critical challenges in the computational prediction of peptides presented on the major histocompatibility complex (MHC) protein, which play a key role in T-cell immunity. Protein structures can also be exploited in molecular diagnostics for the identification of missense variants, that are pathogenic sequence alterations in patients with inherited diseases.

[comment]: <> (What makes using 3D protein structures with DL possible)
In the past decades, a variety of experimental methods (e.g., X-ray crystallography, nuclear magnetic resonance, cryogenic electron microscopy) have determined and accumulated a large number of atomic-resolution 3D structures of proteins and protein-protein complexes. Since the experimental structure determination is a tedious and expensive process, several computational methods have also been developed over the past few years, such as Alphafold for protein structures, and PANDORA [@pandora], HADDOCK [@haddock], and Alphafold-Multimer [@alphafold_multi] for protein complexes. The large amount of data available makes it possible to use DL to leverage 3D structures and learn their complex patterns. Unlike other machine learning (ML) techniques, deep neural networks hold the promise of learning from millions of data without reaching a performance plateau quickly, which is made computationally feasible by hardware accelerators (i.e., GPUs, TPUs) and parallel file system technologies.

[comment]: <> (Examples of DL with PPIs and variants)
3D CNNs have been trained on 3D grids for the classification of biological vs. crystallographic PPIs [@renaud_deeprank_2021], and for the scoring of models of protein-protein complexes generated by computational docking [@renaud_deeprank_2021,@dove]. Gaiza et al. have applied geodesic CNNs to extract protein interaction fingerprints by applying 2D CNNs on spread-out protein surface patches [@masif]. 3D CNNs have been used for exploiting protein structure data for predicting mutation-induced changes in protein stability [@mut_cnn] and identifying novel gain-of-function mutations [@shroff]. Contrary to CNNs, in GNNs the convolution operations on graphs can rely on the relative local connectivity between nodes, making graphs rotational invariant, and such networks can accept any size of graph, making them more representative of the PPIs diversity. Based on these arguments, different GNN-based tools have been designed to predict patterns from PPIs [@dove_gnn,@fout_protein_nodate,@reau_deeprank-gnn_2022]. Eisman et al. developed a rotation-equivariant neural network trained on point-based representation of the protein atomic structure to classify PPIs [@rot_eq_gnn].

# Statement of need

[comment]: <> (Motivation for a flexible framework)
Data mining 3D proteins structures still presents several unique challenges because of the different physico-chemical rules that govern them, the possibility of characterizartion at different levels (atom-atom level, residue-residue level, and secondary structure level), and the highly diversity in terms of shapes and sizes. The efficient processing and featurization of a large number of atomic coordinates files of proteins is also critical in terms of computational cost and file storage requirements. Existing software solutions are often highly specialized, developed for analysis and visualization of specific research projects’ results, and cannot be easily adapted to diverse applications and predictive tasks, not being developed as reusable and flexible frameworks. Examples include DeepAtom [@deepatom], for protein-ligand binding affinity prediction only, MaSIF [@masif], for deciphering patterns in protein surfaces, and MHCFlurry 2.0 [@2020mhcflurry], for predicting binding affinity for a specific type of protein-protein complex, the peptide-major histocompatibility complex (MHC). Other frameworks instead, such as TorchProtein and TorchDrug [@torchdrug], configure themselves as general-purpose ML libraries for both molecular sequences and 3D structures. However, they only implement geometric-related features and do not incorporate fundamental physico-chemical information in the molecules’ 3D representations, which may be crucial for accurate predictions. 

These limitations create a growing demand for generic and flexible DL frameworks that researchers can readily utilize for their specific research questions while cutting down the tedious data preprocessing stages. Generic DL frameworks have already emerged in diverse scientific fields, such as computational chemistry (e.g., DeepChem [@deepchem]) and condensed matter physics (e.g., NetKet [@netket]), which have promoted collaborative efforts, facilitated novel insights, and benefited from continuous improvements and maintenance by engaged user communities.

# Key features

DeepRank2 allows to transform and store 3D representations of both PPIs and individual proteins' variants into 3D grids or graphs containing geometric and physico-chemical information, and provides a DL pipeline which can be used for training pre-implemented neural networks for whatever specific pattern of interest for the user.

The 3D protein structures provided in the form of PDB files are mapped to graphs in which nodes represent residues or atoms, according to the resolution chosen by the user, and edges the interactions between them. The user can configure two types of 3D structures as input for the featurization phase:
- PPIs, for mining interaction patterns within protein-protein complexes; 
- missense variants, for mining pathologic mutations within protein structures. 

The graphs can also be mapped to volumetric grids (i.e., 3D image-like representations). Then the physico-chemical and geometrical features for the grids and/or graphs are computed. The user can choose which features to generate from several ones already defined in the package, and can also define custom features modules, as explained in the documentation. Examples of pre-defined node features are the type of the amino acid, its polarity, the solvent-accessible surface area. Examples of pre-defined edge features are distance, covalent bond, electrostatic potential. The full and detailed list of features can be found in the [documentation's features page](https://deeprank2.readthedocs.io/en/latest/features.html). Multiple CPUs can be used to parallelize and speed up the featurization process. The mapped data are finally saved into HDF5 files, designed to efficiently store and organize big data. Users can then use the data saved into HDF5 files for whatever architecture and DL framework is more suited for the application. In particular, graphs can be used for the training of GNNs, and 3D grids can be used for the training of CNNs.

DeepRank2 provides also handy modules for training Pytorch neural networks with the data stored into the HDF5 files. A few simple GNNs and CNNs are pre-implemented within the package, and can be trained on the generated data. The neural networks have been developed using [PyTorch](https://pytorch.org/), and the user is also free to implement custom networks using the same PyTorch framework. The data can be loaded across multiple CPUs, and the training can be run on GPUs. The data stored within the HDF5 files are read into customized datasets, and the user-friendly API allows the selection of specific features across the ones generated, the definition of the targets and the predictive tasks. Then the datasets can be used for training, validating, and testing the chosen neural network, and the results together with the trained model can be saved using the DeepRank2 exporters' module.

The package embraces the best practices of open-source development by utilizing platforms like GitHub and Git, unit testing (as of August 2023 coverage is 83%), continuous integration, automatic documentation, and Findable, Accessible, Interoperable, and Reusable (FAIR) principles. Detailed [documentation](https://deeprank2.readthedocs.io/en/latest/?badge=latest) and [tutorials](https://github.com/DeepRank/deeprank2/blob/main/tutorials/TUTORIAL.md) for getting started with the package are publicly available. The project aims to create high-quality software that can be easily accessed, used and contributed by a wide range of researchers.

The project is expected to have an impact across structural bioinformatics domains, enabling advancements in the disciplines that rely on molecular complex analysis, such as structural biology, protein engineering, and rational drug design. The target community includes researchers working with molecular complexes data, such as computational biologists, immunologists, and structural bioinformatics scientists. The existing features, as well as the sustainable package formatting and its great modularity make DeepRank2 an excellent framework to build upon, to generate the all-purpose deep learning tool that is currently lacking in the field of biomolecular interactions.

![DeepRank2 framework overview. 3D coordinates of protein structures are extracted from PDB files and converted into graphs and grids, using either an atomic or a residual level, depending on the user’s requirements. The data are enriched with geometrical and physicochemical information and are stored into HDF5 files, and can then be used in the pre-implemented DL pipeline for training PyTorch networks and computing predictions.\label{fig:flowchart}](deeprank2.png)

# Acknowledgements

This work was supported by the [Netherlands eScience Center](https://www.esciencecenter.nl/) under grant number NLESC.OEC.2021.008, and [SURF](https://www.surf.nl/en) infrastructure, and was developed in collaboration with the [Department of Medical BioSciences](https://www.radboudumc.nl/en/research/departments/medical-biosciences) at RadboudUMC.

[comment]: <> (TODO: Acknowledgement of any financial support (already asked Pablo, waiting for an answer).)

# References
