---
title: 'Deeprank-Core: Mining 3D-Protein Structures with Geometric Deep Learning'
tags:
  - Python
  - PyTorch
  - structural biology
  - geometric deep learning
  - protein protein interfaces
  - missense variants
authors:
  - name: Giulia Crocioni
    orcid: 0000-0002-0823-0121
    corresponding: true
    equal-contrib: true
    affiliation: 1
  - name: Author Without ORCID
    equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 2
  - name: Author with no affiliation
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 3
affiliations:
 - name: Netherlands eScience Center, Amsterdam, The Netherlands
   index: 1
 - name: Institution Name, Country
   index: 2
 - name: Independent Researcher, Country
   index: 3
date: 17 August 2023
bibliography: paper.bib

---

[comment]: <> (TODO: update readme and getstarted with variants and with new title/summary after having well defined the entire paper)

# Summary
[comment]: <> (CHECK: Do the summary describe the high-level functionality and purpose of the software for a diverse, non-specialist audience?)

We present DeepRank-Core, an open-source Deep Learning (DL) framework that offers researchers unified and user-friendly APIs to accelerate development of software solutions allowing biologically relevant predictions to gain knowledge on protein 3D structures for a wide variety of purposes such as drug design, immunotherapy, or designing novel proteins. DeepRank-Core allows to transform and store 3D representations of both Protein-Protein Interfaces (PPIs) and individual proteins' variants into grids or graphs containing structural and physico-chemical information, which can then be used for training Neural Networks for whatever specific pattern of interest for the user. DeepRank-Core also offers a pre-implemented training pipeline which can use either Convolutional Neural Networks (CNNs) or Graph Neural Networks (GNNs), as well as handy output exporters for evaluating performances. The entire framework flowchart is visualized in \autoref{fig:flowchart}. DeepRank-Core software aims at unifying previously developed DL frameworks for data mining PPIs (DeepRank [@renaud_deeprank_2021], DeepRank-GNN [@reau_deeprank-gnn_2022]), and proteins' variants (DeepRank-Mut [@]), following the community-endorsed FAIR principles for Research Software, improving the APIs, and enriching the documentation, which is [publicily available](https://deeprankcore.readthedocs.io/en/latest/). Additionally, the software allows for much greater flexibility, allowing users to easily tailor the framework to specific patterns of interest and features, and select the pipeline's steps that best suits their requirements.

# Statement of need

[comment]: <> (CHECK: Do the authors clearly state what problems the software is designed to solve and who the target audience is?)
[comment]: <> (CHECK: Do the authors describe how this software compares to other commonly-used packages?)
[comment]: <> (TODO: Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it.)
[comment]: <> (TODO: Rephrase everything once)

[comment]: <> (Motivation for using 3D protein structures)
Individual proteins' and protein complexes' 3D structures provide fundamental information to decipher biological processes at the molecular scale. Gaining knowledge on how those biomolecules interact in 3D space is key for understanding their functions and exploiting or engineering these molecules for a wide variety of purposes such as drug design [@GANE2000401], immunotherapy [@sadelain_basic_2013], or designing novel proteins [@nonnaturalppi]. For example, protein structures can be exploited in molecular diagnostics for the identification of pathogenic sequence alterations in patients with inherited diseases. Regarding PPI instead, such data can be harnessed to address critical challenges in the computational prediction of peptides presented on the Major Histocompatibility Complex (MHC) protein, which play a key role in T-cell immunity.
[comment]: <> (What makes using 3D protein structures possible)
In the past decades, a variety of experimental methods (e.g., X-ray crystallography, nuclear magnetic resonance, cryogenic electron microscopy) have determined and accumulated a large number of atomic-resolution 3D structures of proteins and protein-protein complexes. Also computational method to compute such structures are available, such as Alphafold for protein structures, and PANDORA [@], HADDOCK [@], and Alphafold-Multimer [@] for protein complexes.
[comment]: <> (Motivation for using DL)
Unlike other machine learning techniques, deep neural networks hold the promise of learning from millions of data without reaching a performance plateau quickly, which is computationally tractable by harvesting hardware accelerators (such as GPUs, TPUs) and parallel file system technologies.
[comment]: <> (TODO: Examples of DL with proteins' variants)
[...]
[comment]: <> (Examples of DL with PPIs)
3D deep convolutional networks (CNNs) have been trained on 3D grids for the classification of biological vs. crystallographic PPIs [@deeprank], and for the scoring of models of protein-protein complexes generated by computational docking [@deeprank,@dove]. Gaiza et al. [@] have applied Geodesic CNNs to extract protein interaction fingerprints by applying 2D ConvNets on spread-out protein surface patches [@MaSIF]. GNNs have also been applied to predict protein interfaces [@Fout,@deeprank-gnn]. Finally, rotation-equivariant neural networks have recently been used by Eisman et al. on point-based representation of the protein atomic structure to classify PPIs [@]. 
[comment]: <> (Motivation for a flexible framework)
Data mining 3D proteins and protein complexes presents several unique challenges because of the physico-chemical rules that govern them, the possibility of characterizartion at different levels (atom-atom level, residue-residue level, and secondary structure level), the highly diversity in terms of shapes and sizes, and finally, efficient processing and featurization of a large number of atomic coordinates files of proteins is daunting in terms of computational cost and file storage requirements. There is therefore an emerging need for generic and extensible deep learning frameworks that scientists can easily re-use for their particular problems, while removing tedious phases of data preprocessing. Such generic frameworks have already been developed in various scientific fields ranging from computational chemistry (DeepChem [@]) to condensed matter physics (NetKet [@]) and have significantly contributed to the rapid adoption of machine learning techniques in these fields. They have stimulated collaborative efforts, generated new insights, and are continuously improved and maintained by their respective user communities. This calls for open-source frameworks that can be easily modified and extended by the community for data mining protein complexes and can expedite knowledge discovery on related scientific questions.
[comment]: <> (TODO: add examples of frameworks less flexible and with less features)
[comment]: <> (TODO: add ref to getstarted and describe it at an high level, inserting examples images of the two different starting PDB, stored into the high performance HDF5 file format)
[comment]: <> (TODO: evidenzia modularita', possibilita' di aggiungere le features e di usare solo un pezzo della pipeline)
[comment]: <> (TODO: evidenzia test coverage, continuous integration)

[comment]: <> (TODO: Redo the flowchart including variants as well, and add HDF5 storage and PyTorch)
![DeepRank-Core framework overview. 3D coordinates of PPIs or proteins' variants are extracted from PDB files and converted into graphs or grids, using either an atomic or a residual level, depending on the user’s requirements. The data can be then used in the pre-implemented training pipeline implemented in PyTorch to compute predictions. \label{fig:flowchart}](deeprankcore.png)

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

TODO: Acknowledgement of any financial support (ask Pablo).

# References
