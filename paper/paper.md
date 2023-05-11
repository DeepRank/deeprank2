---
title: 'Deeprank-Core: Mining Protein-Protein Structures with Geometric Deep Learning'
tags:
  - Python
  - PyTorch
  - structural biology
  - geometric deep learning
  - 
authors:
  - name: Giulia Crocioni
    orcid: 0000-0002-0823-0121
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
date: 13 August 2017
bibliography: paper.bib

# Optional fields if submitting to a AAS journal too, see this blog post:
# https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary

TODO: A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience.
Has a clear description of the high-level functionality and purpose of the software for a diverse, non-specialist audience been provided?

The combination of physics-based 3D modeling and data-driven deep learning has the potential
to revolutionize drug design and protein engineering. There is a wealth of Protein-Protein Interface
(PPI) data available, both experimentally and computationally obtained, that can be used to train deep
learning models for biologically relevant predictions. We previously developed DeepRank [@renaud_deeprank_2021] and DeepRank-GNN [@reau_deeprank-gnn_2022],
two deep learning frameworks for PPIs data mining using Convolutional Neural Networks (CNNs) and Graph
Neural Networks (GNNs), respectively. We present here DeepRank-Core, a unified and user-friendly open-source
deep learning framework that converts 3D representations of PPIs into either grids or graphs for efficient
training of CNNs or GNNs. DeepRank-Core is designed to be customizable, offering users the ability to choose
the deep learning architecture that best fits the specific interaction patterns they aim to model. 

# Statement of need

TODO: A Statement of need section that clearly illustrates the research purpose of the software and places it in the context of related work.
Do the authors clearly state what problems the software is designed to solve and who the target audience is?
TODO: State of the field: Do the authors describe how this software compares to other commonly-used packages?
TODO: Mention (if applicable) a representative set of past or ongoing research projects using the software and recent scholarly publications enabled by it.

![DeepRank-Core framework overview. 3D coordinates of interface residues are extracted from PDB files and converted into interface graphs or grids, depending on the user’s choice. The data are then passed through a Neural Network to compute predictions. \label{fig:flowchart}](deeprankcore.png)
and referenced from text using \autoref{fig:flowchart}.

# Mathematics

Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures

Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements

We acknowledge contributions from Brigitta Sipocz, Syrtis Major, and Semyeong
Oh, and support from Kathryn Johnston during the genesis of this project.

TODO: Acknowledgement of any financial support (ask Pablo).

# References
