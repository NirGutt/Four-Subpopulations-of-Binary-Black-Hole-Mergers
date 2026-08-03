# Revealing Four Subpopulations of Binary Black-Hole Mergers with the Fifth Gravitational-Wave Transient Catalog


This repository provides the code, data products, and supporting results for the paper:

**“Revealing Four Subpopulations of Binary Black-Hole Mergers with the Fifth Gravitational-Wave Transient Catalog”**

[arXiv:2607.22011](https://arxiv.org/abs/2607.22011)

The repository is organised into two main parts:


1. **Supporting data and results**

   The [Supporting data and results](<Supporting data and results/>)  directory contains the prior-range tables and corner plots supporting the conclusions of the paper.


2. **Analysis code and reproduction instructions**

   The [code](code/) directory contains the population models, priors, software environment, data-download scripts, and instructions required to reproduce the hierarchical population analysis using `gwpopulation` and `gwpopulation_pipe`. In addition, a demo folder which runs with quick settings.

   The complete hierarchical inference requires the public GWTC event-level posterior samples and injection data, as well as access to a GPU (preferably on a computing cluster). These large input files are not stored directly in this repository; scripts and additional information for downloading and preparing them are provided under `code/data/`.



