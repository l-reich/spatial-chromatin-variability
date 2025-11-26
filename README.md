# Investigating Multi-Omic Mapping Methods to Uncover Chromatin Variability at the Single-Cell Level

**Bachelor’s Thesis – Luis Raphael Reich**  
Technical University of Munich & LMU Munich  

## Overview

This project investigates **spatial variability in chromatin accessibility** at the single-cell level by integrating **scRNA-seq**, **scATAC-seq**, and **spatial transcriptomics** data. The work combines state-of-the-art methods from computational biology and spatial omics to identify putative cis-regulatory interactions and uncover spatially structured gene regulation.

## Methods

- **Datasets**
  - [Spatial transcriptomics](https://doi.brainimagelibrary.org/doi/10.35077/g.21) (MERFISH, mouse primary motor cortex)
  - [Joint scRNA-seq + scATAC-seq](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE140203) (SHARE-seq, whole mouse brain)

- **Core tools**
  - [Tangram](https://github.com/broadinstitute/Tangram): Spatial mapping of single-cell data
  - [Descart](https://github.com/likeyi19/Descart): Detection of spatially variable peaks
  - [scverse stack](https://github.com/scverse): `scanpy`, `squidpy`, `muon` for data processing

- **Analyses**
  - Mapping chromatin accessibility onto spatial coordinates
  - Correlation analysis of peak accessibility with spatial gene expression
  - cCRE annotation overlap and novel peak discovery
  - Motif enrichment analysis (JASPAR)

## Key Findings

- Tangram preserves known spatial cell-type distributions when mapping SHARE-seq data.
- Peaks correlated with niche-specific marker genes are enriched in those spatial niches.
- Many such peaks overlap with canonical cCREs, and some may represent novel elements.
- SPI1 motif-enriched peaks show strong microglial enrichment, reflecting known biology.

## Thesis

A detailed explanation of the methods, analyses, and findings is provided in the thesis manuscript (work in progress):

> **[Thesis PDF](./06-17-thesis-draft.pdf)**

## Structure (important files)

```
├── README.md                  # Project overview
├── results/                   # Output results and figures (not yet uploaded)
├── src/                       # Source code and notebooks
│   ├── niche_analysis.ipynb   # Spatial niche detection with NichePCA (https://github.com/imsb-uke/nichepca)
│   ├── running_tg.ipynb       # Tangram spatial mapping
│   ├── preprocessing/         # Preprocessing scripts
│   │   ├── SHARE-seq_pp.ipynb
│   │   └── spatial_pp.py
│   └── Descart/               # Descart-based peak-gene correlation
│       ├── Gene-peak_interaction_detection.ipynb
│       ├── correlation_analysis.ipynb
│       ├── descart.py
│       ├── peak_gene_utils.py
```
