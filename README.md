**TRIAGE-MIL: Multi-Axis Instance Selection and Semantic Hypergraph Modeling for Survival Prediction from Whole-Slide Images**.

TRIAGE-MIL is a weakly supervised survival prediction framework for whole-slide images (WSIs). It combines:

1. **MASS**: Multi-Axis Stratified Sampling for unsupervised tile selection.
2. **Semantic hierarchical hypergraph modeling** for higher-order tissue relationship modeling.
3. **Gated attention MIL survival prediction** trained with Cox partial likelihood.

## Overview

Whole-slide images contain tens of thousands of tiles, but only a subset may carry prognostic information. TRIAGE-MIL first filters low-quality tile embeddings, then selects a fixed-size subset using four phenotype-inspired axes:

- Morphologic heterogeneity
- Tissue regularity
- Microenvironmental interface
- Tissue diversity

The selected tiles are used to construct a semantic hierarchical hypergraph for patient-level survival prediction.

## Repository Structure

```text
configs/     Example configuration files
src/         Core implementation
scripts/     Training and evaluation scripts
examples/    Toy metadata/config examples
docs/        Data format and method documentation
assets/      Figures and overview images
