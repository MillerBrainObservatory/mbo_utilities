# mbo_utilities

This repository contains python functions to pre/post process datasets recording at the [Miller Brain Observatory](https://mbo.rockefeller.edu)

## Overview

```{toctree}
---
maxdepth: 2
---
notebooks/assembly.ipynb
api/index
glossary
```

## Assembling 

Assembling reconstructed images from raw LBM datasets consists of 3 main processing steps:

1. {ref}`De-interleave <ex_deinterleave>` z-planes and timesteps.
2. {ref}`Correct Scan-Phase <ex_scanphase>` alignment for each ROI.
3. {ref}`Re-tile <assembly_retiled>` vertically concatenated ROI's horizontally.

```{thumbnail} ../_images/assembly_stripped.svg
```
