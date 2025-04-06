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

(assembly_guide)=
## Assembling 

Assembling reconstructed images from raw LBM datasets consists of 3 main processing steps:

1. {ref}`De-interleave <ex_deinterleave>` z-planes and timesteps.
2. {ref}`Correct Scan-Phase <ex_scanphase>` alignment for each ROI.
3. {ref}`Re-tile <assembly_retiled>` vertically concatenated ROI's horizontally.

```{thumbnail} ../_images/assembly_stripped.svg
```

(scan_phase)=
### Scan Phase

In addition to the standard parameters, users should be aware of the implications that bidirectional scan offset correction has on your dataset.

The {code}`fix_scan_phase` parameter attempts to maximize the phase-correlation between each line (row) of each vertically concatenated strip.

This example shows that shifting every *other* row of pixels +2 (to the right) in our 2D reconstructed image will maximize the correlation between adjacent rows.

```{thumbnail} ../_images/ex_phase.png

```
