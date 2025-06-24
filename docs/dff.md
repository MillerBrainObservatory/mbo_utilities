---
bibliography:
  - references.bib
---

# Quantifying Cell Activity (ΔF/F₀)

$\Delta F/F_0 = \frac{F - F_0}{F_0}$

This guide covers common approaches to extracting calcium activity and how this will differ depending on which pipeline you are using.

The below guide assumes you've read this please read [this blog-post](https://www.scientifica.uk.com/learning-zone/how-to-compute-%CE%B4f-f-from-calcium-imaging-data) by Dr Peter Rupprecht at the University of Zurich.

::::{grid}
:::{grid-item-card} Blog-Post Takeaways
:columns: 12

- $\Delta F/F$ reflects intracellular calcium levels, but often in a nonlinear way  
- $\Delta F/F$ was initially introduced for organic dye calcium indicators in the 1930s  
- GCaMP behaves differently due to its low baseline brightness and its nonlinearity  
- There is no single recipe on how to compute $\Delta F/F$  
- Computation of $\Delta F/F$ must be adapted to cell types, activity patterns, and noise  
- Interpretation of $\Delta F/F$ requires knowledge about indicators, cell types, and confounds
:::
::::

## Overview

This table summarizes Calcium Activity detection methods reviewed by {cite:t}`huang`.

| **Method**                         | **How It's Performed**                                                                 | **Pros**                                                      | **Cons**                                                             |
|-----------------------------------|-----------------------------------------------------------------------------------------|----------------------------------------------------------------|------------------------------------------------------------------------|
| **Percentile Baseline Subtraction**| Use a moving window to define F₀ as low percentile (e.g., 10–20th) of the trace.       | Adaptive baseline; handles long-term drift.                    | Choice of window size/percentile affects sensitivity.                |
| **Z-Score Thresholding**          | Normalize trace by mean/SD, define events above N standard deviations.                | Removes baseline drift; good for noisy data.                   | Sensitive to noise if SD is low; assumes Gaussian distribution.      |
| **dF/F Thresholding**             | Compute (F − F₀)/F₀ and define a fixed or adaptive threshold for events.              | Widely used; compatible with GCaMP, Fluo dyes.                | Sensitive to F₀ definition; arbitrary thresholds can bias results.  |
| **Standard Deviation (SD) Masking**| Define active frames/regions where ΔF exceeds N×SD of baseline.                        | Objective thresholding for event detection.                    | Threshold choice heavily affects results.                            |
| **Image Subtraction (Frame-to-Frame)**| Compute ΔF = Fₙ − Fₙ₋₁ or F − background to detect sudden changes.                   | Simple, fast; used for wave detection.                         | Sensitive to noise; misses gradual changes.                          |

## Pipelines

You need to know the output units f

### CaImAn

See: {func}`detrend_df_f <caiman:caiman.source_extraction.cnmf.utilities.detrend_df_f>`

CaImAn computes ΔF/F₀ using a **running low-percentile baseline**.
By default, it uses the **8th percentile** over a **500 frame** window.
The idea is to track the lower envelope of the signal to get F₀ without being biased by transients.

**Neuropil/background:** CaImAn handles this as part of its CNMF model {cite:p}`cnmf`.
Background and neuropil are explicitly separated into distinct spatial/temporal components, so the output traces are already cleaned.

{cite:t}`caiman`

### Suite2p

Suite2p does **not** output traces in ΔF/F₀ format directly.
Instead, it gives you baseline-detrended fluorescence (i.e., $\Delta F$).

The default F₀ estimate comes from a **"maximin"** filter: smooth the trace with a Gaussian (default \~10 s), take a rolling **min**, then a **max** over a 60 s window. Alternatively, you can use a **constant F₀**, either the **8th percentile** or the minimum of a smoothed trace.

If you want ΔF/F₀, you divide the detrended trace by an F₀ value manually after the fact.

Suite2p subtracts **0.7 × F<sub>neu</sub>** (surrounding fluorescence) from each ROI trace before baseline correction.
This is a fixed fraction, applied uniformly.

{cite:t}`suite2p`

### EXTRACT

EXTRACT outputs raw fluorescence signals without built-in ΔF/F₀ calculation. You compute it yourself using something like a low-percentile (e.g. 10%) as F₀. Most people use a global or sliding percentile window.

**Neuropil:** Handled implicitly. The algorithm uses robust factorization to ignore background and neuropil. There’s no explicit subtraction or coefficient to tune. It isolates only what fits a consistent spatial footprint and suppresses outliers by design.

{cite:t}`extract2017`, {cite:t}`extract2021`


## Comparison Table

| **Pipeline** | **F₀ Method**                       | **ΔF/F₀**                 | **Neuropil Handling**                            |
| ------------ | ----------------------------------- | ------------------------- | ------------------------------------------------ |
| **CaImAn**   | 8th percentile, 500-frame window    | Yes, in pipeline          | Modeled via CNMF, no manual subtraction          |
| **Suite2p**  | Maximin (default) or 8th percentile | No, user divides post hoc | 0.7 × F<sub>neu</sub> subtracted before baseline |
| **EXTRACT**  | User-defined (e.g. 10th percentile) | No, user computes         | Implicitly handled via robust model              |

## Calculating ΔF/F

The most important consideration you must consider is how you calculate your baseline activity, which depends on your experimental question.

```{figure} ./_images/dff_1.png
:name: fig-dff-example
:width: 600px
:alt: Example ΔF/F trace baseline comparisons

Example of ΔF/F trace showing different baseline choices. Adapted from Fig. 1.
```

1. Spike inference from mouse spinal cord calcium imaging data ({cite:t}`rupprecht2024`)
- DF/F = Lower 10th percentile  
- GCaMP6s

2. Spike inference on GCaMP8 indicators {cite:t}`rupprecht2025`  
- GCaMP8, GCaMP7f

3. GCaMP6 calibration study {cite:t}`huang`
- ΔF/F = (F − F₀,local)/F₀,global, where F₀,local is the mean fluorescence over 100 ms before the first AP, and F₀,global is the minimum F₀,local across trials  
- GCaMP6f

## References

```{bibliography}
```
