# Calculating ΔF/F0

Before reading the below guide, please read [this blog-post](https://www.scientifica.uk.com/learning-zone/how-to-compute-%CE%B4f-f-from-calcium-imaging-data) by the great Dr Peter Rupprecht at the University of Zurich.

This guide covers common approaches to extracting calcium activity and how this will differ depending on which pipeline you are using.

## Key takeaways

- ∆F/F reflects intracellular calcium levels, but often in a nonlinear way
- ∆F/F was initially introduced for organic dye calcium indicators in the 1980s
- GCaMP behaves differently due to its low baseline brightness and its nonlinearity
- There is no single recipe on how to compute ∆F/F
- Computation of ∆F/F must be adapted to cell types, activity patterns, and noise
- Interpretation of ∆F/F requires knowledge about indicators, cell types, and confounds
- Potential confounds are brain motion, neuropil contamination, and response variability across neurons

## What is ΔF/F0

"Delta F over F naught", abbr. ΔF/F0, is the change in calcium signal over time, normalized by the baseline signal.

## Overview

This table summarizes Calcium Activity (ΔF/F) detection methods reviewed by Paudel et al. [(2024)](https://doi.org/10.3390/biom14010138).

| **Method**                         | **How It's Performed**                                                                 | **Pros**                                                      | **Cons**                                                             |
|-----------------------------------|-----------------------------------------------------------------------------------------|----------------------------------------------------------------|------------------------------------------------------------------------|
| **Manual Spike Counting**         | Visual inspection of raw fluorescence or dF/F traces to count events.                  | Simple; no code needed.                                        | Subjective; non-scalable; low reproducibility.                       |
| **dF/F Thresholding**             | Compute (F − F₀)/F₀ and define a fixed or adaptive threshold for events.              | Widely used; compatible with GCaMP, Fluo dyes.                | Sensitive to F₀ definition; arbitrary thresholds can bias results.  |
| **Z-Score Thresholding**          | Normalize trace by mean/SD, define events above N standard deviations.                | Removes baseline drift; good for noisy data.                   | Sensitive to noise if SD is low; assumes Gaussian distribution.      |
| **Percentile Baseline Subtraction**| Use a moving window to define F₀ as low percentile (e.g., 10–20th) of the trace.       | Adaptive baseline; handles long-term drift.                    | Choice of window size/percentile affects sensitivity.                |
| **Ratiometric Imaging (Fura, YC2)**| Compute ratio of Ca²⁺-sensitive and insensitive fluorophores per ROI.                 | Controls for volume/motion artifacts; yields [Ca²⁺].          | Requires dual excitation/emission; reduced spatial/temporal res.    |
| **Photon Counting (Aequorin)**    | Count emitted photons per ROI/pixel over time; map to Ca²⁺ levels via calibration.     | Quantitative; good dynamic range.                              | Low spatial resolution; complex calibration.                         |
| **Image Subtraction (Frame-to-Frame)**| Compute ΔF = Fₙ − Fₙ₋₁ or F − background to detect sudden changes.                   | Simple, fast; used for wave detection.                         | Sensitive to noise; misses gradual changes.                          |
| **Savitzky-Golay Filtering**      | Smooth trace to reduce noise while preserving spikes.                                  | Good for noisy signals.                                        | Requires tuning; can mask small or fast events.                      |
| **Tensor Voting / Cluster Detection**| Identify spatially coordinated activity from 2D/3D image stacks.                       | Detects population events (e.g., waves).                       | Not standard; needs spatially dense data.                            |
| **Standard Deviation (SD) Masking**| Define active frames/regions where ΔF exceeds N×SD of baseline.                        | Objective thresholding for event detection.                    | Threshold choice heavily affects results.                            |

## Calculating ΔF/F

The most important consideration you must consider is how you calculate your baseline activity.

```{figure} ./_images/dff_1.png
:name: fig-dff-example
:width: 600px
:alt: Example ΔF/F trace baseline comparisons

Example of ΔF/F trace showing different baseline choices. Adapted from Fig. 1 of {cite}`10.3390/biom14010138`.
```

## References 

https://www.scientifica.uk.com/learning-zone/how-to-compute-%CE%B4f-f-from-calcium-imaging-data
