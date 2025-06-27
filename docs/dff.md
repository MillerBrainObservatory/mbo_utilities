---
bibliography:
  - references.bib
---

# Quantifying Cell Activity

The gold-standard formula for measuring cellular activity is "Delta F over F₀", or the change in fluorescence intensity normalized by the baseline activity:

```{math}
:class: center large

\Delta F/F_0 = \frac{F - F_0}{F_0}
```
Here, F₀ a user-defined baseline that may be static (e.g., median over all frames) or dynamically estimated using a rolling window.

This guide assumes you've read this please read [this scientifica article](https://www.scientifica.uk.com/learning-zone/how-to-compute-%CE%B4f-f-from-calcium-imaging-data) by Dr Peter Rupprecht at the University of Zurich.

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

This table summarizes Calcium Activity detection methods used in embryoinic development reviewed by {cite:t}`huang`.

| **Method**                         | **How It's Performed**                                                                 | **Pros**                                                      | **Cons**                                                             |
|-----------------------------------|-----------------------------------------------------------------------------------------|----------------------------------------------------------------|------------------------------------------------------------------------|
| **Percentile Baseline Subtraction**| Use a moving window to define F₀ as low percentile (e.g., 10–20th) of the trace.       | Adaptive baseline; handles long-term drift.                    | Choice of window size/percentile affects sensitivity.                |
| **Z-Score Thresholding**          | Normalize trace by mean/SD, define events above N standard deviations.                | Removes baseline drift; good for noisy data.                   | Sensitive to noise if SD is low; assumes Gaussian distribution.      |
| **dF/F Thresholding**             | Compute (F − F₀)/F₀ and define a fixed or adaptive threshold for events.              | Widely used; compatible with GCaMP, Fluo dyes.                | Sensitive to F₀ definition; arbitrary thresholds can bias results.  |
| **Standard Deviation (SD) Masking**| Define active frames/regions where ΔF exceeds N×SD of baseline.                        | Objective thresholding for event detection.                    | Threshold choice heavily affects results.                            |
| **Image Subtraction (Frame-to-Frame)**| Compute ΔF = Fₙ − Fₙ₋₁ or F − background to detect sudden changes.                   | Simple, fast; used for wave detection.                         | Sensitive to noise; misses gradual changes.                          |

## Pipelines

There are several pipelines for users to choose from when deciding how to process calcium imaging datasets.

Often times, the outputs of these pipelines are in units that are not well documented.
Even worse, the *inputs* to downstream pipelines pipelines require data to have particular units.

(bg_sub_example)=
```{admonition} Example
:class: dropdown

[This discussion](https://gcamp6f.com/2021/10/04/large-scale-calcium-imaging-noise-levels/), which details how to calculate noise for a variety of datasets, used traces that had [the background signal subtracted](https://github.com/cajal/microns_phase3_nda/issues/21).

This happens because CaImAn subtracts the background signal from each neuron under the hood.
The resulting units are no longer raw signal, they are background-subtracted raw-signal.
This process reduces the baseline F₀ to nearly zero and heavily skew ΔF/F₀ calculations.
```

### [CaImAn](https://github.com/flatironinstitute/CaImAn)

See: {func}`detrend_df_f <caiman:caiman.source_extraction.cnmf.utilities.detrend_df_f>`

CaImAn computes ΔF/F₀ using a **running low-percentile baseline**.

```{figure}

```

By default, it uses the **8th percentile** over a **500 frame** window.
The idea is to track the lower envelope of the signal to get F₀ without being biased by transients.

**Neuropil/background:** CaImAn handles this as part of its CNMF model {cite:p}`cnmf`.

Background and neuropil are explicitly separated into distinct spatial/temporal components, so the output traces are background subtracted during this [factorization](https://en.wikipedia.org/wiki/Matrix_decomposition) (as was the issue in the above {ref}`example <bg_sub_example>`).

There is a strong argument to be made that a matrix factorization `CNMF` is not complex enough to model the true background and neuropil.

{cite:t}`caiman`

### [Suite2p](https://github.com/MouseLand/suite2p/tree/main)

Suite2p does **not** output traces in ΔF/F₀ format directly.

Instead, it gives you the raw trace and the neuropil, along with spike estimates if you ran [deconvolution](https://suite2p.readthedocs.io/en/latest/deconvolution.html).
The neuropil represents fluorescence from the surrounding non-somatic tissue. 
As an optional step, many experimentors apply a fixed subtraction:

```python
# F is an [n_neurons x time] array of raw signal
# Fneu is an [n_neurons x time] array of neuropil
F_corrected = F - 0.7 * Fneu
```

The 0.7 is an empirically chosen scalar to account for the partial contamination.

To compute ΔF/F₀, you divide trace, be that neuropil-corrected or not, by an F₀ you calculate yourself.

The default F₀ estimate comes from a **"maximin"** filter: smooth the trace with a Gaussian (default \~10 s), take a rolling **min**, then a **max** over a 60 s window.

{cite:t}`suite2p`

### [EXTRACT](https://github.com/schnitzer-lab/EXTRACT-public?tab=readme-ov-file)

EXTRACT outputs raw fluorescence signals without built-in ΔF/F₀ calculation. You compute it yourself using something like a low-percentile (e.g. 10%) as F₀. Most people use a global or sliding percentile window.

**Neuropil:** Handled implicitly. The algorithm uses robust factorization to ignore background and neuropil. There’s no explicit subtraction or coefficient to tune. It isolates only what fits a consistent spatial footprint and suppresses outliers by design.

{cite:t}`extract2017`, {cite:t}`extract2021`

| **Pipeline** | **F₀ Method**                       | **ΔF/F₀**                 | **Neuropil**                            |
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

(dynamics)=
## Indicator Dynamics

| **GCaMP Variant**     | **Optimal Tau (s)** | **Notes / Sources** |
|-----------------------|---------------------|----------------------|
| **GCaMP6f (fast)**    | ~0.5–0.7 s          | Suite2p: ~0.7 s ([docs](https://suite2p.readthedocs.io)). OASIS/CNMF: ~0.5–0.7 s ([PMC](https://www.ncbi.nlm.nih.gov/pmc/)). CaImAn: ~0.4 s ([docs](https://caiman.readthedocs.io)). |
| **GCaMP6m (medium)**  | ~1.0–1.25 s         | Suite2p: ~1.0 s. OASIS: ~1.25 s ([PMC](https://www.ncbi.nlm.nih.gov/pmc/)). CaImAn: ~1.0 s ([docs](https://caiman.readthedocs.io)). |
| **GCaMP6s (slow)**    | ~1.5–2.0 s          | Suite2p: 1.25–1.5 s. OASIS/Suite2p: ~2.0 s ([PMC](https://www.ncbi.nlm.nih.gov/pmc/)). CaImAn: ~1.5–2.0 s. |
| **GCaMP7f (fast)**    | ~0.5 s              | Spike inference tuning ([bioRxiv](https://www.biorxiv.org)). Similar to GCaMP6f. |
| **GCaMP7m (medium)**  | ~1.0 s (est.)       | Estimated by analogy to GCaMP6m. No default in tools. |
| **GCaMP7s (slow)**    | ~1.0–1.5 s          | In vivo half-decay ~0.7 s ([eLife](https://elifesciences.org)). Tau ≈ 1.0 s. |
| **GCaMP8f (fast)**    | ~0.3 s              | OASIS fine-tuning ([bioRxiv](https://www.biorxiv.org)). Fastest decay; tenfold faster than 6f/7f ([Nature](https://www.nature.com)). |
| **GCaMP8m (medium)**  | ~0.3 s              | Slightly slower than 8f, still ~0.3 s ([bioRxiv](https://www.biorxiv.org); [Nature](https://www.nature.com)). |
| **GCaMP8s (slow)**    | ~0.7 s              | Spike inference optimal tau ~0.7 s ([bioRxiv](https://www.biorxiv.org)). Faster than 6s ([Nature](https://www.nature.com)). |

## References

```{bibliography}
```
