# Canonical Metadata Layer

How `mbo_utilities` represents imaging metadata so that one canonical name
(`fs`, `dx`, `dy`, `dz`, …) is authoritative everywhere — resolved from any
source's aliases on read, queried through a single array API, and fanned back
out to each format's keys on write.

> **Status.** This document describes the target design. The registry,
> alias resolution, and the array query API already exist; the sections
> marked **(planned)** are not implemented yet. See
> [Implementation status](#implementation-status) for the build plan.

## Why

The same physical quantity is stored under different keys by different
software:

| Quantity        | mbo canonical | ScanImage         | Suite2p   | ImageJ/Fiji      | OME            |
|-----------------|---------------|-------------------|-----------|------------------|----------------|
| X pixel size    | `dx`          | `pixel_resolution[0]` | `dx`  | `XResolution`*   | `PhysicalSizeX`|
| Z step          | `dz`          | `actualStackZStepSize`| —     | `spacing`        | `PhysicalSizeZ`|
| Frame rate      | `fs`          | `scanFrameRate`   | `fs`      | `finterval`*     | —              |
| Timepoints      | `num_timepoints` | —              | `nframes` | `frames`         | `SizeT`        |

`*` ImageJ stores the **inverse** (`finterval` = 1/`fs`, `XResolution` =
px-per-µm = 1/`dx`) — see [transform aliases](#two-kinds-of-alias).

The goal: load a stack a collaborator edited and saved in Fiji, and have
`array.fs` / `array.dx` / `array.dz` return the right numbers — with no
call site knowing which key the value came from — then write it back out
to OME-Zarr with all the OME keys populated.

## The four edges + the reactive core

```
                       ┌─────────────────────────────┐
   any source meta ──▶ │  inbound: alias resolution  │ ──▶ canonical dict
   (SI/Fiji/OME/...)   └─────────────────────────────┘
                                     │
                            ┌────────▼────────┐
                            │  reactive core  │  dims + shape + metadata
                            │ DimensionSpecs  │  → per-dim {role,size,scale,unit}
                            └────────┬────────┘
                  ┌──────────────────┼──────────────────┐
            query │                  │ display          │ outbound
        arr.dx/fs │           mbo studio viewer         │ per-format keys
                  ▼                  ▼                  ▼
           (used everywhere)   (canonical values)  (OME-Zarr / ImageJ / ops)
```

## The canonical vocabulary

Defined in **one place**: `mbo_utilities/metadata/base.py`, the
`METADATA_PARAMS` registry. Each entry is a `MetadataParameter`:

```python
@dataclass
class MetadataParameter:
    canonical: str                 # the mbo truth, e.g. "dx"
    aliases: tuple[str, ...] = ()  # other keys that mean the same value
    dtype: type = float            # coerced on read
    unit: str | None = None        # "µm", "Hz", ...
    default: Any = None            # value when absent
    description: str = ""
    label: str = ""                # GUI display label
```

The reverse lookup `ALIAS_MAP` (alias → canonical) is built automatically
from every entry, so you never maintain it by hand.

## Adding a format / alias — the single place

> Someone wants `pixresX` to also mean `dx`.

Add it to the `aliases` tuple of the `dx` entry in `METADATA_PARAMS`. That
is the only edit:

```python
# mbo_utilities/metadata/base.py
"dx": MetadataParameter(
    canonical="dx",
    aliases=(
        "PhysicalSizeX",
        "pixel_size_x",
        "XResolution",
        "pixresX",      # <-- added; nothing else changes
    ),
    dtype=float,
    unit="µm",
    default=1.0,
),
```

`ALIAS_MAP` rebuilds at import, and every consumer downstream —
`get_param`, `array.dx`, `normalize_metadata`, the GUI viewer, every
writer — picks it up. No call site changes.

Adding a brand-new canonical quantity is the same: add a new
`MetadataParameter` entry keyed by its canonical name.

(two-kinds-of-alias)=
### Two kinds of alias

1. **Rename alias** — the value is identical, only the key differs
   (`PhysicalSizeX` ≡ `dx`). Add it to `aliases`.

2. **Transform alias** — the value needs a function to reach the canonical
   form:
   - ImageJ `finterval` (seconds) = `1 / fs`
   - TIFF `XResolution` (px per µm) = `1 / dx`
   - ScanImage `pixel_resolution` (tuple) → `(dx, dy)` (handled as a
     one-off special case in `get_param`).

   Declared alongside the param via `transforms`, so it still lives in one
   place. Each entry maps a source key → `(to_canonical, from_canonical)`:

   ```python
   "fs": MetadataParameter(
       canonical="fs",
       aliases=("frame_rate", "fps", "scanFrameRate"),
       unit="Hz",
       # alias -> (parse_to_canonical, emit_from_canonical)
       transforms={"finterval": (_reciprocal, _reciprocal)},
   ),
   ```

   On read, `get_param("fs")` returns `1/finterval` when only `finterval`
   is present (direct/rename values still win); on write,
   `normalize_metadata` emits `finterval` from `fs`. The nested lookup runs
   with transforms disabled, so reciprocal pairs (`fs`↔`finterval`) can't
   recurse.

   `get_param(meta, "fs")` would then resolve `fs` from a Fiji file that
   only carries `finterval`, and writers could emit `finterval` from `fs`.

## Inbound: `imread`

A reader's only metadata job is to **deposit the source's raw keys** into
`self._metadata` — it must *not* pre-normalize. Resolution happens through
the registry:

```python
from mbo_utilities.metadata import get_param

meta = {"PhysicalSizeX": 0.5, "frame_rate": 7.5}  # what the reader stored
get_param(meta, "dx")   # 0.5   (via PhysicalSizeX)
get_param(meta, "fs")   # 7.5   (via frame_rate)
```

To stamp every alias into a dict (so plain-dict consumers and external
tools see them all), use `normalize_metadata(meta)` /
`normalize_resolution(meta)`.

## Query: the array API

On any `LazyArray`, read canonical values directly — this is **the**
sanctioned access path:

| Property | Meaning | Source |
|----------|---------|--------|
| `arr.dx`, `arr.dy` | pixel size (µm) | metadata, via registry |
| `arr.dz` | z-step (µm), `None` if no Z | metadata, via registry |
| `arr.fs` | frame rate (Hz) | metadata, via registry |
| `arr.finterval` | frame interval (s) | derived from `fs` |
| `arr.num_timepoints`, `arr.num_zplanes` | sizes by dim name | dims + shape |
| `arr.nt/nc/nz/ny/nx` | sizes by 5D position | shape |

```python
arr = mbo.imread("fiji_edited.tif")
arr.dx        # resolves XResolution/PhysicalSizeX/pixel_resolution → µm
arr.fs        # resolves frame_rate/scanFrameRate/finterval → Hz
arr.num_zplanes
```

These delegate to `dimension_specs` (below), so they stay correct when
`dims` or `metadata` change. **Do not** hand-roll fallback chains; the
following anti-pattern in `arrays/mp4.py` is exactly what the layer
exists to delete:

```python
# before — every call site re-implements alias resolution
candidates = [getattr(arr, "dx", None),
              md.get("dx"),
              (md.get("pixel_resolution") or [None])[0]]
dx = next((float(v) for _, v in candidates if v), None)

# after
dx = arr.dx
```

> **Note.** `num_channels` is *not* a base property — `ScanImageArray`
> assigns `self.num_channels` as instance state, so a base property would
> collide. Use `arr.nc` for the channel count.

> **Unknown values.** When a quantity was never stored, `arr.dx`/`arr.dy`
> return `1.0`, `arr.dz`/`arr.fs` return `None` (registry defaults). The
> layer does not distinguish "1.0 µm" from "unknown".

## Reactive core: `DimensionSpecs`

`LazyArray.dimension_specs` builds, from `dims` + `shape` + `metadata`, a
per-dimension `{role, size, scale, unit}` model and caches it. The cache is
dropped by `invalidate_dimension_specs()` whenever `dims` is reassigned, so
the canonical accessors reflect the current state.

Selection/stride reactivity (writing a subset of planes/timepoints) lives
in `OutputMetadata`, which carries a provenance stamp so `dz`/`fs` scale
correctly across multi-hop writes (raw → zarr → tiff → bin) without
double-scaling. `OutputMetadata` is the authoritative output-side layer and
should not be replaced by the simpler one-shot scaling in
`DimensionSpecs.with_selections`.

## Outbound: writers

A writer emits the canonical value under the keys its target format
expects:

- **OME-Zarr (NGFF):** axes + per-axis `scale` (`dz` for Z, `1/fs` for T) +
  `PhysicalSizeX/Y/Z`.
- **ImageJ TIFF:** `finterval`, `spacing`, `XResolution`/`YResolution`,
  `unit`.
- **Suite2p ops:** `fs`, `dx`, `dy`, `nplanes`.

Today `normalize_metadata` and `VoxelSize.to_dict(include_aliases=True)`
fan out the OME/ImageJ/legacy spatial keys. The **(planned)** consolidation
is to drive per-format emission from the registry (and the transform
aliases) instead of the hand-written scale loops in
`OutputMetadata.to_ome_ngff` / `to_imagej`.

## Display: mbo studio

The metadata viewer is driven by `IMAGING_METADATA_KEYS` and reads the
canonical, resolved values (with `unit` and `label` from the registry), so
it shows accurate numbers regardless of which software wrote the file.

## Adding a new array class

A new reader participates in the whole layer for free by following the
`LazyArray` contract:

1. Implement `can_open`, `_shape5d`, `__getitem__`, `dtype`.
2. Set `self._metadata` to the source's raw keys (do not pre-normalize).

That's it — `arr.dx`, `arr.fs`, `arr.num_zplanes`, `dimension_specs`, the
GUI viewer, and the writers all work, resolving whatever aliases the source
used.

(implementation-status)=
## Implementation status

Already in place:

- ✅ `METADATA_PARAMS` + `ALIAS_MAP` registry (`metadata/base.py`)
- ✅ `get_param` rename-alias resolution + `pixel_resolution` tuple case
- ✅ `normalize_metadata` / `normalize_resolution` / `VoxelSize.to_dict`
- ✅ `dimension_specs` reactive derivation; `OutputMetadata` provenance
- ✅ `arr.dx/dy/dz/fs/finterval` and `arr.num_timepoints/num_zplanes`
- ✅ **Transform aliases** (`finterval`↔`fs`, `XResolution`↔`dx`,
  `YResolution`↔`dy`) — reciprocal derivation on read + emission on write,
  recursion-guarded. `arr.fs` now resolves an ImageJ `finterval`.

Planned (this initiative):

- 🔲 **Wide adoption of `arr.dx`/`arr.fs`** — replace scattered
  `get_param(arr.metadata, …)` and the `mp4`-style fallback chains.
- 🔲 **Inbound coverage audit** — confirm each reader (esp. the ImageJ/OME
  TIFF path) deposits the source tags so the registry can resolve them.
- 🔲 **Registry-driven per-format write emission** for OME-Zarr / ImageJ /
  ops.

See also: [Dim/metadata contract consolidation](dim_metadata_refactor.md).
