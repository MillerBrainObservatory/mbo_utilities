# suite3d registration flow

## overview

suite3d computes per-plane (y,x) rigid shifts from raw ScanImage TIFFs.
these shifts align z-planes spatially and get applied during binary/tiff
writing to correct for axial misalignment in LBM data.

## entry points

### 1. `writer.py` imwrite (programmatic API)

```
imwrite(lazy_array, outpath, register_z=True)
```

- gets `filenames` from `lazy_array.filenames` (line 306)
- for ScanImageArray: these are the raw .tiff source files
- passes them directly to `register_zplanes_s3d(filenames=lazy_array.filenames, ...)`

### 2. `gui/tasks.py` task_save_as (GUI save-as worker)

```python
filenames = getattr(arr, "filenames", [])     # line 220
if not filenames and hasattr(arr, "_files"):
    filenames = arr._files                     # line 222
```

- `arr` comes from `imread(input_path)` where input_path is the user's input
- passes filenames to `register_zplanes_s3d(filenames=filenames, ...)`

### 3. `gui/tasks.py` task_suite2p (GUI suite2p worker)

```python
filenames = getattr(pipeline_input, "filenames", [])  # line 429
```

- `pipeline_input` may be a string path, list of paths, or an array
- bug: only loads array for `str | Path`, not `list` (fixed in recent commit)

### 4. `gui/widgets/pipelines/settings.py` _run_pipeline_worker_thread

```python
filenames = getattr(input_data, "filenames", [])  # line 1960
```

- `input_data` is the arr or _ChannelView from the pipeline config

## what filenames actually are

for ScanImageArray (the LBM reader):
```python
# arrays/tiff.py line 1041
self.filenames = [files] if isinstance(files, (str, Path)) else list(files)
```

**these are the raw ScanImage .tiff source files** (e.g.
`E:\datasets\lbm\single_cavity\2026-02-26_wsnyder\00001.tif`).

for TiffArray (generic tiff reader):
```python
# arrays/tiff.py line 239  (_SingleTiffPlaneReader)
self.filenames = files  # list of Path objects

# arrays/tiff.py line 522 (volume from groups)
self.filenames = [f for group in plane_groups for f in group]

# arrays/tiff.py line 549 (interleaved)
self.filenames = files

# arrays/tiff.py line 644 (ImageJ hyperstack)
self.filenames = [path]  # single file

# arrays/tiff.py line 694 (explicit plane files)
self.filenames = plane_files
```

**problem**: for TiffArray (non-ScanImage), filenames may include:
- previously-written output tiffs (not raw ScanImage format)
- files that are NOT valid raw ScanImage TIFFs
- files that lack the ScanImage `Artist` metadata tag that suite3d needs

## what suite3d expects

### Job.__init__ (suite3d/job.py)

```python
Job(root_dir, job_id, create=True, tifs=filenames, params=params)
```

- stores `tifs` as `self.tifs` and `self.params["tifs"]`
- calls `self.preregister_tifs()` which for LBM data is a no-op
  (only runs for non-LBM, non-FACED data)

### init_pass.py run_init_pass

```python
init_tifs = choose_init_tifs(tifs, n_init_files=1, ...)
init_mov = jobio.load_data(init_tifs)
```

- `choose_init_tifs` picks `n_init_files` (default 1) from the tifs list
- `jobio.load_data` dispatches to `_load_lbm_tifs` when `params["lbm"]=True`

### s3dio._load_lbm_tifs

```python
for tif_path in paths:
    im = load_and_stitch_full_tif_mp(tif_path, channels, n_ch_tif, ...)
```

### lbmio.load_and_stitch_full_tif_mp

```python
tiffile = tifffile.imread(path)           # <-- MUST be a raw ScanImage TIFF
rois = get_meso_rois(path, ...)           # reads ScanImage Artist JSON tag
```

**requirements**:
1. file MUST be a valid TIFF (not bin, zarr, h5, etc.)
2. file MUST have ScanImage metadata (Artist tag with RoiGroups JSON)
3. pages must be organized as interleaved planes (n_ch_tif planes per volume)

## where results are saved

`register_zplanes_s3d` creates:
```
{outpath}/s3d-{job_id}/
    params.npy
    dirs.npy
    summary/
        summary.npy          <-- main output
        init_mov.npy         <-- saved init movie
```

`summary.npy` contains a dict with key fields:
- `plane_shifts`: ndarray (n_planes, 2) — [dy, dx] per plane
- `ref_img_3d`: registered 3D reference image
- `raw_img`: raw mean image
- `xpad`, `ypad`: padding sizes
- `init_tifs`: which tif files were used

default job_id in mbo_utilities: `"preprocessed"` (from `_registration.py:166`)
default job_id in GUI tasks: `"s3d-preprocessed"` (from `tasks.py:211,405`)

**note**: the output directory is `s3d-{job_id}`, so:
- from `_registration.py`: `{outpath}/s3d-preprocessed/`
- `validate_s3d_registration` checks `{candidate}/summary/summary.npy`
- in GUI tasks, candidate = `{output_dir}/s3d-preprocessed`

## how results are consumed

### load_registration_shifts (_writers.py:207)

```python
summary_path = Path(s3d_job_dir) / "summary" / "summary.npy"
summary = load_npy(summary_path).item()
plane_shifts = np.asarray(summary["plane_shifts"])
padding = compute_pad_from_shifts(plane_shifts)
return True, plane_shifts, padding
```

called from `_write_plane()` during binary/tiff writing. applies shifts via
`apply_shifts_to_chunk()` which pads output and translates each plane by its
(dy, dx) offset.

metadata keys used:
- `apply_shift` (bool): whether to apply shifts
- `s3d-job` (str): path to suite3d job directory
- `_s3d_event` (threading.Event): wait for background suite3d thread (new)

## the bug: "not a TIFF file"

the error occurs at `lbmio.py:33`:
```python
tiffile = tifffile.imread(path)
```

**root cause**: `filenames` passed to `register_zplanes_s3d` contains files
that are NOT raw ScanImage TIFFs. the header `b'\xf6\xff\xf4\xff'` is not
a TIFF magic number (which is `b'II\x2a\x00'` for little-endian or
`b'MM\x00\x2a'` for big-endian).

this happens when:
1. the array was loaded from a non-ScanImage source (e.g. previously written
   tiff, bin, or other format)
2. the `filenames` list contains non-tiff files (e.g. `.bin` files that ended
   up in the list)
3. the input directory contains mixed file types and `imread` picks up non-tiff
   files

the `writer.py` path works because it only runs on `ScanImageArray` objects
which always have valid raw ScanImage TIFF filenames. the GUI task paths
(`task_save_as`, `task_suite2p`, `_run_pipeline_worker_thread`) don't verify
that filenames are actually raw ScanImage TIFFs before passing them to suite3d.

## fix needed

before calling `register_zplanes_s3d`, filter filenames to only include files
that are:
1. actual TIFF files (check magic bytes or extension)
2. ideally raw ScanImage TIFFs (have the Artist metadata tag)

alternatively, if the array is not a ScanImageArray (no ROI metadata), suite3d
registration should be skipped entirely since `_load_lbm_tifs` depends on
ScanImage-specific ROI metadata from the Artist tag.
