# Migrating from v3 to v4

v4 makes `imread()` a pluggable, always-5D API. This is a hard break.

## 1. Arrays are always 5D TCZYX

Every array `imread()` returns now reports a 5-tuple shape and `ndim == 5`,
with singleton axes kept (not squeezed).

```python
arr = imread("image2d.tif")
# v3: arr.shape == (Y, X),        arr.ndim == 2
# v4: arr.shape == (1, 1, 1, Y, X), arr.ndim == 5

arr = imread("timeseries.tif")
# v3: (T, Y, X)
# v4: (T, 1, 1, Y, X)
```

`shape5d` still works and is now just an alias for `shape` (deprecated; will
be removed in a later minor).

## 2. Indexing follows numpy 5D semantics

An integer index drops only that axis:

```python
arr[0]          # -> (C, Z, Y, X)  (drops T)
arr[0, 0, 0]    # -> (Y, X)        a single frame
arr[:, 0, 0]    # -> (T, Y, X)
```

In v3 a natural-rank `arr[0]` on a `(T, Y, X)` array returned `(Y, X)`. In v4,
index the singleton axes explicitly (`arr[0, 0, 0]`) or use squeeze (below).

## 3. Opt-in squeeze for the old ergonomics

```python
arr = imread("timeseries.tif", squeeze=True)   # -> (T, Y, X)
# or
arr = imread("timeseries.tif").squeeze()       # SqueezedView, (T, Y, X)
```

`squeeze()` returns a lightweight `SqueezedView`; indexing on it is translated
back to the 5D array, and writers still operate on the canonical 5D data.

## 4. `imread()` dispatch is pluggable

Format selection now runs through `can_open()` + `PRIORITY` instead of a fixed
if/elif chain. Built-ins behave the same. To add or override a format, register
a `LazyArray` subclass — no edits to `mbo_utilities` (see `docs/forking.md`).

## 5. Unaffected helpers

- `BinArray` (Suite2p `.bin` read/write helper) stays 3D `(T, Y, X)`.
- `MP4Array` (video I/O) stays as-is.

Both remain reachable through `imread()`; they are not part of the 5D dispatch
set.
