# BinaryFile in the Suite2p Pipeline

! Generated with ChatGPT

The `BinaryFile` class from `suite2p.io.binary` is used to read, write, and process raw binary image data for the Suite2p pipeline.
This file provides several methods and properties that allow it to act like a NumPy array, making it easy to load movies, save data from TIFFs or numpy arrays, and support subsequent operations (such as detection and extraction).
Below you will find an overview of its usage.

---

## Table of Contents

- [Overview](#overview)
- [Initialization and Construction](#initialization-and-construction)
- [Reading and Writing Data](#reading-and-writing-data)
- [Key Properties and Methods](#key-properties-and-methods)
  - [Properties](#properties)
  - [Methods](#methods)
- [Example Usages in the Pipeline](#example-usages-in-the-pipeline)
- [Converting a TIFF to a Suite2p Binary](#converting-a-tiff-to-a-suite2p-binary)
- [Additional Utilities](#additional-utilities)
- [References](#references)

---

## Overview

The `BinaryFile` class is designed to represent image data stored in a raw binary file format that is used by Suite2p. It creates or opens a binary file with specific dimensions:
- **Ly**: Height of each frame.
- **Lx**: Width of each frame.
- **n_frames**: Total number of frames (if writing).

It uses NumPy’s memmap so that the file can be accessed as if it were an array, supporting efficient read/write operations without loading the entire file into memory.

---

## Initialization and Construction

When you instantiate a `BinaryFile` object, you must provide:
- **Ly** and **Lx**: The dimensions of each frame.
- **filename**: The location of the binary file.
- **n_frames**: The number of frames (required when writing a new file).
- **dtype**: The data type (default is `"int16"`).

**Examples:**

```python
# Writing a new binary file
bf = BinaryFile(Ly=448, Lx=448, filename="data_raw.bin", n_frames=1000, dtype="int16")

# Opening an existing binary file for reading/updating (n_frames is determined automatically)
bf = BinaryFile(Ly=448, Lx=448, filename="data_raw.bin", dtype="int16")
```

When a new file is created, if `n_frames` is not provided (and the file does not exist), the constructor raises a `ValueError` to ensure the full size is specified.

---

## Reading and Writing Data

The `BinaryFile` object behaves like a NumPy array for both reading and writing:

- **Writing**: You can assign an entire movie (or slices) to the object using standard NumPy assignment.
  
  ```python
  bf[:] = my_movie  # writes all frames into the binary file
  ```

- **Reading**: You retrieve portions of the data via slicing.
  
  ```python
  frame0 = bf[0]  # retrieves the first frame
  movie_slice = bf[10:20]  # retrieves frames 10 to 19
  ```

The class also supports closing the memmap using a context manager:

```python
with BinaryFile(Ly=448, Lx=448, filename="data_raw.bin", n_frames=1000) as bf:
    movie = bf[:]
# The file is automatically closed afterward.
```

---

## Key Properties and Methods

### Properties

- **n_frames**:  
  Returns the total number of frames in the file as determined by the file size and the number of bytes per frame.

- **shape**:  
  A tuple `(n_frames, Ly, Lx)` showing the dimensions of the movie.

- **nbytesread**:  
  The number of bytes per frame (calculated as `2 * Ly * Lx` for `int16` data).

- **nbytes**:  
  The total size in bytes of the binary file.

- **data**:  
  Returns all frames in the file (using slicing on the underlying memmap).

### Methods

- **`__getitem__` and `__setitem__`**:  
  Allow for NumPy-like slicing and assignment.
  
- **`close()`**:  
  Closes the memmap, releasing system resources.

- **`sampled_mean()`**:  
  Computes the mean image from a subset of frames.

- **`bin_movie(bin_size, ...)`**:  
  Returns a binned version of the movie by computing the mean over time bins. This is used during processing when you need to reduce the temporal resolution.

- **`write_tiff(fname, range_dict={})`**:  
  Exports the binary file’s contents as a TIFF file. You can pass a dictionary specifying ranges for frames, x, or y dimensions.

- **`convert_numpy_file_to_suite2p_binary(from_filename, to_filename)`**:  
  A static method to convert numpy arrays (saved as `.npy` or `.npz`) into the Suite2p binary format.

---

## Example Usages in the Pipeline

### 1. Writing Binaries for Each Plane

Within the Suite2p pipeline, `BinaryFile` is used to write raw binary data for each imaging plane. For example, if the binary data already exists for a plane, the pipeline writes the ops (operation settings) to disk:

```python
ops_paths = [os.path.join(f, "ops.npy") for f in plane_folders]
for i, (f, opf) in enumerate(zip(plane_folders, ops_paths)):
    ops["bin_file"] = os.path.join(f, "data.bin")
    ops["Ly"] = ops["Lys"][i]
    ops["Lx"] = ops["Lxs"][i]
    nbytesread = np.int64(2 * ops["Ly"] * ops["Lx"])
    ops["nframes"] = os.path.getsize(ops["bin_file"]) // nbytesread
    np.save(opf, ops)
```

### 2. Reading Data for Detection

During the detection phase, the pipeline opens the binary file to read its contents and process the detected signals:

```python
with BinaryFile(Ly=op['Ly'], Lx=op['Lx'], filename=op['reg_file']) as f_reg:
    op['neuropil_extract'] = True
    op, stat = suite2p.detection.detection_wrapper(f_reg, ops=op)
    cell_masks, neuropil_masks = masks.create_masks(stat, op['Ly'], op['Lx'], ops=op)
    # Save output for further processing or GUI display
    np.save('expected_detect_output_%ip%ic%i.npy' % (ops['nchannels'], ops['nplanes'], i), output_dict)
```

Here, the `BinaryFile` is used in a context manager so that the data in the registration binary file can be processed by the detection wrapper.

---

## Converting a TIFF to a Suite2p Binary

The following example shows how to convert a TXY TIFF into a Suite2p-compatible binary file using `BinaryFile`:

```python
import tifffile
import numpy as np
from suite2p.io.binary import BinaryFile
from pathlib import Path

def tiff_to_suite2p_binary(tiff_path, out_binary, dtype="int16"):
    # Read the TIFF file, expecting shape (T, Y, X)
    data = tifffile.imread(str(Path(tiff_path)))
    if data.ndim != 3:
        raise ValueError("TIFF must be 3D (T, Y, X)")
    
    T, Y, X = data.shape
    # Create a BinaryFile for writing the data
    bf = BinaryFile(Ly=Y, Lx=X, filename=str(Path(out_binary)), n_frames=T, dtype=dtype)
    
    # Write the entire movie into the binary file
    bf[:] = data
    bf.close()
    
    print(f"Wrote binary file '{out_binary}' with {T} frames, each of size ({Y}, {X}), dtype {dtype}.")

# Example usage:
# tiff_to_suite2p_binary("my_movie.tif", "data_raw.bin")
```

This function:
- Loads the TIFF into memory.
- Creates a new binary file with the correct dimensions.
- Writes the data to disk, so that the resulting file is compatible with the Suite2p processing pipeline.

---

## Additional Utilities

- **`BinaryFile.convert_numpy_file_to_suite2p_binary(...)`**  
  This static method simplifies conversion from numpy files (e.g., `.npz` or pickled `.npy`) directly into Suite2p’s binary format.

- **Combined Binary File (BinaryFileCombined)**  
  In cases where multiple planes are combined into one dataset, `BinaryFileCombined` uses multiple `BinaryFile` objects and assembles the full image from several files. This class supports operations such as slicing across combined planes.

---

## References

- [Suite2p GitHub Repository](https://github.com/cortex-lab/Suite2p) – for overall pipeline usage and additional I/O utilities.
- Zarr and NumPy memmap documentation for details on data storage and efficient file I/O.

---

This markdown file serves as a comprehensive guide to understanding how `BinaryFile` is used across different parts of the Suite2p pipeline, from file conversion and writing to reading and processing during detection.
