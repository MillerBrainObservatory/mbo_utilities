Optimally Loading Large ScanImage TIFF Files with Dask
Understanding the Data Layout in ScanImage TIFFs

ScanImage (especially in Light Beads Microscopy mode) produces multi-page BigTIFF files that encode multi-dimensional data. Important metadata (often provided separately or in the first TIFF page) includes:

    Number of fields (ROIs) – ScanImage’s multi-ROI acquisitions can save multiple distinct imaging regions in one file.

    Number of channels – In Light Beads Microscopy, the “channels” actually correspond to Z-stack planes (physical Z positions), since the system acquires multiple depths per timepoint and labels them as separate channels​
    millerbrainobservatory.github.io
    . (No actual Z coordinate is stored – just an index.)

    Number of frames – The number of time points (volumetric frames) in the time series.

    Field dimensions – The pixel width and height of each field (usually all fields share the same X×Y size).

Each 2D image in the TIFF is one page. ScanImage interleaves pages by Z-plane and time. In other words, the file’s page sequence goes: time0–plane0, time0–plane1, ..., time0–planeN, time1–plane0, time1–plane1, ... and so on​
millerbrainobservatory.github.io
. If there are multiple fields, typically the pages are grouped by time and field as well (e.g. for each timepoint, all planes of field1, then all planes of field2, etc., assuming the acquisition cycles through fields each frame). Thus, the total page count = fields × frames × channels. We can map a given 4D index (field f, frame t, plane z, Y, X) to a specific TIFF page index. For example, if pages are ordered by time→field→z, the page index = t * (fields*channels) + f * (channels) + z. This mapping is known from the metadata and will be used to locate data in the file.

Why is this layout an issue? Standard TIFF readers will treat a multi-page file as a simple stack of images and often scan every page’s header on open. For large ScanImage files (potentially hundreds of thousands of pages), a naive open can be extremely slow and memory-heavy​
github.com
. Moreover, because Z planes are mislabeled as channels and no explicit Z dimension is in the metadata, we must interpret the dimensions correctly in our loading code. The goal is to leverage the known structure to load data lazily (on demand) with Dask, avoiding an expensive upfront read of all page indices.
Bypassing the Slow TIFF Index Scan

When opening a TIFF, libraries like tifffile normally iterate through every Image File Directory (IFD) to gather metadata (e.g. image shape, offsets). This is what we want to skip. As the tifffile author notes, the “fastij” shortcut (reading only first page metadata) works only for ImageJ TIFFs where all necessary info is in page 0, which is not the case for general ScanImage files​
github.com
. In a standard TIFF there isn’t an easy way to get the total page count or all offsets without reading the IFD chain sequentially​
github.com
. However, we can exploit our a priori knowledge of the file structure:

    Known page count and layout: Since we know fields, frames, channels, we know exactly how many pages to expect. We don’t need to iterate to count pages; we can trust the provided metadata (e.g. if metadata says 4 fields, 1730 frames, 30 channels, then there are 4×1730×30 pages). This saves us from calling len(TiffFile.pages) (which would force a scan).

    Offsets of image data: We also know each image’s dimensions and data type (e.g. 512×512 pixels, 16-bit). If the images are stored uncompressed (ScanImage uses uncompressed data in most cases​
    vidriotech.gitlab.io
    ), the byte size of each page is fixed (e.g. 512×512×2 bytes = 524,288 bytes per page). This means page data likely occupies predictable segments of the file. In many ScanImage files, image data for each page is stored contiguously, and large frame-invariant metadata is stored in a separate block to avoid per-page overhead​
    vidriotech.gitlab.io
    ​
    vidriotech.gitlab.io
    . We can therefore calculate or quickly look up where each page’s pixel data begins.

Approach to skip the full scan: We can open the file in a way that reads only minimal info (just the first page or the metadata block) and avoids iterating all pages. For example, using tifffile we can do TiffFile(path, pages=[0]) to read only the first page header​
github.com
. From the first page, we get image shape, dtype, and perhaps a template for each page’s tag layout. We also obtain the offset of the first image data. Because all pages are identical in size, we can infer subsequent image offsets by a simple formula if they are back-to-back. (In practice, TIFF IFDs themselves are scattered in the file, so a safer method is to let tifffile or another tool gather offsets without reading image data.)

Modern versions of tifffile have an optimized way to access image data lazily: it can expose the TIFF as a Zarr store. This is done via TiffFile.aszarr(). We can pass chunkmode='page' to this function, which tells tifffile to treat each whole page as a chunk in a Zarr array​
github.com
. Internally, this builds a mapping of page offsets and byte counts into a Zarr-compatible interface​
github.com
. Crucially, it does so without reading all the pixel data upfront. There may still be a one-time read of all IFDs to get those offsets, but this is implemented in C (and with optional lazy tag loading) for speed, and it avoids storing large metadata per page. In tests, using chunkmode="page" drastically reduced the overhead of random access reads (e.g. ~90% speedup in some scenarios)​
github.com
.

Summary: To bypass the slow scan, either:

    Use tifffile’s optimized Zarr interface to get direct access to pages by index (preferred), or

    Manually compute offsets by reading just what’s needed (the first IFD and then jumping through the file). Given the complexity of TIFF, we lean on tifffile’s built-in methods rather than writing a custom TIFF parser.

Also note, the official ScanImage TIFF C++ reader library can retrieve the whole dataset quickly, but it only provides the entire volume at once (no random sub-volume reads)​
vidriotech.gitlab.io
, which doesn’t meet our lazy-loading requirement. Therefore, we proceed with a Python/Dask solution.
Constructing a Lazy Dask Array for the TIFF

We will create a Dask array that represents the 5D data (fields, frames, channels [Z], Y, X) and loads pixels on demand. There are a few ways to do this:

1. Using tifffile + Zarr + Dask: This is the most straightforward way. Tifffile’s Zarr mechanism will give us a Zarr array-like object that Dask can wrap.

    Open the TIFF and get a Zarr store:

import tifffile, dask.array as da
with tifffile.TiffFile("scanimage_file.tif") as tif:
    zarr_array = tif.aszarr(series=0, chunkmode="page")  # each TIFF page is one chunk
dask_arr = da.from_zarr(zarr_array)
print(dask_arr.shape, dask_arr.chunksize)

At this point, dask_arr is a lazy 3D stack of shape (total_pages, Y, X). Each chunk is one page (the chunkmode='page' ensured that)​
github.com
. No actual image data is read yet. We then reshape this array to separate the dimensions:

    dask_arr = dask_arr.reshape((num_fields, num_frames, num_channels, Y, X))
    dask_arr = dask_arr.rechunk((1, 1, 1, Y, X))  # ensure chunking aligns with the 5 dims

    Now dask_arr has shape (F, T, C, Y, X), and still each chunk is effectively one image plane. Dask knows how to index into this array and will ask tifffile for the correct page when needed. The advantage here is that tifffile manages all the offsets and I/O. When a chunk is actually computed, tifffile will seek to the byte offset of that page and read it. This method works on Linux or Windows (pure Python I/O) and avoids us explicitly handling file pointers.

    Memory considerations: The Zarr interface does not memory-map the whole file; it reads chunks on demand via Python file handle. This is good for cross-platform compatibility. (Memory-mapping a whole BigTIFF could be problematic, especially on Windows, and isn’t directly supported unless the file were formatted as an ImageJ hyperstack​
    github.com
    , which ScanImage is not.) Each chunk read will allocate a NumPy array of the page’s size (e.g. a 512×512 uint16 array = 0.5 MB). Dask will garbage collect chunks as they are no longer needed, preventing memory blow-up as long as you’re not caching huge portions.

2. Using Dask Delayed with a custom read function: This approach gives more control and can incorporate reading multiple pages at once if desired. We create a Python function to read a specific sub-block of the TIFF (for example, one plane or one volume), wrap it with dask.delayed, and then assemble a Dask array from many delayed calls.

    Define a reader function that uses tifffile (or another TIFF reader) to load the needed part. For instance, a simple function to read one image plane by index:

import tifffile, numpy as np
def read_plane(field, frame, channel):
    index = (frame * num_fields + field) * num_channels + channel  # compute page index
    with tifffile.TiffFile("scanimage_file.tif") as tif:
        arr = tif.pages[index].asarray()  # read the single page into a numpy array
    return arr  # shape (Y, X), dtype e.g. uint16

We use tifffile.TiffFile.pages[index].asarray() to fetch the page. This will internally seek to that page’s offset and read it. We open the file inside the function to ensure each Dask task opens and closes it independently (avoiding issues with file handles across processes). Because we do not iterate over all pages in that call (we index directly to one page), tifffile will still have to traverse the IFD chain to reach that page. To mitigate that, we could cache the offsets in a global structure. For example, we might first do a one-time parse of all page offsets (using tifffile or our own method) and store them in a list accessible to read_plane. Then read_plane could seek directly to the byte offset using Python’s open().seek(). This extra optimization avoids even the per-call IFD traversal. (Tifffile’s internal implementation might cache IFDs after one pass; however, since each with TiffFile(...) is a fresh open, caching is lost. An alternative is to keep a tif = TiffFile(path) open globally and use tif.filehandle to seek, but that has to be managed carefully in Dask workers.)

Wrap with dask.delayed and build the Dask array:
We create delayed calls for each needed chunk. If we choose each chunk = one image plane (smallest unit), we’ll make F*T*C delayed objects, which could be a lot. We can instead group channels or other dims to reduce the task count (more on chunking strategy later). For illustration, let’s do one plane per chunk:

    import dask
    lazy_reader = dask.delayed(read_plane)
    delayed_planes = [
        lazy_reader(f, t, c) 
        for f in range(num_fields) 
        for t in range(num_frames) 
        for c in range(num_channels)
    ]
    # Now delayed_planes is a list of delayed objects, each representing a (Y,X) numpy array.
    # Convert each to a tiny Dask array, then stack:
    import dask.array as da
    sample_shape = (Y, X)
    sample_dtype = np.uint16  # or infer from first page
    dask_planes = [da.from_delayed(d, shape=sample_shape, dtype=sample_dtype) 
                   for d in delayed_planes]
    stacked = da.stack(dask_planes).reshape((num_fields, num_frames, num_channels, Y, X))

    This uses the same pattern as in the Napari lazy loading tutorial (except that was stacking multiple files)​
    napari.org
    . We take each delayed result and make it a Dask array block, then combine them. The outcome stacked is equivalent to our 5D array. No TIFF data is read until a .compute() or other operation triggers those particular index(s). This approach is flexible: we could modify read_plane to read a batch of planes in one go (for example, read a whole Z-stack for one field & frame). Then each delayed task returns a 3D array (channels×Y×X) and we would stack those appropriately. Reading a batch of planes sequentially from disk can be more efficient than reading them one-by-one in separate tasks, at the cost of a chunk containing more data than maybe needed for a small slice request.

    Pros and cons: The pure dask.delayed approach avoids any up-front processing of the entire file (each task looks up what it needs when it runs). However, creating a huge list of delayed tasks can incur some scheduler overhead if F*T*C is very large. For example, 200,000 tasks is feasible for Dask but not trivial. Grouping into bigger chunks (e.g. one task per time frame, reading all Z for that frame) reduces the task count dramatically (in our example, 207k planes vs. 6.9k volumes). On the flip side, larger tasks mean less granularity in lazy loading. We must balance task overhead with I/O efficiency.

3. Implementing a custom Dask array class (advanced): Dask allows wrapping an object that implements the NumPy array interface. One could create a class ScanImageArray that knows how to retrieve any requested slice from the TIFF. For example, it might implement __getitem__ so that arr[..., 100:200, 100:200] reads just that region from disk. We would then do dask.array.from_array(ScanImageArray(...), chunks=...) to make it lazy. This technique could potentially allow truly arbitrary sub-indexing with minimal reads. However, it requires careful coding: the class needs to handle multi-dimensional indexing logic and perform file seeks/reads for possibly partial regions. Given the complexity (especially if reading a sub-region of a TIFF page, which may require reading full strips as explained below), this is only worth it if the simpler methods above prove insufficient. In most cases, using tifffile’s optimized reading or a delayed task per chunk is easier to maintain.

Choosing an approach: For most users, Approach 1 (tifffile + Dask) is recommended due to its simplicity and use of well-tested library code. Approach 2 is useful if you need custom reading logic (e.g., reading around a specific ROI or doing preprocessing per chunk). Both can achieve the same performance if tuned properly. The key is to define an appropriate chunking scheme for the Dask array.
Optimizing Chunking and Slicing for Performance

Choosing Dask chunk sizes is critical for IO performance and memory usage. Here we consider how to chunk across each dimension and the trade-offs:

    Chunking over fields: In many cases, it makes sense to keep fields as a chunk dimension of size 1 (i.e., treat each field independently). Different fields might be stored far apart in the file (if the TIFF stored all frames of field1 followed by all frames of field2, then reading across fields is non-contiguous). Moreover, analysis might often focus on one field at a time. So we typically use chunks of size 1 in the field dimension.

    Chunking over frames (time): If temporal access is random (accessing arbitrary timepoints), a chunk size of 1 frame is simplest. However, reading one frame’s worth of data could involve multiple planes. It might be efficient to chunk as one frame (containing all its Z planes) per chunk, because on disk all planes of a given time might be grouped closely. For example, if the file stores frame0-plane0...planeN, then frame1-plane0...planeN, etc., then all planes of frame t are in one relatively contiguous block. Reading them in one go is a sequential read of a large block, which is faster than many tiny reads. So a chunk shape like (1 frame, all channels, full Y, full X) is a good candidate. This means each Dask chunk is a 3D volume (C×Y×X) for one field & time. The chunk size in bytes would be num_channels * Y * X * bytes_per_pixel. For example, 30 channels of 512×512×2-byte images ≈ 15 MB chunk. This is reasonable. If that’s too large, we could split it (e.g. 5 channels per chunk -> 6 chunks per frame).

    Chunking over channels (Z slices): If users frequently access or process one Z-plane at a time (e.g., looking at individual planes), having each chunk be a single plane might be more efficient. This was essentially the chunkmode="page" approach – chunks of shape (1, 1, 1, Y, X) after we reshaped to (F,T,C,Y,X). The advantage is maximum flexibility: any single plane can be pulled with minimal IO (only that page is read). The disadvantage is a larger number of chunks/tasks and potentially more disk seeking if many planes are read in sequence. In practice, one can start with single-plane chunks (which is what our lazy reader examples above naturally produce) and rely on the filesystem cache or Dask prefetching to mitigate overhead if reading sequential planes. Dask’s task scheduler can even read multiple planes in parallel (which on an SSD or RAID can speed up throughput, but on an HDD might cause more seeking – tuning may be needed).

    Chunking in Y, X (spatial slicing): By default, we treat each image plane as a single chunk in the spatial dimensions (Y and X). If the images are large and users may request small subregions (e.g. a small ROI within the plane), one might consider breaking each plane into sub-image tiles as chunks (e.g., chunk of 256×256 pixels). Dask would then only load the needed tiles. However, TIFF is not optimized for reading arbitrary tiles unless it was written in a tiled format. ScanImage by default uses strips (rows or few-rows blocks) for storage. That means the image data for one plane is stored as strips of bytes (each strip maybe one row or a set of rows)​
    github.com
    . To read an arbitrary rectangle ROI, you might still have to read full strips overlapping that ROI. For example, if RowsPerStrip = 512 (the whole image height), then the entire plane is one strip – you cannot read just a 100×100 sub-block without reading the whole 512×512 (since it’s one contiguous block). If RowsPerStrip = 1 (each row separate), you could seek to each row of the ROI and read that row segment, but that results in 100 small reads for a 100×100 square – which is slow on a spinning disk. In short, unless the TIFF was written with small tiles, spatial sub-chunking doesn’t save much disk IO; it just slices the array after reading. Because the question assumes we know the layout, we can decide accordingly. Typically, ScanImage BigTIFFs are not tiled (and often not compressed)​
    vidriotech.gitlab.io
    , so we prefer to leave Y and X as a single chunk for contiguous reads. If truly needed, one could convert the TIFF to a zarr or HDF5 dataset with tile chunks (as noted by the Napari documentation, chunked formats like Zarr are “superior in many ways” for random subsets​
    napari.org
    ). But that involves data duplication or conversion, which we assume we want to avoid here.

Sequential vs Random Access considerations: If your analysis will ultimately read the entire dataset, you want to maximize sequential throughput (larger contiguous chunks). If you only ever need small subsets (like one timepoint, or one plane out of many), smaller chunks are better. It’s a trade-off:

    Sequential reading: e.g. reading a full frame (all Z) at once means one chunk = one ~20 MB read, very efficient sequentially. To get a time series of one pixel, though, you’d load a lot of unnecessary data per frame.

    Random access: e.g. chunk = single plane (0.5 MB) means minimal over-read for that plane, but reading a whole volume of 30 planes means 30 separate reads (which might be fine if the OS/disk can handle it in parallel or if cached). Randomly jumping between far-apart frames might defeat read-ahead caching; in that case each chunk read is truly random. On an SSD this is okay; on an HDD, you might want to ensure Dask reads in a somewhat sorted order or increase chunk size to reduce seeks.

Dask’s lazy evaluation helps because it will only load chunks that you actually index or compute on. So you might choose fine-grained chunks (one plane each) to be safe, and in practice if you do need a whole volume, you can request that volume slice and Dask will pull all relevant planes anyway. If performance is an issue, you could then rechunk the Dask array to combine planes into bigger chunks after initial loading.

To visualize the effect of chunking, consider the Dask array representation below, where each small cube is a chunk (in this example, 1200 total frames split into 1200 chunks of a volume)​

. More chunks (smaller pieces) mean more flexibility but also more overhead (3600 tasks in that example). Fewer, larger chunks mean fewer tasks but less granularity.

In our context, a reasonable default is chunk by single pages (the natural output of tifffile’s chunkmode='page') and see if it meets performance needs. Each chunk read will be a contiguous read of ~Y×X pixels. If needed, we can adjust:

    For faster sequential volume reading: chunk the channel dimension to include all (or many) Z per chunk.

    For smaller memory footprint when accessing single planes: keep chunk=1 plane.

    For extremely large Y,X with frequent small ROI access: consider tile-chunking (but weigh the cost as discussed).

Additional Notes on Implementation and Compatibility

The method described uses only Python libraries (tifffile, Dask, NumPy), which are cross-platform. Both Linux and Windows will handle this similarly. A few practical tips:

    File access on Windows: Make sure to open files in binary mode ('rb'). Tifffile does this internally. Closing the TiffFile after use (with the context manager or tif.close()) is important, especially on Windows, to avoid locking the file. In a Dask distributed context, each worker process will open its own handle – this is generally fine.

    Ensure single-threaded reading if using HDD: If you use delayed tasks for many small chunks on an HDD, you might want to limit the number of concurrent reads (to avoid excessive seeking). This can be done by configuring the Dask scheduler or using dask.local with a single-thread pool for the IO part. On SSD/NVMe, concurrency is less of an issue.

    Memory usage: Dask will load chunks into memory when needed. If you request a large slice (e.g. an entire 3D volume), it may load many chunks at once. Ensure your chunk size and Dask cluster memory are chosen accordingly. If memory is tight, prefer smaller chunks and only compute a few at a time.

    Testing: It’s wise to test the lazy loading on a smaller example first. For instance, try reading one page, one volume, etc., and measure the timing to ensure the overhead is indeed low. Check that slicing dask_arr[0, 0, 0, :100, :100] only triggers reading of that one chunk (you can watch disk activity or use dask.diagnostics.ProgressBar).

Finally, be aware that tifffile’s performance can depend on TIFF-specific details like RowsPerStrip. In benchmarks, reading whole-page chunks was faster when the TIFF stored image data in large strips​
github.com
. Since ScanImage does not use tiny strips, we benefit from that. If in the future compression or tiling is used, the strategy might need revisiting (for example, with compressed data, you cannot read partial chunks without decompressing the whole strip/tile).

In summary, the optimal design is:

    Leverage known metadata to avoid scanning: directly construct the Dask array’s shape from those metadata.

    Use tifffile’s aszarr(chunkmode='page') or a dask.delayed reader to lazily load each page.

    Choose Dask chunks aligning with the TIFF’s layout (e.g. one page per chunk, or one time-point per chunk) for efficient sequential disk reads, while still enabling random access to individual elements.

    Explain to users of this loading function the expected chunk structure so they can slice in ways that minimize unnecessary IO (for example, slicing all channels of a frame vs. slicing one channel across all frames will have different IO patterns).

By following this design, you get a lazy-loading Dask array that can be indexed in [field, frame, z, y, x] fashion, loading only the needed parts of the TIFF on demand. This provides interactive-speed access to huge ScanImage datasets, without the long initial load time that a full TIFF scan or read would entail. All of this works seamlessly on both Linux and Windows, since it relies on cross-platform file reading and Dask’s scheduler (the ScanImage team even provides a reader that supports all these platforms, indicating no inherent OS limitation aside from file system performance)​
vidriotech.gitlab.io
.
Trade-offs Summary (I/O, Memory, Chunking)

    I/O – Sequential vs Random: Large contiguous reads (e.g. reading an entire frame’s data sequentially) maximize disk throughput​
    github.com
    . Randomly reading small chunks (e.g. one 512×512 plane out of a huge stack) avoids reading unneeded data but can lead to many seeks. Our approach lets you do either. If your access pattern is known, tailor the chunking: e.g. chunk by frame for mostly sequential time series processing, or by plane for random access to slices.

    Memory: Lazy loading ensures you only hold the chunks you need in RAM. A single 16-bit plane ~0.5 MB; a full volume of 30 planes ~15 MB. Dask can work out-of-core, but avoid requesting too large a slice that loads massive data at once. If using approach 2 with delayed, note that holding hundreds of thousands of tiny task objects has some memory cost on the scheduler; grouping into moderate chunk sizes can alleviate this.

    Dask overhead vs simplicity: Finer chunks = more Dask task overhead (the scheduler must manage more tasks). There is a sweet spot where chunk size is a few megabytes to a few tens of MB – this tends to give good performance​
    napari.org
    . In our case each plane is already on that order. If your file has millions of pages, you might even group 5–10 planes per chunk to keep task count manageable. On the other hand, if you only have, say, 1000 pages, chunking each one is fine.

    Cross-platform considerations: There is essentially no difference in our method between Windows and Linux, aside from filesystem performance. We rely on Python I/O, which works the same on both. Just be mindful of file path conventions and the fact that Windows may not allow deleting the TIFF while it’s open (if that matters at all). Our lazy loader does not use any OS-specific calls.

By carefully designing the Dask array construction as above, you achieve efficient lazy loading of big ScanImage TIFF stacks. You skip the costly upfront scan of thousands of TIFF pages, yet retain the ability to slice along any dimension (field, time, Z/channel, or space) with on-demand reading. This design provides an optimal balance for interactive and large-scale processing of volumetric microscopy data​
napari.org
​
napari.org
. The trade-offs can be tuned by adjusting chunk sizes, but the outlined approach should serve as a solid foundation.

Sources: The solution builds on insights from tifffile’s documentation and discussions​
github.com
​
github.com
, ScanImage’s TIFF format notes​
millerbrainobservatory.github.io
​
millerbrainobservatory.github.io
, and best practices in Dask for lazy image loading​
napari.org

