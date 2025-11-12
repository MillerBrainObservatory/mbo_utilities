selector_to_html = {"a[href=\"#suite2parray\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">Suite2pArray<a class=\"headerlink\" href=\"#suite2parray\" title=\"Link to this heading\">#</a></h3><p><strong>Returned when:</strong> Reading a directory containing <code class=\"docutils literal notranslate\"><span class=\"pre\">ops.npy</span></code></p>", "a[href=\"#overview\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Overview<a class=\"headerlink\" href=\"#overview\" title=\"Link to this heading\">#</a></h2><p><code class=\"docutils literal notranslate\"><span class=\"pre\">mbo_utilities.imread()</span></code> is a smart file reader that automatically detects the file type and returns the appropriate lazy array class. This guide explains what to expect when reading different file formats.</p>", "a[href=\"#h5array\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">H5Array<a class=\"headerlink\" href=\"#h5array\" title=\"Link to this heading\">#</a></h3><p><strong>Returned when:</strong> Reading HDF5 files</p>", "a[href=\"#tiffarray-mbotiffarray\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">TiffArray &amp; MBOTiffArray<a class=\"headerlink\" href=\"#tiffarray-mbotiffarray\" title=\"Link to this heading\">#</a></h3><p><strong>Returned when:</strong> Reading processed TIFF files (not raw ScanImage)</p>", "a[href=\"#decision-tree-what-will-imread-return\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Decision Tree: What Will imread() Return?<a class=\"headerlink\" href=\"#decision-tree-what-will-imread-return\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#lazy-array-types\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Lazy Array Types<a class=\"headerlink\" href=\"#lazy-array-types\" title=\"Link to this heading\">#</a></h1><p>Understanding what <code class=\"docutils literal notranslate\"><span class=\"pre\">imread()</span></code> returns and when to use each array type.</p>", "a[href=\"#binarray\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">BinArray<a class=\"headerlink\" href=\"#binarray\" title=\"Link to this heading\">#</a></h3><p><strong>Returned when:</strong> Explicitly reading a <code class=\"docutils literal notranslate\"><span class=\"pre\">.bin</span></code> file path</p>", "a[href=\"#array-type-details\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Array Type Details<a class=\"headerlink\" href=\"#array-type-details\" title=\"Link to this heading\">#</a></h2><h3>MboRawArray<a class=\"headerlink\" href=\"#mborawarray\" title=\"Link to this heading\">#</a></h3><p><strong>Returned when:</strong> Reading raw ScanImage TIFF files with multi-ROI metadata</p>", "a[href=\"#npyarray\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">NpyArray<a class=\"headerlink\" href=\"#npyarray\" title=\"Link to this heading\">#</a></h3><p><strong>Returned when:</strong> Reading <code class=\"docutils literal notranslate\"><span class=\"pre\">.npy</span></code> memory-mapped files</p>", "a[href=\"#quick-reference\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Quick Reference<a class=\"headerlink\" href=\"#quick-reference\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#zarrarray\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">ZarrArray<a class=\"headerlink\" href=\"#zarrarray\" title=\"Link to this heading\">#</a></h3><p><strong>Returned when:</strong> Reading Zarr stores</p>", "a[href=\"#mborawarray\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">MboRawArray<a class=\"headerlink\" href=\"#mborawarray\" title=\"Link to this heading\">#</a></h3><p><strong>Returned when:</strong> Reading raw ScanImage TIFF files with multi-ROI metadata</p>"}
skip_classes = ["headerlink", "sd-stretched-link"]

window.onload = function () {
    for (const [select, tip_html] of Object.entries(selector_to_html)) {
        const links = document.querySelectorAll(` ${select}`);
        for (const link of links) {
            if (skip_classes.some(c => link.classList.contains(c))) {
                continue;
            }

            tippy(link, {
                content: tip_html,
                allowHTML: true,
                arrow: true,
                placement: 'auto-start', maxWidth: 500, interactive: false,

            });
        };
    };
    console.log("tippy tips loaded!");
};
