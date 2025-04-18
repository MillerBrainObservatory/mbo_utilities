selector_to_html = {"a[href=\"#id1\"]": "<figure class=\"align-center\" id=\"id1\">\n<img alt=\"_images/ex_diagram.png\" src=\"_images/ex_diagram.png\"/>\n<figcaption>\n<p><span class=\"caption-text\">Overview of pre-processing steps that convert the raw scanimage tiffs into planar timeseries.\nStarting with a raw, multi-page ScanImage Tiff, frames are deinterleaved, optionally pre-processed to eliminate scan-phase artifacts,\nand fused to create an assembled timeseries.</span><a class=\"headerlink\" href=\"#id1\" title=\"Link to this image\">#</a></p>\n</figcaption>\n</figure>", "a[href=\"#input-data-path-to-your-raw-tiff-file-s\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Input data: Path to your raw .tiff file(s)<a class=\"headerlink\" href=\"#input-data-path-to-your-raw-tiff-file-s\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#vizualize-data-with-fastplotlib\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Vizualize data with <a class=\"reference external\" href=\"https://www.fastplotlib.org/user_guide/guide.html#what-is-fastplotlib\">fastplotlib</a><a class=\"headerlink\" href=\"#vizualize-data-with-fastplotlib\" title=\"Link to this heading\">#</a></h2><p>To get a rough idea of the quality of your extracted timeseries, we can create a fastplotlib visualization to preview traces of individual pixels.</p><p>Here, we simply click on any pixel in the movie, and we get a 2D trace (or \u201ctemporal component\u201d as used in this field) of the pixel through the course of the movie:</p>", "a[href=\"#initialize-a-scanreader-object\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Initialize a scanreader object<a class=\"headerlink\" href=\"#initialize-a-scanreader-object\" title=\"Link to this heading\">#</a></h2><p>Pass a list of files, or a wildcard \u201c/path/to/files/*\u201d to <code class=\"docutils literal notranslate\"><span class=\"pre\">mbo.read_scan()</span></code>.</p>", "a[href=\"#overview\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Overview<a class=\"headerlink\" href=\"#overview\" title=\"Link to this heading\">#</a></h2><p>This section covers the steps to convert raw scanimage-tiff files into assembled, planar timeseries.</p>", "a[href=\"#accessing-data-in-the-scan\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Accessing data in the scan<a class=\"headerlink\" href=\"#accessing-data-in-the-scan\" title=\"Link to this heading\">#</a></h2><p>The scan can be indexed like a numpy array, data will be loaded lazily as only the data you access here is loaded in memory.</p>", "a[href=\"#save-assembled-files\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Save assembled files<a class=\"headerlink\" href=\"#save-assembled-files\" title=\"Link to this heading\">#</a></h2><p>The currently supported file extensions are <code class=\"docutils literal notranslate\"><span class=\"pre\">.tiff</span></code>.</p>", "a[href=\"#image-assembly\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Image Assembly<a class=\"headerlink\" href=\"#image-assembly\" title=\"Link to this heading\">#</a></h1><h2>Overview<a class=\"headerlink\" href=\"#overview\" title=\"Link to this heading\">#</a></h2><p>This section covers the steps to convert raw scanimage-tiff files into assembled, planar timeseries.</p>"}
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
