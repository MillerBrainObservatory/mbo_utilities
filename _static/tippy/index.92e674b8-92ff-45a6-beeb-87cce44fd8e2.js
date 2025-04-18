selector_to_html = {"a[href=\"assembly.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Image Assembly<a class=\"headerlink\" href=\"#image-assembly\" title=\"Link to this heading\">#</a></h1><h2>Overview<a class=\"headerlink\" href=\"#overview\" title=\"Link to this heading\">#</a></h2><p>This section covers the steps to convert raw scanimage-tiff files into assembled, planar timeseries.</p>", "a[href=\"#miller-brain-observatory-python-utilities\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Miller Brain Observatory: Python Utilities<a class=\"headerlink\" href=\"#miller-brain-observatory-python-utilities\" title=\"Link to this heading\">#</a></h1><p>This repository contains python functions to pre/post process datasets recording at the <a class=\"reference external\" href=\"https://mbo.rockefeller.edu\">Miller Brain Observatory</a></p>", "a[href=\"api/index.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">API<a class=\"headerlink\" href=\"#api\" title=\"Link to this heading\">#</a></h1><p>Python API provides helper functions for saving, loading, processing and visualizing mbo datasets.</p>", "a[href=\"#overview\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Overview<a class=\"headerlink\" href=\"#overview\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#useful\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Useful<a class=\"headerlink\" href=\"#useful\" title=\"Link to this heading\">#</a></h2><p><a class=\"reference external\" href=\"https://www.saaspegasus.com/guides/uv-deep-dive/#cheatsheet-common-operations-in-uvs-workflows\">uv-cheatsheet</a></p>"}
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
