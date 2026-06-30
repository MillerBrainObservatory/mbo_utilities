selector_to_html = {"a[href=\"#gui-mode\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">GUI Mode<a class=\"headerlink\" href=\"#gui-mode\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#init\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Init<a class=\"headerlink\" href=\"#init\" title=\"Link to this heading\">#</a></h2><p>Create starter notebooks (mbo + LBM-Suite2p user guides).</p>", "a[href=\"#convert\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Convert<a class=\"headerlink\" href=\"#convert\" title=\"Link to this heading\">#</a></h2><p>Convert between formats, optionally selecting planes/timepoints and applying phase correction.</p>", "a[href=\"#shortcut\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Shortcut<a class=\"headerlink\" href=\"#shortcut\" title=\"Link to this heading\">#</a></h2><p>Create a desktop icon that opens the GUI.</p>", "a[href=\"#command-line-interface\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Command Line Interface<a class=\"headerlink\" href=\"#command-line-interface\" title=\"Link to this heading\">#</a></h1><p>The <code class=\"docutils literal notranslate\"><span class=\"pre\">mbo</span></code> command provides tools for viewing, converting, and analyzing imaging data.</p>", "a[href=\"#gpu\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">GPU<a class=\"headerlink\" href=\"#gpu\" title=\"Link to this heading\">#</a></h2><p>Show which GPU renders the viewer and which runs compute (suite2p / cellpose / cupy), plus device memory.</p>", "a[href=\"#info\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Info<a class=\"headerlink\" href=\"#info\" title=\"Link to this heading\">#</a></h2><p>Display shape, dtype, imaging metadata, and any Suite2p results found alongside the data. Nothing is loaded into memory.</p>", "a[href=\"#utilities\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Utilities<a class=\"headerlink\" href=\"#utilities\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#formats\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Formats<a class=\"headerlink\" href=\"#formats\" title=\"Link to this heading\">#</a></h2><p><strong>Input:</strong> <code class=\"docutils literal notranslate\"><span class=\"pre\">.tif</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">.tiff</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">.zarr</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">.bin</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">.h5</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">.hdf5</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">.npy</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">.json</span></code>\n<strong>Output:</strong> <code class=\"docutils literal notranslate\"><span class=\"pre\">.tiff</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">.zarr</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">.bin</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">.h5</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">.npy</span></code></p>", "a[href=\"#upgrade\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Upgrade<a class=\"headerlink\" href=\"#upgrade\" title=\"Link to this heading\">#</a></h2>"}
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
