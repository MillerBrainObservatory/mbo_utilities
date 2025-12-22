selector_to_html = {"a[href=\"#command-line-interface\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Command Line Interface<a class=\"headerlink\" href=\"#command-line-interface\" title=\"Link to this heading\">#</a></h1><p>The <code class=\"docutils literal notranslate\"><span class=\"pre\">mbo</span></code> command provides tools for viewing, converting, and analyzing imaging data.</p>", "a[href=\"#info\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Info<a class=\"headerlink\" href=\"#info\" title=\"Link to this heading\">#</a></h2><p>Display array information without loading data.</p>", "a[href=\"#tips\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">Tips<a class=\"headerlink\" href=\"#tips\" title=\"Link to this heading\">#</a></h3>", "a[href=\"#download\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Download<a class=\"headerlink\" href=\"#download\" title=\"Link to this heading\">#</a></h2><p>Download files from github (auto-converts blob to raw urls).</p>", "a[href=\"#usage\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">Usage<a class=\"headerlink\" href=\"#usage\" title=\"Link to this heading\">#</a></h3>", "a[href=\"#convert\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Convert<a class=\"headerlink\" href=\"#convert\" title=\"Link to this heading\">#</a></h2><p>Convert between formats with optional processing.</p>", "a[href=\"#utilities\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Utilities<a class=\"headerlink\" href=\"#utilities\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#gui-mode\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">GUI Mode<a class=\"headerlink\" href=\"#gui-mode\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#commands\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Commands<a class=\"headerlink\" href=\"#commands\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#output-files\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">Output Files<a class=\"headerlink\" href=\"#output-files\" title=\"Link to this heading\">#</a></h3>", "a[href=\"#formats\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Formats<a class=\"headerlink\" href=\"#formats\" title=\"Link to this heading\">#</a></h2><p>Show supported file formats:</p>", "a[href=\"#scan-phase-analysis\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Scan-Phase Analysis<a class=\"headerlink\" href=\"#scan-phase-analysis\" title=\"Link to this heading\">#</a></h2><p>Bidirectional resonant scanning causes alternating rows to be shifted horizontally. This tool measures that shift to help configure correction parameters.</p>", "a[href=\"#interpreting-results\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">Interpreting Results<a class=\"headerlink\" href=\"#interpreting-results\" title=\"Link to this heading\">#</a></h3><p><strong>temporal.png</strong>: time series should be flat. large jumps indicate motion or hardware issues. typical offset is 0.5-2.0 px.</p><p><strong>windows.png</strong>: shows how estimate stabilizes with more frames. left plot: offset converges to stable value. right plot: variance decreases with window size. red line marks where std drops below 0.1 px. use this to determine how many frames to average for correction.</p>"}
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
