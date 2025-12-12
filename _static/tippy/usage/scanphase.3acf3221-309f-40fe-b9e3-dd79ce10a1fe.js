selector_to_html = {"a[href=\"#spatial-png\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">spatial.png<a class=\"headerlink\" href=\"#spatial-png\" title=\"Link to this heading\">#</a></h3><p>heatmaps show variation across fov. edges different from center is normal. gray = low signal (unreliable).</p>", "a[href=\"#scan-phase-analysis\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Scan-Phase Analysis<a class=\"headerlink\" href=\"#scan-phase-analysis\" title=\"Link to this heading\">#</a></h1><p>Bidirectional resonant scanning causes alternating rows to be shifted horizontally. This tool measures that shift to help configure correction parameters.</p>", "a[href=\"#windows-png\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">windows.png<a class=\"headerlink\" href=\"#windows-png\" title=\"Link to this heading\">#</a></h3><p>shows how estimate stabilizes with more frames. left plot: offset converges to stable value. right plot: variance decreases with window size. red line marks where std drops below 0.1 px.</p><p>use this to determine how many frames to average for correction.</p>", "a[href=\"#running\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Running<a class=\"headerlink\" href=\"#running\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#zplanes-png\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">zplanes.png<a class=\"headerlink\" href=\"#zplanes-png\" title=\"Link to this heading\">#</a></h3><p>assess if offset varies with depth, owing to the angle on the resonant scanner</p>", "a[href=\"#parameters-png\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">parameters.png<a class=\"headerlink\" href=\"#parameters-png\" title=\"Link to this heading\">#</a></h3><p>shows offset reliability vs signal. low signal = unreliable (high/variable offset). red line suggests intensity threshold below which measurements are noisy.</p>", "a[href=\"#tips\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Tips<a class=\"headerlink\" href=\"#tips\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#temporal-png\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">temporal.png<a class=\"headerlink\" href=\"#temporal-png\" title=\"Link to this heading\">#</a></h3><p>time series should be flat. large jumps indicate motion or hardware issues. typical offset is 0.5-2.0 px.</p>", "a[href=\"#output-files\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Output Files<a class=\"headerlink\" href=\"#output-files\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#what-to-look-for\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">What to Look For<a class=\"headerlink\" href=\"#what-to-look-for\" title=\"Link to this heading\">#</a></h2><h3>temporal.png<a class=\"headerlink\" href=\"#temporal-png\" title=\"Link to this heading\">#</a></h3><p>time series should be flat. large jumps indicate motion or hardware issues. typical offset is 0.5-2.0 px.</p>", "a[href=\"#what-it-does\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">What It Does<a class=\"headerlink\" href=\"#what-it-does\" title=\"Link to this heading\">#</a></h2>"}
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
