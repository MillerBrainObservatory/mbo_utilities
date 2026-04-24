selector_to_html = {"a[href=\"#as-a-library\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">As a library<a class=\"headerlink\" href=\"#as-a-library\" title=\"Link to this heading\">#</a></h3>", "a[href=\"#from-the-command-line\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">From the command line<a class=\"headerlink\" href=\"#from-the-command-line\" title=\"Link to this heading\">#</a></h3>", "a[href=\"#usage\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Usage<a class=\"headerlink\" href=\"#usage\" title=\"Link to this heading\">#</a></h2><h3>As a library<a class=\"headerlink\" href=\"#as-a-library\" title=\"Link to this heading\">#</a></h3>", "a[href=\"#fiber-activity-detection\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Fiber Activity Detection<a class=\"headerlink\" href=\"#fiber-activity-detection\" title=\"Link to this heading\">#</a></h1><p>The fiber activity pipeline detects and extracts fluorescence activity from neurite-like structures in timelapse calcium imaging data.\nIt is designed for single-plane timelapse stacks (2D + time) where the structures of interest are thin, elongated processes such as axons or dendrites rather than cell bodies.</p>", "a[href=\"#overview\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Overview<a class=\"headerlink\" href=\"#overview\" title=\"Link to this heading\">#</a></h2><p>The pipeline performs the following steps:</p>", "a[href=\"#parameters\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Parameters<a class=\"headerlink\" href=\"#parameters\" title=\"Link to this heading\">#</a></h2>"}
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
