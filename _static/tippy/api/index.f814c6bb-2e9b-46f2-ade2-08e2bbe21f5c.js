selector_to_html = {"a[href=\"core.html#core-api\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">1. </span>Core<a class=\"headerlink\" href=\"#core\" title=\"Link to this heading\">#</a></h1><p>Functions central to data analysis on datasets collected at the MBO.</p>", "a[href=\"core.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">1. </span>Core<a class=\"headerlink\" href=\"#core\" title=\"Link to this heading\">#</a></h1><p>Functions central to data analysis on datasets collected at the MBO.</p>", "a[href=\"#api\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">API<a class=\"headerlink\" href=\"#api\" title=\"Link to this heading\">#</a></h1><p>Python API provides helper functions for saving, loading, processing and visualizing mbo datasets.</p>", "a[href=\"visualization.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">3. </span>Vizualization<a class=\"headerlink\" href=\"#vizualization\" title=\"Link to this heading\">#</a></h1><p>Functions to help visualize datasets.</p>", "a[href=\"visualization.html#viz-api\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">3. </span>Vizualization<a class=\"headerlink\" href=\"#vizualization\" title=\"Link to this heading\">#</a></h1><p>Functions to help visualize datasets.</p>", "a[href=\"io.html#io-api\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2. </span>I/O<a class=\"headerlink\" href=\"#i-o\" title=\"Link to this heading\">#</a></h1><p>Functions to help with loading and saving data.</p>", "a[href=\"io.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2. </span>I/O<a class=\"headerlink\" href=\"#i-o\" title=\"Link to this heading\">#</a></h1><p>Functions to help with loading and saving data.</p>"}
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
