selector_to_html = {"a[href=\"#writers\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">2. Writers<a class=\"headerlink\" href=\"#writers\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#zarr-i-o-metadata-audit\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Zarr I/O &amp; metadata audit<a class=\"headerlink\" href=\"#zarr-i-o-metadata-audit\" title=\"Link to this heading\">#</a></h1><p>Map of every function involved in zarr reading, writing, chunk/shard layout,\ncompression codecs, and OME-NGFF metadata across <code class=\"docutils literal notranslate\"><span class=\"pre\">mbo_utilities</span></code>.\nGenerated 2026-06-03.</p><p><code class=\"docutils literal notranslate\"><span class=\"pre\">Called</span> <span class=\"pre\">by</span></code> legend: <code class=\"docutils literal notranslate\"><span class=\"pre\">path:line</span></code> = concrete caller; <code class=\"docutils literal notranslate\"><span class=\"pre\">public</span> <span class=\"pre\">API</span></code> = exported /\ncalled by external code or the package surface; <code class=\"docutils literal notranslate\"><span class=\"pre\">tests</span> <span class=\"pre\">only</span></code> = exercised solely\nby <code class=\"docutils literal notranslate\"><span class=\"pre\">tests/</span></code>; <code class=\"docutils literal notranslate\"><span class=\"pre\">CLI</span> <span class=\"pre\">script</span> <span class=\"pre\">entry</span></code> = run as a standalone script.</p>", "a[href=\"#isoview-consolidation-arrays-isoview-consolidate-py\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">4. isoview consolidation (<code class=\"docutils literal notranslate\"><span class=\"pre\">arrays/isoview/consolidate.py</span></code>)<a class=\"headerlink\" href=\"#isoview-consolidation-arrays-isoview-consolidate-py\" title=\"Link to this heading\">#</a></h2><p>All rows are <code class=\"docutils literal notranslate\"><span class=\"pre\">mbo_utilities/arrays/isoview/consolidate.py</span></code>.</p>", "a[href=\"#files-checked-with-no-zarr-ome-functions\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Files checked with no zarr/OME functions<a class=\"headerlink\" href=\"#files-checked-with-no-zarr-ome-functions\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#entry-points-dispatch\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">1. Entry points &amp; dispatch<a class=\"headerlink\" href=\"#entry-points-dispatch\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#readers\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">3. Readers<a class=\"headerlink\" href=\"#readers\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#scripts-scripts\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">6. Scripts (<code class=\"docutils literal notranslate\"><span class=\"pre\">scripts/</span></code>)<a class=\"headerlink\" href=\"#scripts-scripts\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#metadata-pyramid-builders\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">5. Metadata &amp; pyramid builders<a class=\"headerlink\" href=\"#metadata-pyramid-builders\" title=\"Link to this heading\">#</a></h2>"}
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
