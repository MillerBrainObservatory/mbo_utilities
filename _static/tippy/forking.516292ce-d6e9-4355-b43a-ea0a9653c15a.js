selector_to_html = {"a[href=\"#priority-guide\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Priority guide<a class=\"headerlink\" href=\"#priority-guide\" title=\"Link to this heading\">#</a></h2><p><code class=\"docutils literal notranslate\"><span class=\"pre\">can_open()</span></code> is called on every candidate during dispatch \u2014 keep it cheap and\nexception-safe (a raised exception is treated as \u201cno\u201d).</p>", "a[href=\"#minimal-class\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Minimal class<a class=\"headerlink\" href=\"#minimal-class\" title=\"Link to this heading\">#</a></h2><p><code class=\"docutils literal notranslate\"><span class=\"pre\">shape</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">ndim</span></code> (== 5), <code class=\"docutils literal notranslate\"><span class=\"pre\">nt/nc/nz/ny/nx</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">squeeze()</span></code>, and the registry hooks\ncome from <code class=\"docutils literal notranslate\"><span class=\"pre\">LazyArray</span></code>. Everything else (reductions, frame rate, voxel size,\nROIs, phase correction) is opt-in via the mixins in <code class=\"docutils literal notranslate\"><span class=\"pre\">mbo_utilities.arrays</span></code> and\n<code class=\"docutils literal notranslate\"><span class=\"pre\">mbo_utilities.arrays.features</span></code> \u2014 none are required.</p>", "a[href=\"#adding-a-format-forking-the-lazyarray-api\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Adding a format (forking the LazyArray API)<a class=\"headerlink\" href=\"#adding-a-format-forking-the-lazyarray-api\" title=\"Link to this heading\">#</a></h1><p><code class=\"docutils literal notranslate\"><span class=\"pre\">imread()</span></code> dispatches to the highest-<code class=\"docutils literal notranslate\"><span class=\"pre\">PRIORITY</span></code> registered <code class=\"docutils literal notranslate\"><span class=\"pre\">LazyArray</span></code>\nsubclass whose <code class=\"docutils literal notranslate\"><span class=\"pre\">can_open(path)</span></code> returns <code class=\"docutils literal notranslate\"><span class=\"pre\">True</span></code>. A third-party package can add\nor override a format by shipping one class and registering it \u2014 no edits to\n<code class=\"docutils literal notranslate\"><span class=\"pre\">mbo_utilities</span></code>.</p>", "a[href=\"#register-it\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Register it<a class=\"headerlink\" href=\"#register-it\" title=\"Link to this heading\">#</a></h2><p>Either at runtime:</p>"}
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
