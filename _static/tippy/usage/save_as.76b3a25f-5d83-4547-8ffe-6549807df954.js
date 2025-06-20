selector_to_html = {"a[href=\"#terminal-usage-cli\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">1.1. </span>Terminal usage (CLI)<a class=\"headerlink\" href=\"#terminal-usage-cli\" title=\"Link to this heading\">#</a></h2><p>When no <code class=\"docutils literal notranslate\"><span class=\"pre\">--save</span></code> path is given, only metadata is printed.</p>", "a[href=\"#save-as\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">1. </span><code class=\"docutils literal notranslate\"><span class=\"pre\">save_as</span></code><a class=\"headerlink\" href=\"#save-as\" title=\"Link to this heading\">#</a></h1><p><code class=\"xref py py-func docutils literal notranslate\"><span class=\"pre\">mbo_utilities.save_as()</span></code> is a convenience function for exporting common data formats processed from MBO datasets.</p><p>It can save <strong>ScanImage tiffs</strong> or <strong>ScanMultiROIReordered arrays</strong> into:</p>"}
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
