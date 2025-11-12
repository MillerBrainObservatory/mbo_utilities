selector_to_html = {"a[href=\"#save-mp4\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">3. </span><code class=\"docutils literal notranslate\"><span class=\"pre\">save_mp4</span></code><a class=\"headerlink\" href=\"#save-mp4\" title=\"Link to this heading\">#</a></h1><p><code class=\"xref py py-func docutils literal notranslate\"><span class=\"pre\">mbo_utilities.save_mp4()</span></code></p><p><code class=\"docutils literal notranslate\"><span class=\"pre\">save_mp4</span></code> converts a 3D numpy array or TIFF stack (<code class=\"docutils literal notranslate\"><span class=\"pre\">[T,</span> <span class=\"pre\">Y,</span> <span class=\"pre\">X]</span></code>) into an <code class=\"docutils literal notranslate\"><span class=\"pre\">.mp4</span></code> video.</p>", "a[href=\"#load-data\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">3.1. </span>Load Data<a class=\"headerlink\" href=\"#load-data\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#example-usage\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">3.2. </span>Example Usage<a class=\"headerlink\" href=\"#example-usage\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#parameters\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">3.3. </span>Parameters<a class=\"headerlink\" href=\"#parameters\" title=\"Link to this heading\">#</a></h2>"}
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
