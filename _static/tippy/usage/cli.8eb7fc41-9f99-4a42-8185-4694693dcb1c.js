selector_to_html = {"a[href=\"#download\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Download<a class=\"headerlink\" href=\"#download\" title=\"Link to this heading\">#</a></h2><p>Download files from github (auto-converts blob to raw urls).</p>", "a[href=\"#convert\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Convert<a class=\"headerlink\" href=\"#convert\" title=\"Link to this heading\">#</a></h2><p>Convert between formats with optional processing.</p>", "a[href=\"#cli-guide\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">CLI Guide<a class=\"headerlink\" href=\"#cli-guide\" title=\"Link to this heading\">#</a></h1><p>The <code class=\"docutils literal notranslate\"><span class=\"pre\">mbo</span></code> command provides tools for viewing, converting, and analyzing imaging data.</p>", "a[href=\"#utilities\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Utilities<a class=\"headerlink\" href=\"#utilities\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#formats\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Formats<a class=\"headerlink\" href=\"#formats\" title=\"Link to this heading\">#</a></h2><p>Show supported file formats:</p>", "a[href=\"#info\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Info<a class=\"headerlink\" href=\"#info\" title=\"Link to this heading\">#</a></h2><p>Display array information without loading data.</p>", "a[href=\"#gui-mode\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">GUI Mode<a class=\"headerlink\" href=\"#gui-mode\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#overview\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Overview<a class=\"headerlink\" href=\"#overview\" title=\"Link to this heading\">#</a></h2>"}
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
