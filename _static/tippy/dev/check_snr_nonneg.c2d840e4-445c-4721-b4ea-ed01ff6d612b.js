selector_to_html = {"a[href=\"#snr-verification\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">SNR verification<a class=\"headerlink\" href=\"#snr-verification\" title=\"Link to this heading\">#</a></h1><p>dataset: <code class=\"docutils literal notranslate\"><span class=\"pre\">32622_1_00002_00001.tif</span></code></p>"}
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
