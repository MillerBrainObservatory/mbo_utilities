selector_to_html = {"a[href=\"#vizualization\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2. </span>Vizualization<a class=\"headerlink\" href=\"#vizualization\" title=\"Link to this heading\">#</a></h1><p>Functions to help visualize datasets.</p>", "a[href=\"#mbo_utilities.graphics.run_gui\"]": "<dt class=\"sig sig-object py\" id=\"mbo_utilities.graphics.run_gui\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">mbo_utilities.graphics.</span></span><span class=\"sig-name descname\"><span class=\"pre\">run_gui</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"o\"><span class=\"pre\">*</span></span><span class=\"n\"><span class=\"pre\">args</span></span><span class=\"p\"><span class=\"pre\">:</span></span><span class=\"w\"> </span><span class=\"n\"><span class=\"pre\">t.Any</span></span></em>, <em class=\"sig-param\"><span class=\"o\"><span class=\"pre\">**</span></span><span class=\"n\"><span class=\"pre\">kwargs</span></span><span class=\"p\"><span class=\"pre\">:</span></span><span class=\"w\"> </span><span class=\"n\"><span class=\"pre\">t.Any</span></span></em><span class=\"sig-paren\">)</span> <span class=\"sig-return\"><span class=\"sig-return-icon\">\u2192</span> <span class=\"sig-return-typehint\"><span class=\"pre\">t.Any</span></span></span></dt><dd><p>Open a GUI to preview data of any supported type.</p></dd>"}
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
