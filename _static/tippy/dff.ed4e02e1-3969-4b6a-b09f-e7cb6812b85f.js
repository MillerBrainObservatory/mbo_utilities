selector_to_html = {"a[href=\"https://doi.org/10.3390/biom14010138\"]": "\n<div>\n    <h3>Deciphering the Calcium Code: A Review of Calcium Activity Analysis Methods Employed to Identify Meaningful Activity in Early Neural Development</h3>\n    \n    <p><b>Authors:</b> Sudip Paudel, Michelle Yue, Rithvik Nalamalapu, Margaret S. Saha</p>\n    \n    <p><b>Publisher:</b> MDPI AG</p>\n    <p><b>Published:</b> 2024-1-22</p>\n</div>", "a[href=\"#fig-dff-example\"]": "<figure class=\"align-default\" id=\"fig-dff-example\">\n<a class=\"reference internal image-reference\" href=\"_images/dff_1.png\"><img alt=\"Example \u0394F/F trace baseline comparisons\" src=\"_images/dff_1.png\" style=\"width: 600px;\"/>\n</a>\n<figcaption>\n<p><span class=\"caption-text\">Example of \u0394F/F trace showing different baseline choices. Adapted from Fig.\u00a01 of .</span><a class=\"headerlink\" href=\"#fig-dff-example\" title=\"Link to this image\">#</a></p>\n</figcaption>\n</figure>", "a[href=\"#calculating-df-f\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Calculating DF/F<a class=\"headerlink\" href=\"#calculating-df-f\" title=\"Link to this heading\">#</a></h1><p>This table summarizes Calcium Activity (\u0394F/F) detection methods reviewed by Paudel et al. <a class=\"reference external\" href=\"https://doi.org/10.3390/biom14010138\">(2024)</a>.</p>"}
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
