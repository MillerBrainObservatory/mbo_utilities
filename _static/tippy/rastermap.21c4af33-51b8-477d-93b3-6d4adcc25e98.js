selector_to_html = {"a[href=\"#sorting-via-tsp-segment-shifting\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Sorting via TSP + Segment Shifting<a class=\"headerlink\" href=\"#sorting-via-tsp-segment-shifting\" title=\"Link to this heading\">#</a></h2><p><strong>Locality parameter <code class=\"docutils literal notranslate\"><span class=\"pre\">w</span></code></strong> controls:</p>", "a[href=\"#rastermap\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Rastermap<a class=\"headerlink\" href=\"#rastermap\" title=\"Link to this heading\">#</a></h1><h2>Clustering: <code class=\"docutils literal notranslate\"><span class=\"pre\">n_clusters=None</span></code> vs specified values<a class=\"headerlink\" href=\"#clustering-n-clusters-none-vs-specified-values\" title=\"Link to this heading\">#</a></h2><p><strong>Example:</strong></p>", "a[href=\"#embedding-and-upsampling\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Embedding and Upsampling<a class=\"headerlink\" href=\"#embedding-and-upsampling\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#references\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">References<a class=\"headerlink\" href=\"#references\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#clustering-n-clusters-none-vs-specified-values\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Clustering: <code class=\"docutils literal notranslate\"><span class=\"pre\">n_clusters=None</span></code> vs specified values<a class=\"headerlink\" href=\"#clustering-n-clusters-none-vs-specified-values\" title=\"Link to this heading\">#</a></h2><p><strong>Example:</strong></p>", "a[href=\"#summary-model-step-by-step\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Summary: Model Step by Step<a class=\"headerlink\" href=\"#summary-model-step-by-step\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#superneurons-and-binning\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Superneurons and Binning<a class=\"headerlink\" href=\"#superneurons-and-binning\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#final-outputs\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Final Outputs<a class=\"headerlink\" href=\"#final-outputs\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#clustering-step-details\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Clustering Step Details<a class=\"headerlink\" href=\"#clustering-step-details\" title=\"Link to this heading\">#</a></h2><p>This step compresses data from N neurons \u2192 k clusters and denoises by averaging.</p>"}
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
