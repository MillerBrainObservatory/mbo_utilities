selector_to_html = {"a[href=\"#example-2-speed-factor-10x-faster\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Example 2: Speed Factor (10x Faster)<a class=\"headerlink\" href=\"#example-2-speed-factor-10x-faster\" title=\"Link to this heading\">#</a></h2><p>Use <code class=\"docutils literal notranslate\"><span class=\"pre\">speed_factor</span></code> to create fast previews for checking cell stability:</p>", "a[href=\"#summary-generated-files\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Summary: Generated Files<a class=\"headerlink\" href=\"#summary-generated-files\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#example-3-enhanced-quality\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Example 3: Enhanced Quality<a class=\"headerlink\" href=\"#example-3-enhanced-quality\" title=\"Link to this heading\">#</a></h2><p>Use temporal smoothing and gamma correction for cleaner videos:</p>", "a[href=\"#example-5-4d-data-select-z-plane\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Example 5: 4D Data (Select Z-Plane)<a class=\"headerlink\" href=\"#example-5-4d-data-select-z-plane\" title=\"Link to this heading\">#</a></h2><p>For 4D arrays <code class=\"docutils literal notranslate\"><span class=\"pre\">(T,</span> <span class=\"pre\">Z,</span> <span class=\"pre\">Y,</span> <span class=\"pre\">X)</span></code>, use <code class=\"docutils literal notranslate\"><span class=\"pre\">plane=</span></code> to select which z-plane to export:</p>", "a[href=\"#example-6-processing-video\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Example 6: Processing \u2192 Video<a class=\"headerlink\" href=\"#example-6-processing-video\" title=\"Link to this heading\">#</a></h2><p>Load data, apply processing, convert to numpy, then export:</p>", "a[href=\"#example-4-with-colormap\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Example 4: With Colormap<a class=\"headerlink\" href=\"#example-4-with-colormap\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#load-demo-data\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Load Demo Data<a class=\"headerlink\" href=\"#load-demo-data\" title=\"Link to this heading\">#</a></h2><p>Load raw ScanImage TIFFs with <code class=\"docutils literal notranslate\"><span class=\"pre\">mbo.imread()</span></code>:</p>", "a[href=\"#example-1-basic-video-export\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Example 1: Basic Video Export<a class=\"headerlink\" href=\"#example-1-basic-video-export\" title=\"Link to this heading\">#</a></h2><p>Export a single z-plane to video with default settings:</p>", "a[href=\"#to-video-examples\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">to_video Examples<a class=\"headerlink\" href=\"#to-video-examples\" title=\"Link to this heading\">#</a></h1><p>Export calcium imaging data to video files for presentations and sharing.</p><p>Output images/videos saved to <code class=\"docutils literal notranslate\"><span class=\"pre\">docs/_images/to_video/</span></code> for documentation.</p>"}
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
