selector_to_html = {"a[href=\"../api/visualization.html#mbo_utilities.run_gui\"]": "<dt class=\"sig sig-object py\" id=\"mbo_utilities.run_gui\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">mbo_utilities.</span></span><span class=\"sig-name descname\"><span class=\"pre\">run_gui</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">data_in</span></span><span class=\"p\"><span class=\"pre\">:</span></span><span class=\"w\"> </span><span class=\"n\"><span class=\"pre\">None</span><span class=\"w\"> </span><span class=\"p\"><span class=\"pre\">|</span></span><span class=\"w\"> </span><span class=\"pre\">str</span><span class=\"w\"> </span><span class=\"p\"><span class=\"pre\">|</span></span><span class=\"w\"> </span><span class=\"pre\">Path</span><span class=\"w\"> </span><span class=\"p\"><span class=\"pre\">|</span></span><span class=\"w\"> </span><span class=\"pre\">ScanMultiROIReordered</span><span class=\"w\"> </span><span class=\"p\"><span class=\"pre\">|</span></span><span class=\"w\"> </span><span class=\"pre\">ndarray</span></span><span class=\"w\"> </span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"w\"> </span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"o\"><span class=\"pre\">**</span></span><span class=\"n\"><span class=\"pre\">kwargs</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../_modules/mbo_utilities/graphics/run_gui.html#run_gui\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd><p>Open a GUI to preview data.</p></dd>", "a[href=\"#run-gui\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2. </span><code class=\"docutils literal notranslate\"><span class=\"pre\">run_gui</span></code><a class=\"headerlink\" href=\"#run-gui\" title=\"Link to this heading\">#</a></h1><p><a class=\"reference internal\" href=\"../api/visualization.html#mbo_utilities.run_gui\" title=\"mbo_utilities.run_gui\"><code class=\"xref py py-func docutils literal notranslate\"><span class=\"pre\">mbo_utilities.run_gui()</span></code></a> is a convenience function for viewing common data formats used in processing MBO datasets, usable from the terminal, jupyter-lab or a callable from a python script.</p>", "a[href=\"#jupyter-ipython\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.1. </span>Jupyter / IPython<a class=\"headerlink\" href=\"#jupyter-ipython\" title=\"Link to this heading\">#</a></h2><p>Returns a <code class=\"xref py py-class docutils literal notranslate\"><span class=\"pre\">fastplotlib.ImageWidget</span></code>:</p>", "a[href=\"#python-script-non-jupyter\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.2. </span>Python script (non-Jupyter)<a class=\"headerlink\" href=\"#python-script-non-jupyter\" title=\"Link to this heading\">#</a></h2><p>If run from a script without data_in, and Qt is installed, a file dialog will prompt for input.</p>"}
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
