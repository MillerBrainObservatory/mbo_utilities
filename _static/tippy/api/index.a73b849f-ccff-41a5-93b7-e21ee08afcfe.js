selector_to_html = {"a[href=\"usage.html#usage-run-gui\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">4.1. </span><code class=\"docutils literal notranslate\"><span class=\"pre\">run_gui</span></code><a class=\"headerlink\" href=\"#usage-run-gui\" title=\"Link to this heading\">#</a></h2><p><code class=\"xref py py-func docutils literal notranslate\"><span class=\"pre\">mbo_utilities.run_gui()</span></code> opens an interactive viewer for imaging data using <a class=\"reference external\" href=\"https://www.fastplotlib.org/user_guide/guide.html#what-is-fastplotlib\">fastplotlib</a>.\nIt supports execution in both <strong>Jupyter</strong> and <strong>Qt-native</strong> environments.</p>", "a[href=\"core.html#core-api\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">1. </span>Core<a class=\"headerlink\" href=\"#core\" title=\"Link to this heading\">#</a></h1><p>Functions central to data analysis on datasets collected at the MBO.</p>", "a[href=\"usage.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">4. </span>Function Usage<a class=\"headerlink\" href=\"#function-usage\" title=\"Link to this heading\">#</a></h1><p>Examples of some common function use cases.</p>", "a[href=\"visualization.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">3. </span>Vizualization<a class=\"headerlink\" href=\"#vizualization\" title=\"Link to this heading\">#</a></h1><p>Functions to help visualize datasets.</p>", "a[href=\"core.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">1. </span>Core<a class=\"headerlink\" href=\"#core\" title=\"Link to this heading\">#</a></h1><p>Functions central to data analysis on datasets collected at the MBO.</p>", "a[href=\"#api\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">API<a class=\"headerlink\" href=\"#api\" title=\"Link to this heading\">#</a></h1><p>Python API provides helper functions for saving, loading, processing and visualizing mbo datasets.</p>", "a[href=\"io.html#io-api\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2. </span>I/O<a class=\"headerlink\" href=\"#i-o\" title=\"Link to this heading\">#</a></h1><p>Functions to help with loading and saving data.</p>", "a[href=\"visualization.html#viz-api\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">3. </span>Vizualization<a class=\"headerlink\" href=\"#vizualization\" title=\"Link to this heading\">#</a></h1><p>Functions to help visualize datasets.</p>", "a[href=\"usage.html#jupyter-ipython\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">Jupyter / IPython<a class=\"headerlink\" href=\"#jupyter-ipython\" title=\"Link to this heading\">#</a></h3><p>Returns a <code class=\"xref py py-class docutils literal notranslate\"><span class=\"pre\">fastplotlib.ImageWidget</span></code>:</p>", "a[href=\"usage.html#python-script-non-jupyter\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">Python script (non-Jupyter)<a class=\"headerlink\" href=\"#python-script-non-jupyter\" title=\"Link to this heading\">#</a></h3><p>If run from a script without data_in, and Qt is installed, a file dialog will prompt for input.</p>", "a[href=\"io.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2. </span>I/O<a class=\"headerlink\" href=\"#i-o\" title=\"Link to this heading\">#</a></h1><p>Functions to help with loading and saving data.</p>"}
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
