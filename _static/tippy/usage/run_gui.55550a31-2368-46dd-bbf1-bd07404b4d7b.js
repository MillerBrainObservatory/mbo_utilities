selector_to_html = {"a[href=\"#run-gui\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2. </span><code class=\"docutils literal notranslate\"><span class=\"pre\">run_gui</span></code><a class=\"headerlink\" href=\"#run-gui\" title=\"Link to this heading\">#</a></h1><p><code class=\"xref py py-func docutils literal notranslate\"><span class=\"pre\">mbo_utilities.run_gui()</span></code> is a convenience function for viewing common data formats used in processing MBO datasets, usable from the terminal, jupyter-lab or a callable from a python script.</p>", "a[href=\"#python-script-non-jupyter\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.2. </span>Python script (non-Jupyter)<a class=\"headerlink\" href=\"#python-script-non-jupyter\" title=\"Link to this heading\">#</a></h2><p>If run from a script without data_in, and Qt is installed, a file dialog will prompt for input.</p>", "a[href=\"#jupyter-ipython\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2.1. </span>Jupyter / IPython<a class=\"headerlink\" href=\"#jupyter-ipython\" title=\"Link to this heading\">#</a></h2><p>Returns a <a class=\"reference external\" href=\"https://www.fastplotlib.org/api/widgets/ImageWidget_api/fastplotlib.ImageWidget.html#fastplotlib.ImageWidget\" title=\"(in fastplotlib)\"><code class=\"xref py py-class docutils literal notranslate\"><span class=\"pre\">fastplotlib.ImageWidget</span></code></a>:</p>"}
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
