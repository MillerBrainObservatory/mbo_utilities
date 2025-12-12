selector_to_html = {"a[href=\"usage/index.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Usage<a class=\"headerlink\" href=\"#usage\" title=\"Link to this heading\">#</a></h1><p>Command-line tools and graphical interface documentation.</p>", "a[href=\"#contents\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Contents<a class=\"headerlink\" href=\"#contents\" title=\"Link to this heading\">#</a></h2>", "a[href=\"user_guide.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">mbo_utilities: User Guide<a class=\"headerlink\" href=\"#mbo-utilities-user-guide\" title=\"Link to this heading\">#</a></h1><p><a class=\"reference external\" href=\"https://millerbrainobservatory.github.io/mbo_utilities/installation.html\"><strong>Installation</strong></a> |\n<a class=\"reference external\" href=\"https://millerbrainobservatory.github.io/mbo_utilities/array_types.html\"><strong>Array Types</strong></a> |\n<a class=\"reference external\" href=\"https://millerbrainobservatory.github.io/\"><strong>MBO Hub</strong></a></p><p>An image I/O library with an intuitive GUI for scientific imaging data.</p>", "a[href=\"glossary.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Glossary<a class=\"headerlink\" href=\"#glossary\" title=\"Link to this heading\">#</a></h1>", "a[href=\"venvs.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Virtual Environments<a class=\"headerlink\" href=\"#virtual-environments\" title=\"Link to this heading\">#</a></h1><p>This guide covers managing python environments with <a class=\"reference external\" href=\"https://docs.astral.sh/uv/\">UV</a> and <a class=\"reference external\" href=\"https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html\">conda</a>.</p>", "a[href=\"#resources\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Resources<a class=\"headerlink\" href=\"#resources\" title=\"Link to this heading\">#</a></h2>", "a[href=\"api/index.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">API<a class=\"headerlink\" href=\"#api\" title=\"Link to this heading\">#</a></h1><p>Python API provides helper functions for saving, loading, processing and visualizing mbo datasets.</p>", "a[href=\"#quick-start\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Quick Start<a class=\"headerlink\" href=\"#quick-start\" title=\"Link to this heading\">#</a></h2>", "a[href=\"array_types.html\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Lazy Array Types<a class=\"headerlink\" href=\"#lazy-array-types\" title=\"Link to this heading\">#</a></h1><p>Understanding what <code class=\"docutils literal notranslate\"><span class=\"pre\">imread()</span></code> returns and when to use each array type.</p>", "a[href=\"#miller-brain-observatory-python-utilities\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Miller Brain Observatory: Python Utilities<a class=\"headerlink\" href=\"#miller-brain-observatory-python-utilities\" title=\"Link to this heading\">#</a></h1><p>Python tools for pre/post processing datasets at the <a class=\"reference external\" href=\"https://mbo.rockefeller.edu\">Miller Brain Observatory</a>.</p>"}
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
