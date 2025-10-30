selector_to_html = {"a[href=\"#gui-dependencies\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">GUI Dependencies<a class=\"headerlink\" href=\"#gui-dependencies\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#installation-guide\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\">Installation Guide<a class=\"headerlink\" href=\"#installation-guide\" title=\"Link to this heading\">#</a></h1><p>mbo_utilities has been developed to be a pure <code class=\"docutils literal notranslate\"><span class=\"pre\">pip</span></code> install.</p><p>This makes the choice of virtual-environment less relevant, you can use <code class=\"docutils literal notranslate\"><span class=\"pre\">venv</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">uv</span> <span class=\"pre\">(recommended)</span></code>, <code class=\"docutils literal notranslate\"><span class=\"pre\">conda</span></code>, it does not matter.</p>", "a[href=\"#troubleshooting\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Troubleshooting<a class=\"headerlink\" href=\"#troubleshooting\" title=\"Link to this heading\">#</a></h2><h3>Environment Issues<a class=\"headerlink\" href=\"#environment-issues\" title=\"Link to this heading\">#</a></h3><p>Many hard to diagnose installation/import bugs are due to environment issues.</p><p>The first thing you should do is check which python interpreter is being used. Generally this\nwill point to your project like:</p>", "a[href=\"#stable-from-pypi\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Stable from PyPi<a class=\"headerlink\" href=\"#stable-from-pypi\" title=\"Link to this heading\">#</a></h2>", "a[href=\"#latest-from-github\"]": "<h2 class=\"tippy-header\" style=\"margin-top: 0;\">Latest from Github<a class=\"headerlink\" href=\"#latest-from-github\" title=\"Link to this heading\">#</a></h2><p>While this pipeline is early in development, its handing to keep a version of the codebase locally using <code class=\"docutils literal notranslate\"><span class=\"pre\">git</span></code>.</p><p>Code in the <a class=\"reference external\" href=\"https://github.com/MillerBrainObservatory/mbo_utilities/tree/master\">master branch</a> is generally safe.\nIf it isn\u2019t please <a class=\"reference external\" href=\"https://github.com/MillerBrainObservatory/mbo_utilities/issues\">report an issue!</a></p>", "a[href=\"#environment-issues\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">Environment Issues<a class=\"headerlink\" href=\"#environment-issues\" title=\"Link to this heading\">#</a></h3><p>Many hard to diagnose installation/import bugs are due to environment issues.</p><p>The first thing you should do is check which python interpreter is being used. Generally this\nwill point to your project like:</p>", "a[href=\"#id1\"]": "<figure class=\"align-default\" id=\"id1\">\n<img alt=\"_images/env_jupyter.png\" src=\"_images/env_jupyter.png\"/>\n<figcaption>\n<p><span class=\"caption-text\">In jupyter, the terminal from which you ran `jupyter lab/notebook\u2019 will display the path to the python executable.</span><a class=\"headerlink\" href=\"#id1\" title=\"Link to this image\">#</a></p>\n</figcaption>\n</figure>", "a[href=\"#git-lfs-error-smudge-filter-lfs-failed\"]": "<h3 class=\"tippy-header\" style=\"margin-top: 0;\">Git LFS Error: <code class=\"docutils literal notranslate\"><span class=\"pre\">smudge</span> <span class=\"pre\">filter</span> <span class=\"pre\">lfs</span> <span class=\"pre\">failed</span></code><a class=\"headerlink\" href=\"#git-lfs-error-smudge-filter-lfs-failed\" title=\"Link to this heading\">#</a></h3><p>If you see:</p>"}
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
