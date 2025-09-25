selector_to_html = {"a[href=\"#mbo_utilities.save_mp4\"]": "<dt class=\"sig sig-object py\" id=\"mbo_utilities.save_mp4\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">mbo_utilities.</span></span><span class=\"sig-name descname\"><span class=\"pre\">save_mp4</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">fname</span></span><span class=\"p\"><span class=\"pre\">:</span></span><span class=\"w\"> </span><span class=\"n\"><a class=\"reference external\" href=\"https://docs.python.org/3.9/library/stdtypes.html#str\" title=\"(in Python v3.9)\"><span class=\"pre\">str</span></a><span class=\"w\"> </span><span class=\"p\"><span class=\"pre\">|</span></span><span class=\"w\"> </span><a class=\"reference external\" href=\"https://docs.python.org/3.9/library/pathlib.html#pathlib.Path\" title=\"(in Python v3.9)\"><span class=\"pre\">Path</span></a><span class=\"w\"> </span><span class=\"p\"><span class=\"pre\">|</span></span><span class=\"w\"> </span><span class=\"pre\">ndarray</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">images</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">framerate</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">60</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">speedup</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">1</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">chunk_size</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">100</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">cmap</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'gray'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">win</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">7</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">vcodec</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">'libx264'</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">normalize</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">True</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../_modules/mbo_utilities/plot_util.html#save_mp4\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd><p>Save a video from a 3D array or TIFF stack to <cite>.mp4</cite>.</p><p class=\"rubric\">Notes</p><p class=\"rubric\">Examples</p><p>Save a video from a 3D NumPy array with a gray colormap and 2x speedup:</p><p>Save a video with temporal averaging applied over a 5-frame window at 4x speed:</p><p>Save a video from a TIFF stack:</p></dd>", "a[href=\"#vizualization\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">2. </span>Vizualization<a class=\"headerlink\" href=\"#vizualization\" title=\"Link to this heading\">#</a></h1><p>Functions to help visualize datasets.</p>"}
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
