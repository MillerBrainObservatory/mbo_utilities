selector_to_html = {"a[href=\"#mbo_utilities.get_metadata\"]": "<dt class=\"sig sig-object py\" id=\"mbo_utilities.get_metadata\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">mbo_utilities.</span></span><span class=\"sig-name descname\"><span class=\"pre\">get_metadata</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">file</span></span><span class=\"p\"><span class=\"pre\">:</span></span><span class=\"w\"> </span><span class=\"n\"><span class=\"pre\">PathLike</span><span class=\"w\"> </span><span class=\"p\"><span class=\"pre\">|</span></span><span class=\"w\"> </span><span class=\"pre\">str</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">z_step</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">None</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">verbose</span></span><span class=\"o\"><span class=\"pre\">=</span></span><span class=\"default_value\"><span class=\"pre\">False</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../_modules/mbo_utilities/metadata.html#get_metadata\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd><p>Extract metadata from a TIFF file produced by ScanImage or processed via the save_as function.</p><p>This function opens the given TIFF file and retrieves critical imaging parameters and acquisition details.\nIt supports both raw ScanImage TIFFs and those modified by downstream processing. If the file contains\nraw ScanImage metadata, the function extracts key fields such as channel information, number of frames,\nfield-of-view, pixel resolution, and ROI details. When verbose output is enabled, the complete metadata\ndocument is returned in addition to the parsed key values.</p><p class=\"rubric\">Notes</p><p class=\"rubric\">Examples</p></dd>", "a[href=\"#core\"]": "<h1 class=\"tippy-header\" style=\"margin-top: 0;\"><span class=\"section-number\">1. </span>Core<a class=\"headerlink\" href=\"#core\" title=\"Link to this heading\">#</a></h1><p>Functions central to data analysis on datasets collected at the MBO.</p>", "a[href=\"#mbo_utilities.read_scan\"]": "<dt class=\"sig sig-object py\" id=\"mbo_utilities.read_scan\">\n<span class=\"sig-prename descclassname\"><span class=\"pre\">mbo_utilities.</span></span><span class=\"sig-name descname\"><span class=\"pre\">read_scan</span></span><span class=\"sig-paren\">(</span><em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">pathnames</span></span></em>, <em class=\"sig-param\"><span class=\"n\"><span class=\"pre\">dtype=&lt;class</span> <span class=\"pre\">'numpy.int16'&gt;</span></span></em><span class=\"sig-paren\">)</span><a class=\"reference internal\" href=\"../_modules/mbo_utilities/file_io.html#read_scan\"><span class=\"viewcode-link\"><span class=\"pre\">[source]</span></span></a></dt><dd><p>Reads a ScanImage scan from a given file or set of file paths and returns a\nScanMultiROIReordered object with lazy-loaded data.</p><p class=\"rubric\">Notes</p><p>If the provided path string appears to include escaped characters (for example,\nunintentional backslashes), a warning message is printed suggesting the use of a\nraw string (r\u2019\u2026\u2019) or double backslashes.</p><p class=\"rubric\">Examples</p></dd>"}
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
