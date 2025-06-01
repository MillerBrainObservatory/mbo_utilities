# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "tifffile",
#     "mbo_utilities",
#     "numpy",
# ]
#
# [tool.uv.sources]
# mbo_utilities = { git = "https://github.com/MillerBrainObservatory/mbo_utilities", branch = "master" }
# lbm_suite2p_python = { git = "https://github.com/MillerBrainObservatory/lbm_suite2p_python", branch = "master" }
import os
import tifffile
import re
import json


def find_scanimage_metadata(path):
    with tf.TiffFile(path) as tif:
        if hasattr(tif, "scanimage_metadata"):
            return tif.scanimage_metadata
        p = tif.pages[0]
        cand = []
        for tag in ("ImageDescription", "Software"):
            if tag in p.tags:
                cand.append(p.tags[tag].value)
        if getattr(p, "description", None):
            cand.append(p.description)
        cand.extend(str(tif.__dict__.get(k, "")) for k in tif.__dict__)
        for s in cand:
            if isinstance(s, bytes):
                s = s.decode(errors="ignore")
            m = re.search(r"{.*ScanImage.*}", s, re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return m.group(0)
    return None


def main():
    directory = r"/home/flynn/lbm_data/raw"
    for fname in os.listdir(directory):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue
        path = os.path.join(directory, fname)
        tf = tifffile.TiffFile(path)
        meta = getattr(tf, "scanimage_metadata", {}) or {}
        fd = meta.get("FrameData", {})
        num_meta = fd.get("SI.hStackManager.framesPerSlice") or fd.get(
            "SI.hRoiManager.framesPerSlice"
        )
        series_frames = tf.series[0].shape[0]
        pages = len(tf.pages)
        print(
            f"{fname}: meta_frames={num_meta}, series_frames={series_frames}, pages={pages}"
        )


if __name__ == "__main__":
    main()
