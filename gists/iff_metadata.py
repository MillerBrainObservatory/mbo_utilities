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


def main():
    directory = r"D:\W2_DATA\wsnyder\2025_03_06\raw\dot_lv6"
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
