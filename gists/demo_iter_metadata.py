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


def nest_keys(flat_dict, root_prefix="SI."):
    nested = {}
    for key, value in flat_dict.items():
        if not key.startswith(root_prefix):
            continue
        parts = key[len(root_prefix) :].split(".")
        current = nested
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return nested


def to_markdown(d, indent=0):
    md = ""
    for key, val in d.items():
        if isinstance(val, dict):
            md += "  " * indent + f"<details>\n  <summary>{key}</summary>\n\n"
            md += to_markdown(val, indent + 1)
            md += "  " * indent + "</details>\n\n"
        else:
            md += "  " * indent + f"- **{key}**: `{val}`\n"
    return md


def to_markdown_all(files_dict):
    md = ""
    for fname, metadata in files_dict.items():
        md += f"<details>\n  <summary>{fname}</summary>\n\n"
        md += to_markdown(metadata, indent=1)
        md += "</details>\n\n"
    return md


def main():
    directory = r"D://W2_DATA//foconnell//2025-07-10_Pollen"
    files = {}
    for fname in os.listdir(directory):
        if not fname.lower().endswith((".tif", ".tiff")):
            continue
        path = os.path.join(directory, fname)
        tf = tifffile.TiffFile(path)
        meta = getattr(tf, "scanimage_metadata", {}) or {}
        fd = meta.get("FrameData", {})
        metadata = nest_keys(fd, root_prefix="SI.")
        files[fname] = metadata
        return files
    return None


if __name__ == "__main__":
    files = main()
    markdown_output = to_markdown_all(files)
    print(markdown_output)
    # Save to a markdown file
    with open("metadata_summary.md", "w") as f:
        f.write(markdown_output)
