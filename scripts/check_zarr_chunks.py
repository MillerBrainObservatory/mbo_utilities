"""Inspect actual chunk shapes for the corrected zebrafish dataset."""
from pathlib import Path
import zarr
import json

ROOT = Path(r"D:\2026_05_light-sheet_workshop\2_zebrafish\zebrafish.corrected\SPM00\TM000000")

for p in sorted(ROOT.iterdir()):
    if p.suffix != ".zarr":
        continue
    print(f"\n=== {p.name} ===")
    try:
        g = zarr.open(str(p), mode="r")
        # walk to find arrays
        if hasattr(g, "shape"):
            arrs = [("", g)]
        else:
            arrs = []
            def walk(grp, prefix=""):
                for k in grp:
                    item = grp[k]
                    if hasattr(item, "shape"):
                        arrs.append((f"{prefix}/{k}", item))
                    else:
                        walk(item, f"{prefix}/{k}")
            walk(g)
        for name, a in arrs:
            print(f"  path={name or '/'}  shape={a.shape}  chunks={a.chunks}  dtype={a.dtype}")
    except Exception as e:
        print(f"  ERROR: {e}")

    zj = p / "zarr.json"
    if zj.is_file():
        j = json.loads(zj.read_text())
        print(f"  zarr.json zarr_format={j.get('zarr_format')}")
        if "chunk_grid" in j:
            print(f"  chunk_grid={j['chunk_grid']}")
        if "shape" in j:
            print(f"  shape={j['shape']}")
