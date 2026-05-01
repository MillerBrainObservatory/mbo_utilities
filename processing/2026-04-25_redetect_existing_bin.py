"""
Re-run detection on an existing registered binary.

Workflow: run lsp.pipeline once to register + detect, then point lsp.run_plane
at the resulting data.bin and re-run detection only with different params.

Run with: uv run python processing/2026-04-25_redetect_existing_bin.py
"""
from pathlib import Path
import shutil
import numpy as np

import lbm_suite2p_python as lsp


tiff_dir  = Path(r"D:/demo/bidi_compare/tiff_mbo_corrected")
save_path = Path(r"D:/demo/redetect_demo")
save_path.mkdir(parents=True, exist_ok=True)

PLANE = 7  # 1-based


def banner(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}")


# 1. initial pipeline: register + detect
banner("STEP 1: initial pipeline (register + detect)")
ops_initial = lsp.default_ops()
ops_initial["diameter"] = 6
ops_initial["anatomical_only"] = 3

lsp.pipeline(
    input_data=tiff_dir,
    save_path=save_path,
    ops=ops_initial,
    planes=[PLANE],
    keep_reg=True,
    keep_raw=False,
    force_reg=True,
    force_detect=True,
)


# 2. locate the plane dir
banner("STEP 2: locate plane dir + inspect outputs")
plane_dirs = sorted(save_path.glob(f"zplane{PLANE:02d}_*"))
assert plane_dirs, f"no plane{PLANE} output under {save_path}"
plane_dir = plane_dirs[0]
print("plane dir:", plane_dir)
for f in ("ops.npy", "data.bin", "data_raw.bin", "stat.npy", "F.npy", "iscell.npy"):
    p = plane_dir / f
    size = p.stat().st_size if p.exists() else 0
    print(f"  {f:14s} {'OK' if p.exists() else 'MISSING':>8s}  {size:>14,d} bytes")

ops0    = np.load(plane_dir / "ops.npy",    allow_pickle=True).item()
stat0   = np.load(plane_dir / "stat.npy",   allow_pickle=True)
iscell0 = np.load(plane_dir / "iscell.npy", allow_pickle=True)[:, 0].astype(bool)
print(f"\ninitial detection: {len(stat0)} ROIs total, {iscell0.sum()} accepted")
print(f"initial diameter:  {ops0.get('diameter')}")
print(f"Ly,Lx:             {ops0.get('Ly')}, {ops0.get('Lx')}")
print(f"nframes:           {ops0.get('nframes_chan1') or ops0.get('nframes')}")


# 3. snapshot run1
banner("STEP 3: snapshot run1 outputs")
snap_dir = plane_dir / "_snapshot_run1"
snap_dir.mkdir(exist_ok=True)
for fname in ("ops.npy", "stat.npy", "F.npy", "Fneu.npy", "iscell.npy", "spks.npy"):
    src = plane_dir / fname
    if src.exists():
        shutil.copy2(src, snap_dir / fname)
print("snapshotted run1 outputs to", snap_dir)


# 4. re-run detection only on the same data.bin
banner("STEP 4: re-run detection only (force_detect=True)")
ops_redetect = {
    "diameter": 4,
    "anatomical_only": 4,
    "cellprob_threshold": -2.0,
    "do_registration": 0,
    "roidetect": 1,
}

print("invoking lsp.run_plane(input_data=plane_dir/'data.bin', save_path=plane_dir.parent, force_detect=True)")
print("ops overrides:", ops_redetect)
lsp.run_plane(
    input_data=plane_dir / "data.bin",
    save_path=plane_dir.parent,
    ops=ops_redetect,
    force_reg=False,
    force_detect=True,
    keep_reg=True,
)


# 5. compare
banner("STEP 5: compare run1 vs run2")
stat1 = np.load(snap_dir / "stat.npy",   allow_pickle=True)
isc1  = np.load(snap_dir / "iscell.npy", allow_pickle=True)[:, 0].astype(bool)
stat2 = np.load(plane_dir / "stat.npy",   allow_pickle=True)
isc2  = np.load(plane_dir / "iscell.npy", allow_pickle=True)[:, 0].astype(bool)
ops2  = np.load(plane_dir / "ops.npy",    allow_pickle=True).item()

print(f"run1: {len(stat1):4d} total, {isc1.sum():4d} accepted   (diameter={ops0.get('diameter')})")
print(f"run2: {len(stat2):4d} total, {isc2.sum():4d} accepted   (diameter={ops2.get('diameter')})")

if len(stat1) == len(stat2) and isc1.sum() == isc2.sum():
    same_centroids = all(
        np.allclose(s1["med"], s2["med"]) for s1, s2 in zip(stat1, stat2)
    )
    if same_centroids:
        print("\nWARNING: run1 and run2 produced identical ROIs — detection did NOT actually re-run.")
    else:
        print("\nrun2 produced a different set of ROIs (same count but different positions).")
else:
    print("\nrun2 produced a different ROI set — detection re-ran.")
