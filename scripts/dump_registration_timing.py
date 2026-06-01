"""Print plane_times.registration from each plane's ops.npy."""
from pathlib import Path
import numpy as np

ROOT = Path(r"D:\2026_05_light-sheet_workshop\2_zebrafish\Dre_HuC_H2BGCaMP6s_0-1_20150709_195932.corrected.registered\suite2p")

planes = sorted(p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("zplane"))

ops0 = np.load(planes[0] / "ops.npy", allow_pickle=True).item()
pt0 = ops0.get("plane_times", {})
print(f"plane_times keys: {list(pt0.keys())}")
print()

total = 0.0
for pdir in planes:
    ops = np.load(pdir / "ops.npy", allow_pickle=True).item()
    pt = ops.get("plane_times", {}) or {}
    reg = pt.get("registration")
    print(f"{pdir.name}: registration={reg!r}")
    if isinstance(reg, (int, float)):
        total += reg

print(f"\nTotal registration across {len(planes)} planes: {total:.2f}s ({total/60:.2f} min)")
