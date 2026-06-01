"""Fix the multiscales `scale` axis order in an existing BDV dataset.ome.zarr.

BigStitcher's reader interprets scale[0,1,2] positionally as x,y,z (ignoring
the t,c,z,y,x axis labels). Pyramids written with axes-order scale
[1,1,z,y,x] therefore look like factor 1 at every level (XY-only downsampling
lands in z,y,x positions 2,3,4) and BigStitcher collapses them to a single
resolution. This rewrites each sub-zarr's .zattrs with spatial-first scale
[x, y, z, c, t], derived from the on-disk level shapes. Arrays are untouched.

Usage:
    uv run python scripts/fix_bdv_multiscales_scale.py <dataset.ome.zarr>
"""
import json
import sys
from pathlib import Path


def _levels(sub: Path) -> list[str]:
    return sorted((p.name for p in sub.iterdir() if p.name.isdigit()), key=int)


def _spatial(sub: Path, level: str) -> tuple[int, int, int]:
    shp = json.loads((sub / level / ".zarray").read_text())["shape"]  # [t,c,z,y,x]
    return shp[2], shp[3], shp[4]  # z, y, x


def _fix_one(sub: Path) -> int:
    levels = _levels(sub)
    z0, y0, x0 = _spatial(sub, levels[0])
    datasets = []
    for lv in levels:
        z, y, x = _spatial(sub, lv)
        zf, yf, xf = round(z0 / z), round(y0 / y), round(x0 / x)
        datasets.append({
            "path": lv,
            # axes order [t,c,z,y,x] -- matches BigStitcher-Spark's own
            # resave output (verified). n5-universe reverses to F-order on
            # read, so this is what makes mipmaps register correctly.
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, float(zf), float(yf), float(xf)]},
                {"type": "translation", "translation": [0.0] * 5},
            ],
        })
    (sub / ".zattrs").write_text(json.dumps({"multiscales": [{
        "name": "/", "version": "0.4",
        "axes": [
            {"type": "time", "name": "t"}, {"type": "channel", "name": "c"},
            {"type": "space", "name": "z", "unit": "micrometer"},
            {"type": "space", "name": "y", "unit": "micrometer"},
            {"type": "space", "name": "x", "unit": "micrometer"},
        ],
        "datasets": datasets,
        "coordinateTransformations": [{"type": "scale", "scale": [1.0] * 5}],
    }]}))
    return len(levels)


def main() -> None:
    root = Path(sys.argv[1])
    subs = sorted(p for p in root.glob("s*-t*.zarr") if p.is_dir())
    print(f"root: {root}\nsub-zarrs: {len(subs)}")
    for i, sub in enumerate(subs, 1):
        n = _fix_one(sub)
        if i % 100 == 0 or i == len(subs):
            print(f"  [{i}/{len(subs)}] levels={n}", flush=True)
    print("done")


if __name__ == "__main__":
    main()
