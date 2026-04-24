"""comprehensive regression test for the source_path fix.

exercises save -> load -> suite2p on all supported filetypes from a real
multi-file raw LBM recording (D:/demo/raw). raw-scanimage path also runs
axial registration via compute_axial_shifts. every other frame + every
other plane keeps the volume small enough to finish in ~30 min.

each step is benched and caught independently so one failure doesn't
block the remaining tests. final report is a table of pass/fail with
timings and shape/type evidence.
"""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

from mbo_utilities import imread, imwrite
from mbo_utilities.arrays._registration import (
    compute_axial_shifts,
    validate_axial_shifts,
)
from lbm_suite2p_python import pipeline


INPUT = Path("D:/demo/raw")
OUT = Path("D:/demo/bugfix_test")

# every other frame + every other plane (1-based)
EVERY_OTHER_FRAMES = list(range(1, 1575, 2))
EVERY_OTHER_PLANES = list(range(1, 15, 2))
# lsp's pipeline takes 0-based frame_indices
FRAME_INDICES_0B = [i - 1 for i in EVERY_OTHER_FRAMES]

FILETYPES = [".tiff", ".bin", ".zarr", ".h5"]


@dataclass
class Result:
    name: str
    ok: bool = False
    elapsed_s: float = 0.0
    detail: str = ""


def bench(name: str, fn, *args, **kwargs) -> Result:
    r = Result(name=name)
    t0 = time.perf_counter()
    try:
        detail = fn(*args, **kwargs) or ""
        r.ok = True
        r.detail = str(detail)
    except Exception as e:
        r.ok = False
        r.detail = f"{type(e).__name__}: {e}"
        traceback.print_exc()
    r.elapsed_s = time.perf_counter() - t0
    return r


def do_save(ext: str, outdir: Path) -> str:
    arr = imread(INPUT)
    result = imwrite(
        arr,
        outdir,
        ext=ext,
        frames=EVERY_OTHER_FRAMES,
        planes=EVERY_OTHER_PLANES,
        overwrite=True,
    )
    return f"wrote -> {result}"


def do_load(path: Path) -> str:
    arr = imread(path)
    return (
        f"{type(arr).__name__}  shape5d={tuple(arr.shape5d)}  "
        f"source_path={arr.source_path}"
    )


def do_suite2p(
    input_path: Path | str,
    save_path: Path,
    *,
    register_z: bool,
    planes: list[int] | None,
    frame_indices_0b: list[int] | None,
) -> str:
    ops: dict = {}
    if register_z:
        src_arr = imread(input_path)
        n_planes = src_arr.nz
        reg_meta: dict = dict(src_arr.metadata or {})
        if not validate_axial_shifts(reg_meta, n_planes):
            compute_axial_shifts(src_arr, metadata=reg_meta)
            if not validate_axial_shifts(reg_meta, n_planes):
                raise RuntimeError("axial registration produced no valid plane_shifts")
        ops = {"apply_shift": True, "plane_shifts": reg_meta["plane_shifts"]}

    pipeline(
        str(input_path),
        save_path=str(save_path),
        ops=ops,
        planes=planes,
        frame_indices=frame_indices_0b,
        force_reg=True,
        force_detect=False,
    )

    plane_dirs = sorted(
        d for d in save_path.iterdir()
        if d.is_dir() and d.name.startswith("zplane")
    )
    ops_files = list(save_path.rglob("ops.npy"))
    return f"{len(plane_dirs)} plane dir(s), {len(ops_files)} ops.npy file(s)"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results: list[Result] = []

    print("=" * 78)
    print(f"input: {INPUT}")
    print(f"subsampling: {len(EVERY_OTHER_FRAMES)} frames, {len(EVERY_OTHER_PLANES)} planes")
    print(f"output: {OUT}")
    print("=" * 78)

    # phase 1: save to all filetypes
    saved: dict[str, Path] = {}
    for ext in FILETYPES:
        key = ext.lstrip(".")
        outdir = OUT / f"save_{key}"
        outdir.mkdir(parents=True, exist_ok=True)
        r = bench(f"save {ext}", do_save, ext, outdir)
        results.append(r)
        if r.ok:
            saved[key] = outdir

    # phase 2: load each saved artefact
    for key, outdir in list(saved.items()):
        r = bench(f"load {key}", do_load, outdir)
        results.append(r)
        if not r.ok:
            saved.pop(key)

    # phase 3: suite2p on raw (with axial reg)
    s2p_raw = OUT / "s2p_raw_with_axial"
    s2p_raw.mkdir(parents=True, exist_ok=True)
    results.append(
        bench(
            "suite2p raw + axial",
            do_suite2p,
            INPUT,
            s2p_raw,
            register_z=True,
            planes=EVERY_OTHER_PLANES,
            frame_indices_0b=FRAME_INDICES_0B,
        )
    )

    # phase 4: suite2p on each saved artefact (no axial reg — already subsampled)
    for key, outdir in saved.items():
        s2p_out = OUT / f"s2p_from_{key}"
        s2p_out.mkdir(parents=True, exist_ok=True)
        results.append(
            bench(
                f"suite2p from {key}",
                do_suite2p,
                outdir,
                s2p_out,
                register_z=False,
                planes=None,
                frame_indices_0b=None,
            )
        )

    # report
    print()
    print("=" * 78)
    print(f"{'test':<28s}  {'ok':<4s}  {'time':>7s}  detail")
    print("-" * 78)
    for r in results:
        status = "OK" if r.ok else "FAIL"
        print(f"{r.name:<28s}  {status:<4s}  {r.elapsed_s:>6.1f}s  {r.detail}")
    print("-" * 78)
    print(f"passed: {sum(r.ok for r in results)}/{len(results)}")


if __name__ == "__main__":
    main()
