"""Resume multi_fuse into zebrafish.fused_v2, remaining timepoints only.

Matches the existing .fused_v2 run params, overriding only the per-view
transition planes: VW00=71, VW90=40. overwrite=False skips the 98 already
fused and processes TM000098-TM000300.
"""
from pathlib import Path
from isoview import ProcessingConfig, multi_fuse

CORRECTED = Path(r"E:\2026_05_light-sheet_workshop\2_zebrafish\zebrafish.corrected")


def main():
    config = ProcessingConfig(
        input_dir=CORRECTED,
        output_suffix="v2",          # -> zebrafish.fused_v2
        overwrite=False,             # resume: skip existing, fuse the rest
        log=True,
        workers=3,  # lowered from 6: 6 spiked peak RAM and OOM'd on large timepoints
        compression_level=1,
        pixel_spacing_z=5.13,
        detection_objective_mag=16.0,
        blending_method="geometric",
        blending_range=5,
        transition_plane=71,
        front_flag=1,
        search_offsets_x=(-100, 100, 10),
        transition_plane_by_view={0: 71, 90: 40},
    )

    print(f"input_dir : {config.input_dir}")
    print(f"output_dir: {config.output_dir}")
    print(f"fused_dir : {config.fused_dir}")
    print(f"timepoints: {len(config.timepoints)} (overwrite={config.overwrite})")
    print(f"transition_plane_by_view: {config.transition_plane_by_view}")

    multi_fuse(config)
    print("multi_fuse returned")


if __name__ == "__main__":
    main()
