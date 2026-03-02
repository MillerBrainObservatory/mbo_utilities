"""Suite2p rigid registration feature."""

from __future__ import annotations

import numpy as np
import zarr
from tqdm import tqdm
from mbo_utilities import log
from mbo_utilities.util import get_dtype

logger = log.get("arrays.features.registration")


class Suite2pRegistrationMixin:
    """
    Mixin adding Suite2p registration capabilities to arrays.

    Requires DimLabelsMixin.
    """

    def register_rigid(
        self,
        output_path: str,
        ops: dict | None = None,
        batch_size: int = 500,
        overwrite: bool = False,
    ) -> str:
        """
        Run rigid registration and save to a new Zarr store.

        Parameters
        ----------
        output_path : str
            Path to save the registered Zarr (e.g., 'registered.zarr').
        ops : dict, optional
            Suite2p options. Defaults are used if None.
        batch_size : int, optional
            Number of frames to process per batch.
        overwrite : bool, optional
            If True, overwrite existing output.

        Returns
        -------
        str
            Path to the registered Zarr store.
        """
        # 1. Validation
        if not self.has_dim("T") or not self.has_dim("Y") or not self.has_dim("X"):
            raise ValueError("Array must have T, Y, and X dimensions for registration.")

        try:
            from suite2p import default_ops
            from suite2p.registration import register, rigid
        except ImportError:
            raise ImportError(
                "suite2p is required. Install with `uv pip install suite2p`."
            )

        user_ops = ops or {}
        ops = default_ops()
        ops.update(user_ops)

        # Override critical ops with array properties
        t_idx = self.dim_index("T")
        y_idx = self.dim_index("Y")
        x_idx = self.dim_index("X")

        T = self.shape[t_idx]
        Y = self.shape[y_idx]
        X = self.shape[x_idx]

        ops.update({"nframes": T, "Ly": Y, "Lx": X, "batch_size": batch_size})

        # 2. Compute Reference
        # We need to grab a subset of frames.
        # Ideally, use self[slice] to respect lazy loading.
        logger.info("Computing reference image...")

        # Grab init frames (default 400 or less)
        n_init = min(ops.get("nimg_init", 400), T)
        # Use simple slicing for now - assuming T is dim 0 for slicing efficiency in Zarr usually,
        # but we should use dim_index compliant slicing if possible.
        # However, ZarrArray is currently fixed to TZYX or T,H,W.
        # If the array is ZarrArray, it handles slicing.

        # Construct slice tuple for frames [0:n_init]
        # We want all Y, X. If Z exists (dim_index("Z")), we might need to iterate planes?
        # WARNING: This mixin ignores Z for now or assumes 4D input is registered plane-by-plane externally?
        # User prompt said "registration of .zarr files... if DimLabels match".
        # Suite2p is strictly 2D (T, Y, X).
        # If input is 4D (T, Z, Y, X), we should probably error or ask for specific plane?
        # Or loop over Z?
        # The `zarr_s2p.py` script handled a specific Z-plane slice `arr_in`.
        # So this method should operate on the *current view* which implies 3D (T, Y, X).
        # If self is 4D, user should likely slice it first: `arr[0:T, z, :, :].register(...)`.

        # Validate Z dimension
        z_idx = self.dim_index("Z") if self.has_dim("Z") else None

        if self.ndim == 4:
            if z_idx is not None and self.shape[z_idx] > 1:
                raise ValueError(
                    "Input array has multiple Z-planes. "
                    "Please slice to a single plane before registering (e.g. arr[:, z_idx])."
                )

        # Extract init frames
        # Build slice object
        slices = [slice(None)] * self.ndim
        # T subset
        if t_idx is not None:
            slices[t_idx] = slice(0, n_init)

        init_data = self[tuple(slices)]

        # Squeeze singleton Z if present to get (N_init, Y, X)
        # We assume init_data preserves ndim
        # If input was (T, 1, Y, X), output is (N_init, 1, Y, X)
        if hasattr(init_data, "shape") and len(init_data.shape) == 4:
            # Squeeze Z
            # If dims are TZYX, axis 1.
            # We rely on shape check: if dimension is 1, squeeze.
            # But we must be careful not to squeeze T=1 if n_init=1.
            # We want to squeeze the SPATIAL Z.
            if z_idx is not None and init_data.shape[z_idx] == 1:
                init_frames = np.squeeze(init_data, axis=z_idx)
            else:
                init_frames = init_data
        else:
            init_frames = init_data

        # Ensure numpy
        if not isinstance(init_frames, np.ndarray):
            init_frames = np.array(init_frames)

        # Ref computation requires (N, Y, X)
        # Fail if not 3D at this point
        if init_frames.ndim != 3:
            # Might happen if data was strictly 3D and we didn't squeeze anything?
            # Or if we squeezed too much?
            raise ValueError(
                f"Expected 3D frames for reference computation, got {init_frames.shape}"
            )

        # Ref computation
        ref = register.compute_reference(init_frames, ops)

        if isinstance(ref, dict):
            ops.update(ref)
        else:
            ops["refImg"] = ref

        # 3. Compute Masks (The API Fix)
        logger.info(
            f"Reference computed (shape {ops['refImg'].shape}). Computing masks..."
        )
        refAndMasks = register.compute_reference_masks(ops["refImg"], ops)

        # 4. Prepare Output Zarr
        store_path = output_path
        z_out_root = zarr.open(store_path, mode="w")

        z_grp = z_out_root

        dtype = get_dtype(self.dtype)
        out_shape = list(self.shape)

        chunks = [1] * self.ndim
        if y_idx is not None:
            chunks[y_idx] = Y
        if x_idx is not None:
            chunks[x_idx] = X

        z_arr = z_grp.create_dataset(
            "0",
            shape=tuple(out_shape),
            chunks=tuple(chunks),
            dtype=dtype,
            overwrite=True,
        )

        # Metadata
        z_grp.attrs["multiscales"] = [{"version": "0.5", "datasets": [{"path": "0"}]}]

        ops_serializable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in ops.items()
        }
        z_grp.attrs["ops"] = ops_serializable

        # 5. Registration Loop
        logger.info(f"Starting registration loop (T={T})...")

        for i in tqdm(range(0, T, batch_size), desc="Registering"):
            # Slice input
            end = min(i + batch_size, T)
            sl = [slice(None)] * self.ndim
            if t_idx is not None:
                sl[t_idx] = slice(i, end)

            chunk_data = self[tuple(sl)]

            # Reduce to 3D for Suite2p
            chunk_3d = chunk_data
            if chunk_data.ndim == 4 and z_idx is not None:
                chunk_3d = np.squeeze(chunk_data, axis=z_idx)

            if not isinstance(chunk_3d, np.ndarray):
                chunk_3d = np.array(chunk_3d)

            # Register
            outputs = register.register_frames(refAndMasks, chunk_3d, ops=ops)
            reg_chunk = outputs[0]

            if hasattr(reg_chunk, "get"):
                reg_chunk = reg_chunk.get()

            # Expand back to 4D for writing if needed
            if self.ndim == 4 and reg_chunk.ndim == 3:
                # Insert Z axis back
                reg_chunk_out = np.expand_dims(reg_chunk, axis=z_idx)
            else:
                reg_chunk_out = reg_chunk

            # Write output
            z_arr[tuple(sl)] = reg_chunk_out

        logger.info(f"Registration complete. Saved to {store_path}")
        return str(store_path)
