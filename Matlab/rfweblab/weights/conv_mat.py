"""
Convert PyTorch .pt model weights to MATLAB .mat file.

Usage:
    python convert_pt_to_mat.py

Make sure the .pt file is in the same directory as this script,
or update PT_FILE_PATH below.

Requirements:
    pip install torch scipy numpy
"""

import torch
import numpy as np
from scipy.io import savemat
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
PT_FILE_PATH = "DPD_S_0_M_DELTAGRU_TCNSKIP_H_15_F_200_P_999_THX_0.010_THH_0.050.pt"
MAT_FILE_PATH = PT_FILE_PATH.replace(".pt", ".mat")
# ─────────────────────────────────────────────────────────────────────────────


def sanitize_key(key: str) -> str:
    """
    MATLAB variable names must start with a letter and contain only
    alphanumeric characters or underscores. Replace dots and dashes.
    """
    return key.replace(".", "_").replace("-", "_")


def tensor_to_numpy(val):
    """Convert a PyTorch tensor to a numpy array (CPU, float64)."""
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().numpy().astype(np.float64)
    return val


def load_checkpoint(path: str):
    """
    Try loading the checkpoint. PyTorch .pt files can be:
      - A raw state_dict (OrderedDict of tensors)
      - A dict containing 'model_state_dict', 'state_dict', or other keys
      - A full model object (less common for sharing weights)
    """
    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint


def extract_state_dict(checkpoint):
    """Return a flat state_dict regardless of how the checkpoint was saved."""
    if isinstance(checkpoint, dict):
        # Common wrapper keys
        for key in ("model_state_dict", "state_dict", "model", "net"):
            if key in checkpoint:
                print(f"  Found state_dict under key: '{key}'")
                return checkpoint[key]
        # If all values are tensors, it IS the state_dict
        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint
        # Otherwise dump everything (scalars, arrays, nested dicts)
        return checkpoint
    # Raw state_dict (OrderedDict)
    return checkpoint


def flatten_dict(d, parent_key="", sep="_"):
    """Recursively flatten nested dicts so every leaf becomes a top-level key."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def main():
    if not os.path.isfile(PT_FILE_PATH):
        raise FileNotFoundError(
            f"Could not find: {PT_FILE_PATH}\n"
            "Please place the .pt file in the same directory as this script "
            "or update PT_FILE_PATH at the top of the script."
        )

    print(f"Loading checkpoint: {PT_FILE_PATH}")
    checkpoint = load_checkpoint(PT_FILE_PATH)

    print("Extracting state dict …")
    state_dict = extract_state_dict(checkpoint)

    # Flatten any nested structure
    if isinstance(state_dict, dict):
        state_dict = flatten_dict(state_dict)

    # Build the MATLAB-compatible dict
    mat_data = {}
    skipped = []

    for raw_key, val in state_dict.items():
        mat_key = sanitize_key(raw_key)

        if isinstance(val, torch.Tensor):
            arr = tensor_to_numpy(val)
            # Scalars → 1×1 matrix (MATLAB friendly)
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            mat_data[mat_key] = arr
            print(f"  {mat_key:60s}  shape={arr.shape}  dtype={arr.dtype}")

        elif isinstance(val, (int, float, bool)):
            mat_data[mat_key] = np.array([[val]], dtype=np.float64)
            print(f"  {mat_key:60s}  scalar={val}")

        elif isinstance(val, np.ndarray):
            mat_data[mat_key] = val.astype(np.float64)
            print(f"  {mat_key:60s}  shape={val.shape}  (numpy)")

        else:
            skipped.append((raw_key, type(val).__name__))

    if skipped:
        print("\nSkipped (non-numeric) keys:")
        for k, t in skipped:
            print(f"  {k}  ({t})")

    print(f"\nSaving → {MAT_FILE_PATH}")
    savemat(MAT_FILE_PATH, mat_data, do_compression=True)
    print("Done! ✓")
    print(f"\nIn MATLAB, load with:\n  data = load('{MAT_FILE_PATH}');")
    print("Then access weights like:\n  data.weight_ih_l0   (example GRU weight)")


if __name__ == "__main__":
    main()