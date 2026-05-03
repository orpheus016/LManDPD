import numpy as np
from typing import List, Optional, Dict


def build_feature_dict(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray) -> Dict[int, np.ndarray]:
    """
    Build all 26 instantaneous features once to avoid redundant computation.
    
    Args:
        x1, x2, x3: Complex numpy arrays of shape (N,) representing the baseband signals.
        
    Returns:
        Dictionary mapping feature index (0-25) to corresponding feature array of shape (N,).
        
    Feature mapping (indices 0-25):
        0-11:   Intra-band features (x1, x1|x1|^2, x2, x2|x2|^2, x3, x3|x3|^2, |x1|, |x2|, |x3|, |x1|^3, |x2|^3, |x3|^3)
        12-17:  Cross-band envelope features (x1|x2|^2, x1|x3|^2, x2|x1|^2, x2|x3|^2, x3|x1|^2, x3|x2|^2)
        18-23:  IMD3 phase-coherent features (x1^2*conj(x2), x2^2*conj(x1), x2^2*conj(x3), x3^2*conj(x2), x1^3*conj(x3), x3^3*conj(x1))
        24-25:  Tri-band features (x1*x2*conj(x3), conj(x1)*x2^2*conj(x3))
    """
    # Compute magnitudes once
    mag1 = np.abs(x1)
    mag2 = np.abs(x2)
    mag3 = np.abs(x3)
    mag_sq1 = mag1**2
    mag_sq2 = mag2**2
    mag_sq3 = mag3**2
    mag_cube1 = mag1**3
    mag_cube2 = mag2**3
    mag_cube3 = mag3**3
    
    features = {
        # Intra-band features (0-11)
        0: x1,
        1: x1 * mag_sq1,
        2: x2,
        3: x2 * mag_sq2,
        4: x3,
        5: x3 * mag_sq3,
        # Cross-band envelope features (12-17)
        12: x1 * mag_sq2,
        13: x1 * mag_sq3,
        14: x2 * mag_sq1,
        15: x2 * mag_sq3,
        16: x3 * mag_sq1,
        17: x3 * mag_sq2,
        # IMD3 phase-coherent features (18-23)
        18: (x1**2) * np.conj(x2),
        19: (x2**2) * np.conj(x1),
        20: (x2**2) * np.conj(x3),
        21: (x3**2) * np.conj(x2),
        22: (x1**3) * np.conj(x3),
        23: (x3**3) * np.conj(x1),
        # Tri-band features (24-25)
        24: x1 * x2 * np.conj(x3),
        25: np.conj(x1) * (x2**2) * np.conj(x3),
    }
    return features


def generate_dictionary_matrix_H(x1: np.ndarray, x2: np.ndarray, x3: np.ndarray, 
                                 memory_depth_M: int, 
                                 active_groups: Optional[List[int]] = None) -> np.ndarray:
    """
    Generates the dictionary matrix H for a tri-band DPD with optional feature pruning.
    
    Args:
        x1, x2, x3: Complex numpy arrays of shape (N,) representing the baseband signals.
        memory_depth_M: Integer representing the number of past samples to include (memory).
        active_groups: Optional list of feature indices (0-25) to include. If None, all 26 are used.
        
    Returns:
        H: Complex numpy array of shape (N - M, len(active_groups or 26) * M).
    """
    N = len(x1)
    
    # Build all features upfront
    feature_dict = build_feature_dict(x1, x2, x3)
    
    # Determine which features to use
    if active_groups is None:
        # Dynamically extract available keys to prevent KeyErrors
        feature_indices = sorted(feature_dict.keys()) 
    else:
        feature_indices = sorted(active_groups)
    
    # Stack only the active features into shape (N, P_active)
    F = np.column_stack([feature_dict[idx] for idx in feature_indices])
    P_active = len(feature_indices)  # Number of active features
    
    # Construct the final matrix H by incorporating temporal memory (M)
    # We must truncate the first M samples to avoid negative indexing.
    valid_samples = N - memory_depth_M
    H = np.zeros((valid_samples, P_active * memory_depth_M), dtype=complex)
    
    for m in range(memory_depth_M):
        # Shift the features back in time by m
        shifted_F = F[memory_depth_M - m : N - m, :]
        # Place into the corresponding columns of H
        start_col = m * P_active
        end_col = (m + 1) * P_active
        H[:, start_col:end_col] = shifted_F
        
    return H


if __name__ == "__main__":
    import os
    import json
    
    # 1. Load the preprocessed binary data (both inputs and outputs)
    data_path = "datasets/RFWebLab_PA_200MHz/isolated_bands.npz"
    try:
        dataset = np.load(data_path)
        x1_data, x2_data, x3_data = dataset['x1'], dataset['x2'], dataset['x3']
        y1_data, y2_data, y3_data = dataset['y1'], dataset['y2'], dataset['y3']
    except FileNotFoundError:
        print(f"Error: Run band_separation.py first to generate {data_path}")
        exit(1)

    # 2. Define TDNN Memory Depth
    M = 4  # Memory depth as defined in Jaraut

    # 3. Load active_groups from hardware_blueprint.json if available
    blueprint_path = "dpd_out/analysis/basis_selection/hardware_blueprint.json"
    active_groups = None
    if os.path.exists(blueprint_path):
        try:
            with open(blueprint_path, 'r', encoding='utf-8') as f:
                blueprint = json.load(f)
                active_groups = blueprint.get("unified_active_groups")
                print(f"Loaded active_groups from {blueprint_path}: {active_groups}")
        except Exception as e:
            print(f"Warning: Could not load blueprint ({e}). Using all 26 features.")
    
    if active_groups is None:
        print("Using all 20 features (no blueprint found).")

    # 4. Construct the numerical dictionary matrix H (Features)
    H_matrix = generate_dictionary_matrix_H(x1_data, x2_data, x3_data, M, active_groups=active_groups)
    print(f"Shape of Dictionary Matrix H: {H_matrix.shape}")
    
    # 5. Truncate the targets (Y) to perfectly align with the memory-shifted H matrix
    y1_target = y1_data[M:]
    y2_target = y2_data[M:]
    y3_target = y3_data[M:]
    
    # Verify dimensional alignment
    assert H_matrix.shape[0] == y1_target.shape[0], "Temporal alignment failed!"

    # 6. Save the aligned H matrix and target labels for training
    out_file = "datasets/RFWebLab_PA_200MHz/H_matrix_and_Targets_M4.npz"
    np.savez(out_file, H_matrix=H_matrix, y1=y1_target, y2=y2_target, y3=y3_target)
    print(f"Successfully saved pruned basis matrix to {out_file}")
    if active_groups is not None:
        print(f"  Active groups: {active_groups}")
        print(f"  Total basis features: {len(active_groups)}")