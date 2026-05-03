import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class StandardizationParams:
    mean: np.ndarray
    scale: np.ndarray


@dataclass
class SweepResult:
    feature_counts: List[int]
    nmse_db: List[float]
    active_indices: List[np.ndarray]
    model_params: List[Dict[str, float]]


def project_complex_to_real_concat(h_complex: np.ndarray, y_complex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if h_complex.ndim != 2:
        raise ValueError("H_matrix must be 2D (N_samples x P_features).")
    if y_complex.ndim != 1:
        raise ValueError("Y_target must be 1D (N_samples,).")
    if h_complex.shape[0] != y_complex.shape[0]:
        raise ValueError("H_matrix and Y_target must have the same number of samples.")

    h_real = np.column_stack((h_complex.real, h_complex.imag))
    y_real = np.column_stack((y_complex.real, y_complex.imag))
    return h_real, y_real


def parse_stopbands(stopbands: str) -> List[Tuple[float, float]]:
    """
    Parse a stopband string like "-110e6,-90e6;90e6,110e6" into [(f1,f2), (f3,f4)].
    """
    bands = []
    for pair in stopbands.split(";"):
        low_str, high_str = pair.split(",")
        bands.append((float(low_str), float(high_str)))
    return bands


def apply_frequency_weighting(
    h_complex: np.ndarray,
    y_complex: np.ndarray,
    fs: float,
    stopbands: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply frequency-domain band-stop weighting to suppress carrier bands.
    """
    if not stopbands:
        return h_complex, y_complex

    n = y_complex.shape[0]
    freqs = np.fft.fftfreq(n, d=1.0 / fs)
    mask = np.ones(n, dtype=float)
    for f_low, f_high in stopbands:
        f_min, f_max = min(f_low, f_high), max(f_low, f_high)
        mask[(freqs >= f_min) & (freqs <= f_max)] = 0.0

    y_fft = np.fft.fft(y_complex)
    y_weighted = np.fft.ifft(y_fft * mask)

    h_fft = np.fft.fft(h_complex, axis=0)
    h_weighted = np.fft.ifft(h_fft * mask[:, None], axis=0)

    return h_weighted, y_weighted


def compute_condition_number(h_real: np.ndarray) -> float:
    return float(np.linalg.cond(h_real))


def standardize_features(h_real: np.ndarray) -> Tuple[np.ndarray, StandardizationParams, StandardScaler]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    h_scaled = scaler.fit_transform(h_real)
    params = StandardizationParams(mean=scaler.mean_.copy(), scale=scaler.scale_.copy())
    return h_scaled, params, scaler


def nmse_db_real(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    if y_pred.shape != y_true.shape:
        raise ValueError("Prediction and target must have the same shape.")
    mse = np.mean(np.sum((y_true - y_pred) ** 2, axis=1))
    energy = np.mean(np.sum(y_true ** 2, axis=1))
    return float(10.0 * np.log10(mse / energy))


def active_feature_indices(coef: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    if coef.ndim == 1:
        return np.flatnonzero(np.abs(coef) > tol)
    return np.flatnonzero(np.any(np.abs(coef) > tol, axis=0))


def denormalize_coefficients(
    coef_scaled: np.ndarray,
    intercept_scaled: np.ndarray,
    params: StandardizationParams,
) -> Tuple[np.ndarray, np.ndarray]:
    scale = params.scale
    mean = params.mean

    if coef_scaled.ndim == 1:
        coef_unscaled = coef_scaled / scale
        intercept_unscaled = intercept_scaled - np.dot(mean / scale, coef_scaled)
        return coef_unscaled, np.array([intercept_unscaled])

    coef_unscaled = coef_scaled / scale
    intercept_unscaled = intercept_scaled - (mean / scale) @ coef_scaled.T
    return coef_unscaled, intercept_unscaled


def build_group_indices(num_base_features: int, memory_depth_M: int) -> List[np.ndarray]:
    """
    Build group indices for real-projected H.
    Each group = Re(feature_k) across all taps + Im(feature_k) across all taps.
    """
    groups = []
    total_complex = num_base_features * memory_depth_M
    for k in range(num_base_features):
        re_cols = [m * num_base_features + k for m in range(memory_depth_M)]
        im_cols = [total_complex + m * num_base_features + k for m in range(memory_depth_M)]
        groups.append(np.array(re_cols + im_cols, dtype=int))
    return groups


def block_omp_sweep(
    h_train: np.ndarray,
    y_train: np.ndarray,
    h_val: np.ndarray,
    y_val: np.ndarray,
    groups: List[np.ndarray],
) -> SweepResult:
    n_groups = len(groups)
    max_groups = min(n_groups, h_train.shape[0] - 1)
    if max_groups < n_groups:
        print(
            f"Warning: BOMP group count capped at {max_groups} due to sample count. "
            f"Requested {n_groups}."
        )

    active_groups: List[int] = []
    feature_counts: List[int] = []
    nmse_db_values: List[float] = []
    active_indices: List[np.ndarray] = []
    params: List[Dict[str, float]] = []

    residual = y_train.copy()
    selected_cols: List[int] = []

    for k in range(1, max_groups + 1):
        best_group = None
        best_score = -np.inf
        for g_idx, g_cols in enumerate(groups):
            if g_idx in active_groups:
                continue
            corr = h_train[:, g_cols].T @ residual
            score = np.linalg.norm(corr)
            if score > best_score:
                best_score = score
                best_group = g_idx

        if best_group is None:
            break

        active_groups.append(best_group)
        selected_cols.extend(groups[best_group].tolist())

        h_sel = h_train[:, selected_cols]
        coef, _, _, _ = np.linalg.lstsq(h_sel, y_train, rcond=None)
        residual = y_train - h_sel @ coef

        y_pred = h_val[:, selected_cols] @ coef
        nmse_db = nmse_db_real(y_pred, y_val)
        feature_counts.append(len(active_groups))
        nmse_db_values.append(nmse_db)
        active_indices.append(np.array(active_groups, dtype=int))
        params.append({"active_groups": float(len(active_groups))})

    return SweepResult(
        feature_counts=feature_counts,
        nmse_db=nmse_db_values,
        active_indices=active_indices,
        model_params=params,
    )


def group_soft_threshold(w_group: np.ndarray, threshold: float) -> np.ndarray:
    norm = np.linalg.norm(w_group)
    if norm <= threshold:
        return np.zeros_like(w_group)
    return (1.0 - threshold / norm) * w_group


def group_lasso_fit(
    h_train: np.ndarray,
    y_train: np.ndarray,
    groups: List[np.ndarray],
    alpha: float,
    max_iter: int = 200,
    tol: float = 1e-6,
) -> np.ndarray:
    n_features = h_train.shape[1]
    n_outputs = y_train.shape[1]
    w = np.zeros((n_features, n_outputs))

    lipschitz = np.linalg.norm(h_train, 2) ** 2
    if lipschitz == 0:
        return w
    step = 1.0 / lipschitz

    for _ in range(max_iter):
        grad = h_train.T @ (h_train @ w - y_train)
        w_next = w - step * grad
        for g_cols in groups:
            w_group = w_next[g_cols, :]
            w_next[g_cols, :] = group_soft_threshold(w_group, step * alpha * np.sqrt(len(g_cols)))

        if np.linalg.norm(w_next - w) < tol:
            w = w_next
            break
        w = w_next

    return w


def group_lasso_sweep(
    h_train: np.ndarray,
    y_train: np.ndarray,
    h_val: np.ndarray,
    y_val: np.ndarray,
    groups: List[np.ndarray],
    alpha_min: float,
    alpha_max: float,
    alpha_points: int,
) -> SweepResult:
    alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max), alpha_points)

    feature_counts: List[int] = []
    nmse_db_values: List[float] = []
    active_indices: List[np.ndarray] = []
    params: List[Dict[str, float]] = []

    for alpha in alphas:
        w = group_lasso_fit(h_train, y_train, groups, alpha=alpha)
        y_pred = h_val @ w
        nmse_db = nmse_db_real(y_pred, y_val)
        active_groups = []
        for g_idx, g_cols in enumerate(groups):
            if np.linalg.norm(w[g_cols, :]) > 1e-8:
                active_groups.append(g_idx)
        feature_counts.append(len(active_groups))
        nmse_db_values.append(nmse_db)
        active_indices.append(np.array(active_groups, dtype=int))
        params.append({"alpha": float(alpha)})

    return SweepResult(
        feature_counts=feature_counts,
        nmse_db=nmse_db_values,
        active_indices=active_indices,
        model_params=params,
    )


def select_at_threshold(result: SweepResult, target_nmse_db: float) -> Tuple[int, np.ndarray, Dict[str, float]]:
    best_idx = None
    for idx, nmse_db in enumerate(result.nmse_db):
        if nmse_db <= target_nmse_db:
            best_idx = idx
            break

    if best_idx is None:
        best_idx = int(np.argmin(result.nmse_db))
        print(
            f"Warning: target NMSE {target_nmse_db} dB not reached. "
            f"Using best available NMSE {result.nmse_db[best_idx]:.3f} dB."
        )

    return (
        result.feature_counts[best_idx],
        result.active_indices[best_idx],
        result.model_params[best_idx],
    )


def plot_pareto(
    band_results_dict: Dict[str, SweepResult],
    output_path: str,
) -> None:
    """
    Plot BOMP Pareto frontier for all bands on a single figure.
    
    Args:
        band_results_dict: Dictionary mapping band name (y1, y2, y3) to SweepResult
        output_path: Path to save the figure
    """
    plt.figure(figsize=(10, 6))
    colors = {"y1": "blue", "y2": "green", "y3": "red"}
    markers = {"y1": "o", "y2": "s", "y3": "^"}
    
    for band_name, bomp_result in band_results_dict.items():
        color = colors.get(band_name, "black")
        marker = markers.get(band_name, "o")
        plt.plot(
            bomp_result.feature_counts,
            bomp_result.nmse_db,
            f"-{marker}",
            label=f"BOMP ({band_name})",
            color=color,
            markersize=6,
        )
    
    # Compute min/max feature counts across all bands
    all_feature_counts = [fc for result in band_results_dict.values() for fc in result.feature_counts]
    min_fc = min(all_feature_counts) if all_feature_counts else 0
    max_fc = max(all_feature_counts) if all_feature_counts else 1
    
    plt.xlabel("Active Groups", fontsize=12)
    plt.xticks(np.arange(min_fc, max_fc + 1, 2))
    plt.ylabel("NMSE (dB)", fontsize=12)
    plt.title("BOMP Pareto Frontier: NMSE vs Active Groups", fontsize=13)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def load_dataset_keys(path: str, h_key: str, y_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts the specific H matrix and target band from the multi-array archive."""
    if not path.endswith(".npz"):
        raise ValueError("Dataset must be the multi-array .npz archive.")
    data = np.load(path)
    if h_key not in data or y_key not in data:
        raise KeyError(f"Keys {h_key} and/or {y_key} not found in {path}. Available keys: {list(data.keys())}")
    return data[h_key], data[y_key]


def get_feature_names(memory_depth_M):
    """
    Returns a list of strings representing the mathematical identity of 
    each column in the real-projected H matrix.
    Must match the order in generate_dictionary_matrix_H() sequentially.
    """
    # 1. Define the base complex features (MUST match the order in your H-gen script)
    base_complex_names = [
        # 1. Intra-band features (12 total)
        "x1", "x1*|x1|^2", "x2", "x2*|x2|^2", "x3", "x3*|x3|^2",
        # 2. Cross-band envelope features (6 total)
        "x1*|x2|^2", "x1*|x3|^2", "x2*|x1|^2", "x2*|x3|^2", "x3*|x1|^2", "x3*|x2|^2",
        # 3. IMD3 Phase-Coherent Cross-terms (6 total)
        "x1^2*x2^*", "x2^2*x1^*", "x2^2*x3^*", "x3^2*x2^*", "x1^3*x3^*", "x3^3*x1^*",
        # 4. Tri-band IMD features (2 total)
        "x1*x2*x3^*", "x1^* * x2^2 * x3^*"
    ]
    P = len(base_complex_names)
    
    # 2. Reconstruct the full list following the Real-Projection logic
    full_names = []
    for m in range(memory_depth_M):
        # Your projection used: np.column_stack((h_complex.real, h_complex.imag))
        # This means for each tap, all Re columns come first, then all Im columns.
        for name in base_complex_names:
            full_names.append(f"Re({name}) [z^-{m}]")
        for name in base_complex_names:
            full_names.append(f"Im({name}) [z^-{m}]")
            
    return full_names


def get_base_feature_names() -> List[str]:
    return [
        "x1", "x1*|x1|^2", "x2", "x2*|x2|^2", "x3", "x3*|x3|^2",
        "x1*|x2|^2", "x1*|x3|^2", "x2*|x1|^2", "x2*|x3|^2", "x3*|x1|^2", "x3*|x2|^2",
        "x1^2*x2^*", "x2^2*x1^*", "x2^2*x3^*", "x3^2*x2^*", "x1^3*x3^*", "x3^3*x1^*",
        "x1*x2*x3^*", "x1^* * x2^2 * x3^*",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Basis selection for tri-band DPD dictionary matrix.")
    parser.add_argument("--dataset", required=True, help="Path to H_matrix_and_Targets_M4.npz")
    parser.add_argument("--val_ratio", type=float, default=0.25, help="Validation split ratio.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--nmse_threshold", type=float, default=-40.0, help="Target NMSE in dB.")
    parser.add_argument("--alpha_min", type=float, default=1e-5, help="Minimum LASSO alpha.")
    parser.add_argument("--alpha_max", type=float, default=1e-1, help="Maximum LASSO alpha.")
    parser.add_argument("--alpha_points", type=int, default=100, help="Number of LASSO alphas.")
    parser.add_argument("--fs", type=float, default=None, help="Sample rate for frequency weighting (Hz).")
    parser.add_argument(
        "--stopbands",
        default=None,
        help="Carrier stopbands in Hz, e.g. '-110e6,-90e6;90e6,110e6'.",
    )
    parser.add_argument("--output_dir", default="dpd_out/analysis/basis_selection", help="Output directory.")
    args = parser.parse_args()

    target_keys = ["y1", "y2", "y3"]
    band_results = {}
    bomp_results_per_band = {}  # Store BOMP sweep results for plotting
    unified_groups_set = set()
    base_names = get_base_feature_names()
    num_base_features = len(base_names)
    memory_depth_M = None
    groups = None

    print("="*60)
    print("MULTI-BAND BASIS SELECTION (BOMP)")
    print("="*60)

    for target_band in target_keys:
        print(f"\n--- Processing band: {target_band} ---")
        
        h_matrix, y_target = load_dataset_keys(
            path=args.dataset,
            h_key="H_matrix",
            y_key=target_band
        )

        if args.fs is not None and args.stopbands is not None:
            stopbands = parse_stopbands(args.stopbands)
            h_matrix, y_target = apply_frequency_weighting(h_matrix, y_target, args.fs, stopbands)

        h_real, y_real = project_complex_to_real_concat(h_matrix, y_target)

        if memory_depth_M is None:
            if h_matrix.shape[1] % num_base_features != 0:
                raise ValueError(
                    f"H_matrix columns ({h_matrix.shape[1]}) not divisible by base feature count ({num_base_features})."
                )
            memory_depth_M = h_matrix.shape[1] // num_base_features
            groups = build_group_indices(num_base_features, memory_depth_M)

        cond_number = compute_condition_number(h_real)
        print(f"  Condition number: {cond_number:.3e}")
        if cond_number > 1e4:
            print("  Warning: Condition number exceeds 1e4. Severe collinearity detected.")

        h_scaled, scaler_params, scaler = standardize_features(h_real)

        h_train, h_val, y_train, y_val = train_test_split(
            h_scaled,
            y_real,
            test_size=args.val_ratio,
            random_state=args.random_state,
            shuffle=False,
        )

        bomp_result = block_omp_sweep(h_train, y_train, h_val, y_val, groups)
        bomp_results_per_band[target_band] = bomp_result  # Store for plotting

        bomp_count, bomp_active, bomp_param = select_at_threshold(bomp_result, args.nmse_threshold)
        
        band_results[target_band] = {
            "selected_group_count": int(bomp_count),
            "active_group_indices": bomp_active.tolist(),
            "model_param": bomp_param,
        }
        
        unified_groups_set.update(bomp_active.tolist())
        
        print(f"  BOMP active groups @ {args.nmse_threshold} dB: {bomp_active.tolist()}")
        for idx in bomp_active:
            if idx < len(base_names):
                print(f"    Group {idx:2} : {base_names[idx]}")

    unified_groups_sorted = sorted(list(unified_groups_set))
    
    print("\n" + "="*60)
    print("HARDWARE MASTER BASIS SET (UNION ACROSS ALL BANDS)")
    print("="*60)
    print(f"Unified groups: {unified_groups_sorted}")
    print(f"Total unique groups: {len(unified_groups_sorted)}")
    print()
    for idx in unified_groups_sorted:
        if idx < len(base_names):
            print(f"  Group {idx:2} : {base_names[idx]} (all taps, Re+Im)")

    os.makedirs(args.output_dir, exist_ok=True)
    
    hardware_blueprint = {
        "nmse_threshold_db": args.nmse_threshold,
        "unified_active_groups": unified_groups_sorted,
        "total_unique_groups": len(unified_groups_sorted),
        "per_band_results": band_results,
        "group_names": {str(idx): base_names[idx] for idx in unified_groups_sorted},
    }

    blueprint_path = os.path.join(args.output_dir, "hardware_blueprint.json")
    with open(blueprint_path, "w", encoding="utf-8") as handle:
        json.dump(hardware_blueprint, handle, indent=2)

    print(f"\n✓ Hardware blueprint saved to: {blueprint_path}")
    
    # Generate Pareto plot for all bands
    pareto_path = os.path.join(args.output_dir, "pareto_nmse_vs_groups.png")
    plot_pareto(bomp_results_per_band, pareto_path)
    print(f"✓ Pareto plot saved to: {pareto_path}")


if __name__ == "__main__":
    main()
