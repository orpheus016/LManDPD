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
    bomp_result: SweepResult,
    group_lasso_result: SweepResult,
    output_path: str,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(bomp_result.feature_counts, bomp_result.nmse_db, "-o", label="BOMP")
    plt.plot(group_lasso_result.feature_counts, group_lasso_result.nmse_db, "-s", label="Group LASSO")
    plt.xlabel("Active Groups")
    plt.ylabel("NMSE (dB)")
    plt.grid(True)
    plt.legend()
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
        "|x1|", "|x2|", "|x3|", "|x1|^3", "|x2|^3", "|x3|^3",
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
        "|x1|", "|x2|", "|x3|", "|x1|^3", "|x2|^3", "|x3|^3",
        "x1*|x2|^2", "x1*|x3|^2", "x2*|x1|^2", "x2*|x3|^2", "x3*|x1|^2", "x3*|x2|^2",
        "x1^2*x2^*", "x2^2*x1^*", "x2^2*x3^*", "x3^2*x2^*", "x1^3*x3^*", "x3^3*x1^*",
        "x1*x2*x3^*", "x1^* * x2^2 * x3^*",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Basis selection for tri-band DPD dictionary matrix.")
    parser.add_argument("--dataset", required=True, help="Path to H_matrix_and_Targets_M4.npz")
    parser.add_argument("--target_band", required=True, choices=["y1", "y2", "y3"], help="Which PA output band to optimize against.")
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

    h_matrix, y_target = load_dataset_keys(
        path=args.dataset,
        h_key="H_matrix",
        y_key=args.target_band
    )

    if args.fs is not None and args.stopbands is not None:
        stopbands = parse_stopbands(args.stopbands)
        h_matrix, y_target = apply_frequency_weighting(h_matrix, y_target, args.fs, stopbands)

    h_real, y_real = project_complex_to_real_concat(h_matrix, y_target)

    base_names = get_base_feature_names()
    num_base_features = len(base_names)
    if h_matrix.shape[1] % num_base_features != 0:
        raise ValueError(
            f"H_matrix columns ({h_matrix.shape[1]}) not divisible by base feature count ({num_base_features})."
        )
    memory_depth_M = h_matrix.shape[1] // num_base_features
    groups = build_group_indices(num_base_features, memory_depth_M)

    cond_number = compute_condition_number(h_real)
    print(f"Condition number (H_real): {cond_number:.3e}")
    if cond_number > 1e4:
        print("Warning: Condition number exceeds 1e4. Severe collinearity detected.")

    h_scaled, scaler_params, scaler = standardize_features(h_real)
    print("Scaler mean/scale saved for weight denormalization.")

    h_train, h_val, y_train, y_val = train_test_split(
        h_scaled,
        y_real,
        test_size=args.val_ratio,
        random_state=args.random_state,
        shuffle=False, 
    )

    bomp_result = block_omp_sweep(h_train, y_train, h_val, y_val, groups)
    group_lasso_result = group_lasso_sweep(
        h_train,
        y_train,
        h_val,
        y_val,
        groups,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_points=args.alpha_points,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, "pareto_nmse_vs_groups.png")
    plot_pareto(bomp_result, group_lasso_result, plot_path)

    bomp_count, bomp_active, bomp_param = select_at_threshold(bomp_result, args.nmse_threshold)
    gl_count, gl_active, gl_param = select_at_threshold(group_lasso_result, args.nmse_threshold)

    summary = {
        "condition_number": cond_number,
        "nmse_threshold_db": args.nmse_threshold,
        "bomp": {
            "selected_group_count": int(bomp_count),
            "active_group_indices": bomp_active.tolist(),
            "model_param": bomp_param,
        },
        "group_lasso": {
            "selected_group_count": int(gl_count),
            "active_group_indices": gl_active.tolist(),
            "model_param": gl_param,
        },
        "scaler": {
            "mean": scaler_params.mean.tolist(),
            "scale": scaler_params.scale.tolist(),
        },
        "pareto_plot": plot_path.replace("\\", "/"),
    }

    summary_path = os.path.join(args.output_dir, "basis_selection_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"BOMP active groups @ {args.nmse_threshold} dB: {bomp_active.tolist()}")
    print(f"Group LASSO active groups @ {args.nmse_threshold} dB: {gl_active.tolist()}")
    print(f"Summary saved to: {summary_path}")

    # After computing omp_active
    print("\n" + "="*30)
    print("HARDWARE FEATURE BLUEPRINT (BOMP)")
    print("="*30)
    for idx in bomp_active:
        if idx < len(base_names):
            print(f"Group {idx:2} : {base_names[idx]} (all taps, Re+Im)")
        else:
            print(f"Group {idx:2} : [OUT OF BOUNDS - Check P and M alignment]")


if __name__ == "__main__":
    main()
