import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso
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


def omp_sweep(
    h_train: np.ndarray,
    y_train: np.ndarray,
    h_val: np.ndarray,
    y_val: np.ndarray,
) -> SweepResult:
    n_features = h_train.shape[1]
    max_nonzero = min(n_features, h_train.shape[0] - 1)
    if max_nonzero < n_features:
        print(
            f"Warning: OMP n_nonzero_coefs capped at {max_nonzero} due to sample count. "
            f"Requested {n_features}."
        )

    feature_counts: List[int] = []
    nmse_db_values: List[float] = []
    active_indices: List[np.ndarray] = []
    params: List[Dict[str, float]] = []

    for k in range(1, max_nonzero + 1):
        model = OrthogonalMatchingPursuit(n_nonzero_coefs=k, fit_intercept=True)
        model.fit(h_train, y_train)
        y_pred = model.predict(h_val)
        nmse_db = nmse_db_real(y_pred, y_val)
        feature_counts.append(k)
        nmse_db_values.append(nmse_db)
        active_indices.append(active_feature_indices(model.coef_))
        params.append({"n_nonzero_coefs": float(k)})

    return SweepResult(
        feature_counts=feature_counts,
        nmse_db=nmse_db_values,
        active_indices=active_indices,
        model_params=params,
    )


def lasso_sweep(
    h_train: np.ndarray,
    y_train: np.ndarray,
    h_val: np.ndarray,
    y_val: np.ndarray,
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
        model = Lasso(alpha=alpha, fit_intercept=True, max_iter=10000)
        model.fit(h_train, y_train)
        y_pred = model.predict(h_val)
        nmse_db = nmse_db_real(y_pred, y_val)
        active = active_feature_indices(model.coef_)
        feature_counts.append(int(active.shape[0]))
        nmse_db_values.append(nmse_db)
        active_indices.append(active)
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
    omp_result: SweepResult,
    lasso_result: SweepResult,
    output_path: str,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.plot(omp_result.feature_counts, omp_result.nmse_db, "-o", label="OMP")
    plt.plot(lasso_result.feature_counts, lasso_result.nmse_db, "-s", label="LASSO")
    plt.xlabel("Active Features")
    plt.ylabel("NMSE (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def load_array(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    if path.endswith(".npz"):
        data = np.load(path)
        if "arr_0" in data:
            return data["arr_0"]
        raise ValueError("NPZ file must contain 'arr_0' or be provided as .npy.")
    raise ValueError("Only .npy or .npz files are supported.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Basis selection for tri-band DPD dictionary matrix.")
    parser.add_argument("--h_matrix", required=True, help="Path to complex H_matrix (.npy or .npz).")
    parser.add_argument("--y_target", required=True, help="Path to complex Y_target (.npy or .npz).")
    parser.add_argument("--val_ratio", type=float, default=0.25, help="Validation split ratio.")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for split.")
    parser.add_argument("--nmse_threshold", type=float, default=-40.0, help="Target NMSE in dB.")
    parser.add_argument("--alpha_min", type=float, default=1e-5, help="Minimum LASSO alpha.")
    parser.add_argument("--alpha_max", type=float, default=1e-1, help="Maximum LASSO alpha.")
    parser.add_argument("--alpha_points", type=int, default=100, help="Number of LASSO alphas.")
    parser.add_argument("--output_dir", default="dpd_out/analysis/basis_selection", help="Output directory.")
    args = parser.parse_args()

    h_matrix = load_array(args.h_matrix)
    y_target = load_array(args.y_target)

    h_real, y_real = project_complex_to_real_concat(h_matrix, y_target)

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
        shuffle=True,
    )

    omp_result = omp_sweep(h_train, y_train, h_val, y_val)
    lasso_result = lasso_sweep(
        h_train,
        y_train,
        h_val,
        y_val,
        alpha_min=args.alpha_min,
        alpha_max=args.alpha_max,
        alpha_points=args.alpha_points,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    plot_path = os.path.join(args.output_dir, "pareto_nmse_vs_features.png")
    plot_pareto(omp_result, lasso_result, plot_path)

    omp_count, omp_active, omp_param = select_at_threshold(omp_result, args.nmse_threshold)
    lasso_count, lasso_active, lasso_param = select_at_threshold(lasso_result, args.nmse_threshold)

    summary = {
        "condition_number": cond_number,
        "nmse_threshold_db": args.nmse_threshold,
        "omp": {
            "selected_feature_count": int(omp_count),
            "active_feature_indices": omp_active.tolist(),
            "model_param": omp_param,
        },
        "lasso": {
            "selected_feature_count": int(lasso_count),
            "active_feature_indices": lasso_active.tolist(),
            "model_param": lasso_param,
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

    print(f"OMP active features @ {args.nmse_threshold} dB: {omp_active.tolist()}")
    print(f"LASSO active features @ {args.nmse_threshold} dB: {lasso_active.tolist()}")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
