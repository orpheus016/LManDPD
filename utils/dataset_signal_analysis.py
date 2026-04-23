import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils import metrics
from utils import util


def _read_iq_csv(path: str) -> np.ndarray:
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError(f"Expected at least 2 columns in {path}")
    return frame.iloc[:, :2].to_numpy(dtype=np.float64)


def _load_split_iq(dataset_dir: str, prefix: str) -> np.ndarray:
    paths = [
        os.path.join(dataset_dir, f"train_{prefix}.csv"),
        os.path.join(dataset_dir, f"val_{prefix}.csv"),
        os.path.join(dataset_dir, f"test_{prefix}.csv"),
    ]
    arrays = []
    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing split file: {path}")
        arrays.append(_read_iq_csv(path))
    return np.concatenate(arrays, axis=0)


def _load_spec(dataset_dir: str) -> dict:
    spec_path = os.path.join(dataset_dir, "spec.json")
    if not os.path.exists(spec_path):
        return {}
    with open(spec_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _normalized_psd(iq: np.ndarray, fs: float, nperseg: int, smooth_window: int):
    complex_signal = iq[:, 0] + 1j * iq[:, 1]
    freq, psd = metrics.power_spectrum(complex_signal.reshape(1, -1), fs=fs, nperseg=nperseg, axis=-1)
    psd_db = 10 * np.log10(psd / np.max(psd))

    if smooth_window > 1:
        psd_smoothed = metrics.moving_average(psd_db, smooth_window)
        trim_left = smooth_window // 2
        trim_right = smooth_window - trim_left - 1
        if trim_right == 0:
            freq_adj = freq[trim_left:]
        else:
            freq_adj = freq[trim_left:-trim_right]
        return freq_adj, psd_smoothed
    return freq, psd_db


def _moving_mean_by_bin(x: np.ndarray, y: np.ndarray, n_bins: int):
    x_max = np.max(x)
    if x_max <= 0:
        return x, y

    edges = np.linspace(0.0, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    indices = np.digitize(x, edges) - 1

    x_avg = []
    y_avg = []
    for idx in range(n_bins):
        mask = indices == idx
        if np.any(mask):
            x_avg.append(np.mean(x[mask]))
            y_avg.append(np.mean(y[mask]))
    return np.asarray(x_avg), np.asarray(y_avg)


def _wrap_to_180(angle_deg: np.ndarray):
    return (angle_deg + 180.0) % 360.0 - 180.0


def run_analysis(dataset_name: str, smooth_window: int = 10, n_bins: int = 120, max_const_points: int = 20000):
    dataset_dir = os.path.join("datasets", dataset_name)
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    spec = _load_spec(dataset_dir)
    fs = float(spec.get("input_signal_fs", 800e6))
    nperseg = int(spec.get("nperseg", 2560))

    iq_in = _load_split_iq(dataset_dir, "input")
    iq_out = _load_split_iq(dataset_dir, "output")

    n_samples = min(iq_in.shape[0], iq_out.shape[0])
    iq_in = iq_in[:n_samples]
    iq_out = iq_out[:n_samples]

    out_dir = os.path.join(dataset_dir, "signal_analysis")
    os.makedirs(out_dir, exist_ok=True)

    # Spectrum Analyzer style plot (normalized PSD)
    freq_in, psd_in = _normalized_psd(iq_in, fs=fs, nperseg=nperseg, smooth_window=smooth_window)
    freq_out, psd_out = _normalized_psd(iq_out, fs=fs, nperseg=nperseg, smooth_window=smooth_window)
    fig = plt.figure(figsize=(10, 6))
    plt.plot(freq_in / 1e6, psd_in, label="Input", linewidth=1.5)
    plt.plot(freq_out / 1e6, psd_out, label="Output", linestyle="--", linewidth=1.5)
    plt.title(f"Spectrum Analyzer View (Normalized PSD) - {dataset_name}")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Normalized PSD (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    spectrum_path = os.path.join(out_dir, "spectrum_analyzer_psd.png")
    plt.savefig(spectrum_path, dpi=150)
    plt.close(fig)

    # IQ Constellation plot
    plot_points = min(max_const_points, n_samples)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    axes[0].scatter(iq_in[:plot_points, 0], iq_in[:plot_points, 1], s=3, alpha=0.25, edgecolors="none")
    axes[0].set_title("Input IQ Constellation")
    axes[1].scatter(iq_out[:plot_points, 0], iq_out[:plot_points, 1], s=3, alpha=0.25, edgecolors="none")
    axes[1].set_title("Output IQ Constellation")
    for ax in axes:
        ax.set_xlabel("I")
        ax.set_ylabel("Q")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
    fig.suptitle(f"IQ Constellation - {dataset_name}")
    fig.tight_layout()
    constellation_path = os.path.join(out_dir, "iq_constellation.png")
    fig.savefig(constellation_path, dpi=150)
    plt.close(fig)

    # AM-AM plot
    amp_in = util.get_amplitude(iq_in)
    amp_out = util.get_amplitude(iq_out)
    x_mean, y_mean = _moving_mean_by_bin(amp_in, amp_out, n_bins=n_bins)
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(amp_in, amp_out, s=2, alpha=0.12, label="Samples")
    if x_mean.size > 0:
        plt.plot(x_mean, y_mean, color="red", linewidth=2, label="Binned mean")
    plt.title(f"AM-AM - {dataset_name}")
    plt.xlabel("|Input|")
    plt.ylabel("|Output|")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    amam_path = os.path.join(out_dir, "amam.png")
    plt.savefig(amam_path, dpi=150)
    plt.close(fig)

    # AM-PM plot
    phase_in = np.degrees(np.angle(iq_in[:, 0] + 1j * iq_in[:, 1]))
    phase_out = np.degrees(np.angle(iq_out[:, 0] + 1j * iq_out[:, 1]))
    phase_diff = _wrap_to_180(phase_out - phase_in)
    x_mean_pm, y_mean_pm = _moving_mean_by_bin(amp_in, phase_diff, n_bins=n_bins)
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(amp_in, phase_diff, s=2, alpha=0.12, label="Samples")
    if x_mean_pm.size > 0:
        plt.plot(x_mean_pm, y_mean_pm, color="red", linewidth=2, label="Binned mean")
    plt.title(f"AM-PM - {dataset_name}")
    plt.xlabel("|Input|")
    plt.ylabel("Phase(Output)-Phase(Input) [deg]")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    ampm_path = os.path.join(out_dir, "ampm.png")
    plt.savefig(ampm_path, dpi=150)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "n_samples": n_samples,
                "sample_rate_hz": fs,
                "nperseg": nperseg,
                "spectrum_plot": spectrum_path.replace("\\", "/"),
                "constellation_plot": constellation_path.replace("\\", "/"),
                "amam_plot": amam_path.replace("\\", "/"),
                "ampm_plot": ampm_path.replace("\\", "/"),
            }
        ]
    )
    summary_path = os.path.join(out_dir, "signal_analysis_summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"Saved: {spectrum_path}")
    print(f"Saved: {constellation_path}")
    print(f"Saved: {amam_path}")
    print(f"Saved: {ampm_path}")
    print(f"Saved: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate dataset signal analysis plots under datasets/<dataset>/signal_analysis")
    parser.add_argument("--dataset_name", required=True, help="Dataset folder name under datasets/")
    parser.add_argument("--smooth_window", type=int, default=10, help="Smoothing window for PSD")
    parser.add_argument("--n_bins", type=int, default=120, help="Number of bins for AM-AM/AM-PM trend")
    parser.add_argument("--max_const_points", type=int, default=20000, help="Max points per constellation subplot")
    args = parser.parse_args()

    run_analysis(
        dataset_name=args.dataset_name,
        smooth_window=args.smooth_window,
        n_bins=args.n_bins,
        max_const_points=args.max_const_points,
    )


if __name__ == "__main__":
    main()
