import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import metrics
from utils.util import set_target_gain


def _load_iq_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    if not {"I", "Q"}.issubset(df.columns):
        raise ValueError(f"Expected columns I,Q in {path}")
    return df[["I", "Q"]].to_numpy(dtype=np.float64)


def _segment_iq(iq: np.ndarray, nperseg: int) -> np.ndarray:
    segments = []
    for start in range(0, iq.shape[0], nperseg):
        seg = iq[start:start + nperseg]
        if seg.shape[0] < nperseg:
            pad = np.zeros((nperseg - seg.shape[0], 2), dtype=iq.dtype)
            seg = np.vstack((seg, pad))
        segments.append(seg)
    return np.asarray(segments)


def _compute_metrics(x_iq: np.ndarray, y_iq: np.ndarray, spec: dict) -> dict:
    fs = int(spec["input_signal_fs"])
    nperseg = int(spec["nperseg"])
    bw_main_ch = float(spec["bw_main_ch"])
    n_sub_ch = int(spec["n_sub_ch"])

    gain = float(set_target_gain(x_iq, y_iq))
    y_target = gain * x_iq

    y_seg = _segment_iq(y_iq, nperseg)
    t_seg = _segment_iq(y_target, nperseg)

    nmse_db = float(metrics.NMSE(y_seg, t_seg))
    evm_db = float(metrics.EVM(y_seg, t_seg, sample_rate=fs, bw_main_ch=bw_main_ch, n_sub_ch=n_sub_ch, nperseg=nperseg))
    aclr_l_db, aclr_r_db = metrics.ACLR(y_seg, fs=fs, nperseg=nperseg, bw_main_ch=bw_main_ch, n_sub_ch=n_sub_ch)

    return {
        "target_gain": gain,
        "nmse_db": nmse_db,
        "evm_db": evm_db,
        "aclr_left_db": float(aclr_l_db),
        "aclr_right_db": float(aclr_r_db),
        "aclr_avg_db": float((aclr_l_db + aclr_r_db) / 2.0),
    }


def _plot_results(x_iq: np.ndarray, y_iq: np.ndarray, spec: dict, max_points: int = 60000) -> None:
    fs = int(spec["input_signal_fs"])
    nperseg = int(spec["nperseg"])
    bw = float(spec["bw_main_ch"])

    x_c = x_iq[:, 0] + 1j * x_iq[:, 1]
    y_c = y_iq[:, 0] + 1j * y_iq[:, 1]

    freq_x, ps_x = metrics.power_spectrum(x_c.reshape(1, -1), fs=fs, nperseg=nperseg, axis=-1)
    freq_y, ps_y = metrics.power_spectrum(y_c.reshape(1, -1), fs=fs, nperseg=nperseg, axis=-1)

    ps_x_db = 10 * np.log10(ps_x / np.max(ps_x))
    ps_y_db = 10 * np.log10(ps_y / np.max(ps_y))

    plt.figure(figsize=(10, 4))
    plt.plot(freq_x / 1e6, ps_x_db, label="Input")
    plt.plot(freq_y / 1e6, ps_y_db, label="PA Output")
    plt.xlim((-(bw / 2) * 2.2) / 1e6, ((bw / 2) * 2.2) / 1e6)
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Normalized PSD (dB)")
    plt.title("Spectrum")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    n = min(max_points, len(x_c), len(y_c))
    step = max(1, n // max_points)
    idx = np.arange(0, n, step)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(np.real(y_c[idx]), np.imag(y_c[idx]), ".", markersize=1)
    plt.title("IQ Constellation (PA Output)")
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.axis("equal")
    plt.grid(True)

    g = (np.vdot(x_c, y_c) / (np.vdot(x_c, x_c) + 1e-15))
    y_eq = y_c / (g + 1e-15)
    amp = np.abs(x_c)
    phase_deg = np.rad2deg(np.angle(y_eq * np.conj(x_c)))

    low = amp < (0.1 * np.max(amp))
    if np.any(low):
        phase_deg = phase_deg - np.mean(phase_deg[low])

    plt.subplot(1, 2, 2)
    plt.plot(amp[idx], np.abs(y_eq[idx]), ".", markersize=1, label="AM-AM")
    plt.plot(amp[idx], phase_deg[idx] / 180.0, ".", markersize=1, label="AM-PM/180")
    plt.title("AM-AM / AM-PM")
    plt.xlabel("|x|")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="View PA dataset metrics and plots using repo utils.")
    parser.add_argument("--dataset_dir", required=True, help="Path to dataset folder with split CSV files and spec.json")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="Which split to visualize")
    parser.add_argument("--max_points", type=int, default=60000, help="Max points for scatter plots")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    spec = json.loads((dataset_dir / "spec.json").read_text(encoding="utf-8"))

    x_iq = _load_iq_csv(dataset_dir / f"{args.split}_input.csv")
    y_iq = _load_iq_csv(dataset_dir / f"{args.split}_output.csv")

    m = _compute_metrics(x_iq, y_iq, spec)
    print(json.dumps(m, indent=2))

    _plot_results(x_iq, y_iq, spec, max_points=args.max_points)


if __name__ == "__main__":
    main()
