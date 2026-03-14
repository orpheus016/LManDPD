import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import models as model
from modules.data_collector import load_dataset
from utils.util import set_target_gain
from utils import metrics


def _load_spec(dataset_name: str):
    spec_path = os.path.join("datasets", dataset_name, "spec.json")
    with open(spec_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _segment_iq(iq: np.ndarray, nperseg: int):
    n_total = iq.shape[0]
    segments = []
    for start in range(0, n_total, nperseg):
        segment = iq[start:start + nperseg]
        if segment.shape[0] < nperseg:
            pad = np.zeros((nperseg - segment.shape[0], 2), dtype=segment.dtype)
            segment = np.vstack((segment, pad))
        segments.append(segment)
    if len(segments) == 0:
        raise ValueError(f"Signal length {n_total} is too short for segmentation.")
    return np.asarray(segments)


def _find_pa_checkpoint(dataset_name: str, pa_backbone: str, pa_hidden_size: int, pa_num_layers: int):
    net_pa = model.CoreModel(
        input_size=2,
        hidden_size=pa_hidden_size,
        num_layers=pa_num_layers,
        backbone_type=pa_backbone,
    )
    n_params = sum(parameter.numel() for parameter in net_pa.parameters())
    pa_model_id = f"PA_S_0_M_{pa_backbone.upper()}_H_{pa_hidden_size}_F_200_P_{n_params}"
    ckpt = os.path.join("save", dataset_name, "train_pa", pa_model_id + ".pt")
    if os.path.exists(ckpt):
        return ckpt

    fallback = sorted(glob.glob(os.path.join("save", dataset_name, "train_pa", "PA_*.pt")))
    if len(fallback) == 1:
        return fallback[0]
    if len(fallback) == 0:
        raise FileNotFoundError(f"No PA checkpoint found in save/{dataset_name}/train_pa")
    raise FileNotFoundError(
        f"Expected checkpoint {ckpt} not found. Multiple candidates exist: {fallback}. "
        f"Provide matching PA args."
    )


def _pa_output_after_dpd(dataset_name: str, dpd_iq: np.ndarray, pa_backbone: str, pa_hidden_size: int, pa_num_layers: int):
    ckpt = _find_pa_checkpoint(dataset_name, pa_backbone, pa_hidden_size, pa_num_layers)
    net_pa = model.CoreModel(
        input_size=2,
        hidden_size=pa_hidden_size,
        num_layers=pa_num_layers,
        backbone_type=pa_backbone,
    )
    state = torch.load(ckpt, map_location="cpu")
    net_pa.load_state_dict(state)
    net_pa.eval()

    with torch.no_grad():
        tensor_in = torch.tensor(dpd_iq, dtype=torch.float32).unsqueeze(0)
        tensor_out = net_pa(tensor_in)
        pa_out = tensor_out.squeeze(0).cpu().numpy()

    return pa_out, ckpt


def _target_signal(dataset_name: str, input_iq: np.ndarray):
    x_train, y_train, _, _, _, _ = load_dataset(dataset_name=dataset_name)
    target_gain = set_target_gain(x_train, y_train)
    target = target_gain * input_iq
    return target, float(target_gain)


def _psd_for_plot(iq: np.ndarray, fs: float, nperseg: int, smooth_window: int = 10):
    complex_signal = iq[:, 0] + 1j * iq[:, 1]
    freq, psd = metrics.power_spectrum(complex_signal.reshape(1, -1), fs=fs, nperseg=nperseg, axis=-1)
    psd_norm = 10 * np.log10(psd / np.max(psd))

    if smooth_window > 1:
        psd_smoothed = metrics.moving_average(psd_norm, smooth_window)
        trim_left = smooth_window // 2
        trim_right = smooth_window - trim_left - 1
        if trim_right == 0:
            freq_adj = freq[trim_left:]
        else:
            freq_adj = freq[trim_left:-trim_right]
        return freq_adj, psd_smoothed

    return freq, psd_norm


def _evaluate_file(
    csv_path: str,
    output_dir: str,
    pa_backbone: str,
    pa_hidden_size: int,
    pa_num_layers: int,
    smooth_window: int = 10,
):
    parts = csv_path.replace("\\", "/").split("/")
    if len(parts) < 3:
        raise ValueError(f"Cannot infer dataset from path: {csv_path}")
    dataset_name = parts[-2]

    spec = _load_spec(dataset_name)
    fs = spec["input_signal_fs"]
    bw_main_ch = spec["bw_main_ch"]
    n_sub_ch = spec["n_sub_ch"]
    nperseg = spec["nperseg"]

    frame = pd.read_csv(csv_path)
    required_cols = ["I", "Q", "I_dpd", "Q_dpd"]
    for col in required_cols:
        if col not in frame.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path}")

    original_iq = frame[["I", "Q"]].to_numpy(dtype=np.float64)
    dpd_iq = frame[["I_dpd", "Q_dpd"]].to_numpy(dtype=np.float64)

    pa_after_dpd_iq, pa_ckpt = _pa_output_after_dpd(
        dataset_name=dataset_name,
        dpd_iq=dpd_iq,
        pa_backbone=pa_backbone,
        pa_hidden_size=pa_hidden_size,
        pa_num_layers=pa_num_layers,
    )
    target_iq, target_gain = _target_signal(dataset_name=dataset_name, input_iq=original_iq)

    pred_segments = _segment_iq(pa_after_dpd_iq, nperseg=nperseg)
    gt_segments = _segment_iq(target_iq, nperseg=nperseg)

    nmse_db = float(metrics.NMSE(pred_segments, gt_segments))
    evm_db = float(
        metrics.EVM(
            pred_segments,
            gt_segments,
            sample_rate=fs,
            bw_main_ch=bw_main_ch,
            n_sub_ch=n_sub_ch,
            nperseg=nperseg,
        )
    )
    aclr_l_db, aclr_r_db = metrics.ACLR(
        pred_segments,
        fs=fs,
        nperseg=nperseg,
        bw_main_ch=bw_main_ch,
        n_sub_ch=n_sub_ch,
    )
    aclr_avg_db = float((aclr_l_db + aclr_r_db) / 2.0)

    freq_ref, psd_ref = _psd_for_plot(target_iq, fs=fs, nperseg=nperseg, smooth_window=smooth_window)
    freq_dpd, psd_dpd = _psd_for_plot(pa_after_dpd_iq, fs=fs, nperseg=nperseg, smooth_window=smooth_window)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(freq_ref / 1e6, psd_ref, label="Target (gain * input)", linewidth=1.5)
    plt.plot(freq_dpd / 1e6, psd_dpd, label="PA output after DPD", linestyle="--", linewidth=1.5)
    plt.title(f"Normalized PSD: {os.path.basename(csv_path)}")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Normalized PSD (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    plot_path = os.path.join(output_dir, f"{dataset_name}__{base_name}__psd.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {
        "dataset": dataset_name,
        "file": csv_path.replace("\\", "/"),
        "metric_scope": "pa_after_dpd_vs_target",
        "pa_checkpoint": pa_ckpt.replace("\\", "/"),
        "target_gain": target_gain,
        "n_samples": int(min(original_iq.shape[0], dpd_iq.shape[0])),
        "nperseg": int(nperseg),
        "n_segments_used": int(pred_segments.shape[0]),
        "sample_rate_hz": float(fs),
        "bw_main_ch_hz": float(bw_main_ch),
        "n_sub_ch": int(n_sub_ch),
        "nmse_db": nmse_db,
        "evm_db": evm_db,
        "aclr_left_db": float(aclr_l_db),
        "aclr_right_db": float(aclr_r_db),
        "aclr_avg_db": aclr_avg_db,
        "psd_plot": plot_path.replace("\\", "/"),
    }


def _expand_inputs(inputs):
    files = []
    for item in inputs:
        matches = sorted(glob.glob(item))
        if matches:
            files.extend(matches)
        elif os.path.isfile(item):
            files.append(item)
    unique_files = sorted(set(files))
    return [f for f in unique_files if f.lower().endswith(".csv")]


def main():
    parser = argparse.ArgumentParser(description="Compare DPD output CSVs with OpenDPD utils metrics.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input CSV file paths or glob patterns.",
    )
    parser.add_argument(
        "--output_dir",
        default="dpd_out/analysis",
        help="Directory to store plots and summary CSV.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=10,
        help="PSD smoothing window size.",
    )
    parser.add_argument("--PA_backbone", default="dgru", help="PA backbone used for PA checkpoint loading.")
    parser.add_argument("--PA_hidden_size", type=int, default=23, help="PA hidden size.")
    parser.add_argument("--PA_num_layers", type=int, default=1, help="PA number of layers.")
    args = parser.parse_args()

    files = _expand_inputs(args.inputs)
    if not files:
        raise ValueError("No CSV files found from --inputs.")

    results = []
    for csv_path in files:
        result = _evaluate_file(
            csv_path,
            output_dir=args.output_dir,
            pa_backbone=args.PA_backbone,
            pa_hidden_size=args.PA_hidden_size,
            pa_num_layers=args.PA_num_layers,
            smooth_window=args.smooth_window,
        )
        results.append(result)
        print(
            f"[{result['dataset']}] {os.path.basename(csv_path)} | "
            f"NMSE={result['nmse_db']:.3f} dB, EVM={result['evm_db']:.3f} dB, "
            f"ACLR(avg)={result['aclr_avg_db']:.3f} dB"
        )

    summary = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "dpd_metrics_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
