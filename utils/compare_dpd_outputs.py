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


def _infer_dataset_name(csv_path: str, dataset_name_arg: str = None) -> str:
    if dataset_name_arg:
        return dataset_name_arg
    parts = csv_path.replace("\\", "/").split("/")
    if len(parts) < 3:
        raise ValueError(f"Cannot infer dataset from path: {csv_path}")

    candidate = parts[-2]
    spec_path = os.path.join("datasets", candidate, "spec.json")
    if os.path.exists(spec_path):
        return candidate

    candidate = parts[-3]
    spec_path = os.path.join("datasets", candidate, "spec.json")
    if os.path.exists(spec_path):
        return candidate

    raise FileNotFoundError(
        "Dataset spec.json not found for inferred names. "
        "Provide --dataset_name explicitly."
    )


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


def _select_band(iq: np.ndarray, band: int) -> np.ndarray:
    if iq.shape[1] == 2:
        return iq
    if iq.shape[1] != 6:
        raise ValueError(f"Expected IQ input with 2 or 6 columns, got {iq.shape[1]}")
    if band not in (1, 2, 3):
        raise ValueError("band must be 1, 2, or 3")
    start = (band - 1) * 2
    return iq[:, start:start + 2]


def _load_iq_from_csv(csv_path: str):
    frame = pd.read_csv(csv_path)
    triband_in_cols = ["I1", "Q1", "I2", "Q2", "I3", "Q3"]
    triband_out_cols = ["I1_dpd", "Q1_dpd", "I2_dpd", "Q2_dpd", "I3_dpd", "Q3_dpd"]
    if all(col in frame.columns for col in triband_in_cols + triband_out_cols):
        original_iq_full = frame[triband_in_cols].to_numpy(dtype=np.float64)
        dpd_iq_full = frame[triband_out_cols].to_numpy(dtype=np.float64)
        return original_iq_full, dpd_iq_full, True

    required_cols = ["I", "Q", "I_dpd", "Q_dpd"]
    for col in required_cols:
        if col not in frame.columns:
            raise ValueError(f"Missing column '{col}' in {csv_path}")
    original_iq_full = frame[["I", "Q"]].to_numpy(dtype=np.float64)
    dpd_iq_full = frame[["I_dpd", "Q_dpd"]].to_numpy(dtype=np.float64)
    return original_iq_full, dpd_iq_full, False


def _get_fc_list(spec: dict) -> list:
    if "acquisition" in spec:
        acq = spec["acquisition"]
        if all(key in acq for key in ("fc1_hz", "fc2_hz", "fc3_hz")):
            return [float(acq["fc1_hz"]), float(acq["fc2_hz"]), float(acq["fc3_hz"])]
    if all(key in spec for key in ("fc1_hz", "fc2_hz", "fc3_hz")):
        return [float(spec["fc1_hz"]), float(spec["fc2_hz"]), float(spec["fc3_hz"])]
    raise KeyError("fc1_hz/fc2_hz/fc3_hz not found in spec.json")


def _wideband_psd_for_plot(iq_full: np.ndarray, fs: float, nperseg: int, smooth_window: int, fc_list: list):
    if iq_full.shape[1] != 6:
        raise ValueError("wideband PSD requires 6-column IQ input")
    n_total = iq_full.shape[0]
    t = np.arange(n_total) / fs
    wideband = np.zeros(n_total, dtype=np.complex128)
    for idx, fc in enumerate(fc_list):
        band = iq_full[:, idx * 2: idx * 2 + 2]
        complex_band = band[:, 0] + 1j * band[:, 1]
        wideband += complex_band * np.exp(1j * 2.0 * np.pi * fc * t)

    freq, psd = metrics.power_spectrum(wideband.reshape(1, -1), fs=fs, nperseg=nperseg, axis=-1)
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


def _find_pa_checkpoint(dataset_name: str, pa_backbone: str, pa_hidden_size: int, pa_num_layers: int, input_size: int):
    net_pa = model.CoreModel(
        input_size=input_size,
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
    input_size = dpd_iq.shape[1]
    ckpt = _find_pa_checkpoint(dataset_name, pa_backbone, pa_hidden_size, pa_num_layers, input_size)
    net_pa = model.CoreModel(
        input_size=input_size,
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


def _target_gains(dataset_name: str):
    x_train, y_train, _, _, _, _ = load_dataset(dataset_name=dataset_name)
    if x_train.shape[1] == 6:
        gains = []
        for band in (1, 2, 3):
            x_band = _select_band(x_train, band)
            y_band = _select_band(y_train, band)
            gains.append(float(set_target_gain(x_band, y_band)))
        return gains
    gain = float(set_target_gain(x_train, y_train))
    return [gain]


def _target_signal(input_iq: np.ndarray, target_gains: list, band: int):
    if input_iq.shape[1] == 6:
        gain = target_gains[band - 1]
    else:
        gain = target_gains[0]
    target = gain * input_iq
    return target, float(gain)


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


def _band_power(iq: np.ndarray) -> float:
    return float(np.mean(iq[:, 0] ** 2 + iq[:, 1] ** 2))


def _aggregate_metrics(per_band_results: list, aggregate: str, band_powers: list):
    if aggregate == "none":
        return None

    nmse_lin = []
    evm_lin = []
    aclr_l_lin = []
    aclr_r_lin = []
    aclr_avg_lin = []

    for result in per_band_results:
        nmse_lin.append(10 ** (result["nmse_db"] / 10.0))
        evm_lin.append(10 ** (result["evm_db"] / 20.0))
        aclr_l_lin.append(10 ** (result["aclr_left_db"] / 10.0))
        aclr_r_lin.append(10 ** (result["aclr_right_db"] / 10.0))
        aclr_avg_lin.append(10 ** (result["aclr_avg_db"] / 10.0))

    weights = None
    if aggregate == "weighted":
        total = sum(band_powers)
        if total == 0:
            weights = [1.0 / len(band_powers)] * len(band_powers)
        else:
            weights = [p / total for p in band_powers]

    def reduce(values):
        if aggregate == "max":
            return max(values)
        if weights is None:
            return float(np.mean(values))
        return float(np.sum(np.asarray(values) * np.asarray(weights)))

    nmse_db = 10 * np.log10(reduce(nmse_lin))
    evm_db = 20 * np.log10(reduce(evm_lin))
    aclr_left_db = 10 * np.log10(reduce(aclr_l_lin))
    aclr_right_db = 10 * np.log10(reduce(aclr_r_lin))
    aclr_avg_db = 10 * np.log10(reduce(aclr_avg_lin))

    return {
        "nmse_db": float(nmse_db),
        "evm_db": float(evm_db),
        "aclr_left_db": float(aclr_left_db),
        "aclr_right_db": float(aclr_right_db),
        "aclr_avg_db": float(aclr_avg_db),
    }


def _evaluate_file(
    csv_path: str,
    output_dir: str,
    dataset_name: str,
    band: int,
    pa_backbone: str,
    pa_hidden_size: int,
    pa_num_layers: int,
    psd_mode: str,
    wideband_fs: float,
    smooth_window: int = 10,
):
    spec = _load_spec(dataset_name)
    fs = spec["input_signal_fs"]
    bw_main_ch = spec["bw_main_ch"]
    n_sub_ch = spec["n_sub_ch"]
    nperseg = spec["nperseg"]

    original_iq_full, dpd_iq_full, is_triband = _load_iq_from_csv(csv_path)
    original_iq = _select_band(original_iq_full, band)
    dpd_iq = _select_band(dpd_iq_full, band)

    pa_after_dpd_full, pa_ckpt = _pa_output_after_dpd(
        dataset_name=dataset_name,
        dpd_iq=dpd_iq_full,
        pa_backbone=pa_backbone,
        pa_hidden_size=pa_hidden_size,
        pa_num_layers=pa_num_layers,
    )
    target_gains = _target_gains(dataset_name)
    target_iq, target_gain = _target_signal(input_iq=original_iq, target_gains=target_gains, band=band)

    pa_after_dpd_iq = _select_band(pa_after_dpd_full, band)

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

    if psd_mode == "wideband" and is_triband:
        fc_list = _get_fc_list(spec)
        psd_fs = wideband_fs if wideband_fs is not None else fs
        target_iq_full = original_iq_full.copy()
        for idx in range(3):
            band_gain = target_gains[idx]
            target_iq_full[:, idx * 2: idx * 2 + 2] *= band_gain
        freq_ref, psd_ref = _wideband_psd_for_plot(target_iq_full, fs=psd_fs, nperseg=nperseg, smooth_window=smooth_window, fc_list=fc_list)
        freq_dpd, psd_dpd = _wideband_psd_for_plot(pa_after_dpd_full, fs=psd_fs, nperseg=nperseg, smooth_window=smooth_window, fc_list=fc_list)
        psd_label = "wideband"
    else:
        freq_ref, psd_ref = _psd_for_plot(target_iq, fs=fs, nperseg=nperseg, smooth_window=smooth_window)
        freq_dpd, psd_dpd = _psd_for_plot(pa_after_dpd_iq, fs=fs, nperseg=nperseg, smooth_window=smooth_window)
        psd_label = f"band{band}"

    fig = plt.figure(figsize=(10, 6))
    plt.plot(freq_ref / 1e6, psd_ref, label="Target (gain * input)", linewidth=1.5)
    plt.plot(freq_dpd / 1e6, psd_dpd, label="PA output after DPD", linestyle="--", linewidth=1.5)
    plt.title(f"Normalized PSD ({psd_label}): {os.path.basename(csv_path)}")
    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Normalized PSD (dB)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    plot_path = os.path.join(output_dir, f"{dataset_name}__{base_name}__psd_{psd_label}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

    return {
        "dataset": dataset_name,
        "band": int(band),
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
    parser.add_argument(
        "--dataset_name",
        default=None,
        help="Dataset name under ./datasets (used to load spec.json and PA checkpoint).",
    )
    parser.add_argument(
        "--band",
        default="1",
        choices=["1", "2", "3", "all"],
        help="Which band to evaluate for triband CSVs. Use 'all' to evaluate bands 1..3.",
    )
    parser.add_argument(
        "--aggregate",
        default="mean",
        choices=["none", "mean", "max", "weighted"],
        help="Aggregate tri-band metrics across bands when --band all is used.",
    )
    parser.add_argument(
        "--psd",
        default="band",
        choices=["band", "wideband"],
        help="PSD plot mode: per-band baseband or reconstructed wideband.",
    )
    parser.add_argument(
        "--wideband_fs",
        type=float,
        default=None,
        help="Sample rate for wideband PSD (Hz). Defaults to spec input_signal_fs.",
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
        dataset_name = _infer_dataset_name(csv_path, args.dataset_name)
        if args.band == "all":
            bands = [1, 2, 3]
        else:
            bands = [int(args.band)]
        per_band_results = []
        band_powers = []
        for band in bands:
            result = _evaluate_file(
                csv_path,
                output_dir=args.output_dir,
                dataset_name=dataset_name,
                band=band,
                pa_backbone=args.PA_backbone,
                pa_hidden_size=args.PA_hidden_size,
                pa_num_layers=args.PA_num_layers,
                psd_mode=args.psd,
                wideband_fs=args.wideband_fs,
                smooth_window=args.smooth_window,
            )
            results.append(result)
            per_band_results.append(result)
            if result["band"] in (1, 2, 3):
                band_powers.append(_band_power(_select_band(_load_iq_from_csv(csv_path)[0], result["band"])))
            print(
                f"[{result['dataset']}|band{result['band']}] {os.path.basename(csv_path)} | "
                f"NMSE={result['nmse_db']:.3f} dB, EVM={result['evm_db']:.3f} dB, "
                f"ACLR(avg)={result['aclr_avg_db']:.3f} dB"
            )

        if args.band == "all" and args.aggregate != "none":
            agg = _aggregate_metrics(per_band_results, args.aggregate, band_powers)
            if agg:
                agg_result = dict(per_band_results[0])
                agg_result.update(agg)
                agg_result["band"] = "all"
                agg_result["metric_scope"] = f"aggregate_{args.aggregate}"
                results.append(agg_result)
                print(
                    f"[{agg_result['dataset']}|aggregate_{args.aggregate}] {os.path.basename(csv_path)} | "
                    f"NMSE={agg_result['nmse_db']:.3f} dB, EVM={agg_result['evm_db']:.3f} dB, "
                    f"ACLR(avg)={agg_result['aclr_avg_db']:.3f} dB"
                )

    summary = pd.DataFrame(results)
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "dpd_metrics_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
