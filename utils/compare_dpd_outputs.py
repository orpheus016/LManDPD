import argparse
import glob
import json
import os
import sys
import scipy.signal
import math

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


def _wideband_psd_for_plot(iq_full: np.ndarray, baseband_fs: float, wideband_fs: float, nperseg: int, smooth_window: int, fc_list: list):
    if iq_full.shape[1] != 6:
        raise ValueError("wideband PSD requires 6-column IQ input")
    
    n_total_baseband = iq_full.shape[0]
    
    # FIX 1: Map absolute carrier frequencies to relative baseband offsets 
    # to prevent catastrophic aliasing when mixing.
    fc_center = (max(fc_list) + min(fc_list)) / 2.0
    fc_offsets = [fc - fc_center for fc in fc_list]
    
    # FIX 2: Compute exact integer up/down ratios for polyphase resampling
    # (Using 1000 Hz scaling as telecom sample rates are usually exact in kHz)
    up = int(round(wideband_fs / 1000))
    down = int(round(baseband_fs / 1000))
    g = math.gcd(up, down)
    up //= g
    down //= g
    
    n_total_wideband = int(n_total_baseband * (up / down))
    t = np.arange(n_total_wideband) / wideband_fs
    wideband = np.zeros(n_total_wideband, dtype=np.complex128)
    
    for idx, offset in enumerate(fc_offsets):
        band = iq_full[:, idx * 2: idx * 2 + 2]
        complex_band = band[:, 0] + 1j * band[:, 1]
        
        # FIX 3: Use resample_poly instead of resample. 
        # Standard resample uses FFT and assumes periodic boundaries, 
        # causing edge artifacts that severely distort the PSD noise floor.
        complex_band_up = scipy.signal.resample_poly(complex_band, up, down)
        
        # Ensure lengths match due to slight polyphase filter edge trimming
        min_len = min(len(complex_band_up), len(wideband))
        
        # Shift using the RELATIVE offset
        wideband[:min_len] += complex_band_up[:min_len] * np.exp(1j * 2.0 * np.pi * offset * t[:min_len])

    # FIX 4: Scale nperseg so the Welch frequency bins maintain the same 
    # Hz/bin resolution as your single-band plots.
    nperseg_wideband = int(nperseg * (wideband_fs / baseband_fs))

    freq, psd = metrics.power_spectrum(wideband.reshape(1, -1), fs=wideband_fs, nperseg=nperseg_wideband, axis=-1)
    psd_norm = 10 * np.log10(psd / np.max(psd))

    freq = np.fft.fftshift(freq)
    psd_norm = np.fft.fftshift(psd_norm)

    if smooth_window > 1:
        psd_smoothed = metrics.moving_average(psd_norm, smooth_window)
        trim_left = smooth_window // 2
        trim_right = smooth_window - trim_left - 1
        if trim_right == 0:
            freq_adj = freq[trim_left:]
        else:
            freq_adj = freq[trim_left:-trim_right]
        return freq_adj, psd_smoothed, wideband

    return freq, psd_norm, wideband

def extract_constellation_symbols(iq_complex: np.ndarray, spec: dict):
    """
    Extracts frequency-domain constellation symbols from whatever time-domain 
    OFDM IQ data is available, symbol-by-symbol.
    """
    acq = spec.get("acquisition", {})
    nfft = int(acq.get("nfft", 1024))
    cp_len = int(acq.get("cp_len", 72))
    cp_len_first = int(acq.get("cp_len_first", 80))
    symbols_per_slot = int(acq.get("nr_symbols_per_slot", 14))

    symbols = []
    idx = 0
    sym_idx = 0

    # March through the array until we run out of samples
    while True:
        # The first symbol in a slot has a slightly longer CP
        current_cp = cp_len_first if (sym_idx % symbols_per_slot) == 0 else cp_len
        
        # If we don't have enough samples left for another full symbol, stop!
        if idx + current_cp + nfft > len(iq_complex):
            break
            
        # 1. Slice out the useful symbol time (skip the Cyclic Prefix)
        sym_time = iq_complex[idx + current_cp : idx + current_cp + nfft]
        
        # 2. Convert to frequency domain using FFT
        sym_freq = np.fft.fftshift(np.fft.fft(sym_time))
        
        # 3. Extract active subcarriers (grab the middle 60% to avoid empty guard bands)
        active_carriers = sym_freq[int(nfft * 0.2) : int(nfft * 0.8)]
        symbols.append(active_carriers)
        
        # Move our index forward to the start of the next symbol
        idx += (current_cp + nfft)
        sym_idx += 1

    if not symbols:
        print(f"Warning: Signal length ({len(iq_complex)}) is too short to extract even ONE symbol of length {nfft + cp_len_first}.")
        return np.array([]) 

    # Flatten into a single 1D array
    syms_out = np.concatenate(symbols)
    
    # Normalize the power to 1.0 so it scales perfectly into your [-1.6, 1.6] plot window
    rms = np.sqrt(np.mean(np.abs(syms_out)**2))
    if rms > 0:
        syms_out = syms_out / rms
        
    return syms_out

def _plot_constellation(syms_in, syms_out, output_path, label_prefix="Band 1"):
    """Equivalent to the MATLAB Constellation subplot using regular plt."""
    fig = plt.figure(figsize=(10, 5))
    
    # --- Input Constellation (Left) ---
    plt.subplot(1, 2, 1)
    plt.plot(syms_in.real, syms_in.imag, 'k.', markersize=2, alpha=0.5)
    plt.xlim([-1.6, 1.6])
    plt.ylim([-1.6, 1.6])
    plt.gca().set_aspect('equal', adjustable='box')  # Python's equivalent to 'axis square'
    plt.grid(True)
    plt.title(f'Input Constellation ({label_prefix})')
    plt.xlabel('I')
    plt.ylabel('Q')
    
    # --- PA Output Constellation (Right) ---
    plt.subplot(1, 2, 2)
    plt.plot(syms_out.real, syms_out.imag, 'b.', markersize=2, alpha=0.5)
    plt.xlim([-1.6, 1.6])
    plt.ylim([-1.6, 1.6])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.title(f'PA Output Constellation ({label_prefix})')
    plt.xlabel('I')
    plt.ylabel('Q')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_am_am_pm(x_complex, y_complex, output_path, label_prefix="Band 1"):
    """Equivalent to the MATLAB AM-AM and AM-PM subplot using regular plt."""
    abs_x = np.abs(x_complex)
    abs_y = np.abs(y_complex)
    
    # Calculate phase difference in degrees
    phase_diff = np.angle(y_complex * np.conj(x_complex), deg=True)
    mx = np.max(abs_x)
    
    fig = plt.figure(figsize=(12, 5))
    
    # --- AM-AM (Left) ---
    plt.subplot(1, 2, 1)
    plt.plot(abs_x, abs_y, '.', markersize=2, label='Measured', alpha=0.3)
    plt.plot([0, mx], [0, mx], 'k--', linewidth=1.2, label='Ideal Linear')
    plt.xlabel('|x|')
    plt.ylabel('|y_{eq}|')
    plt.title(f'AM-AM ({label_prefix})')
    plt.grid(True)
    plt.legend(loc='best')
    
    # --- AM-PM (Right) ---
    plt.subplot(1, 2, 2)
    plt.plot(abs_x, phase_diff, '.', markersize=2, label='Measured', alpha=0.3)
    plt.plot([0, mx], [0, 0], 'k--', linewidth=1.2, label='Ideal Linear')
    plt.xlabel('|x|')
    plt.ylabel(r'$\Delta\phi$ (deg)')
    plt.title(f'AM-PM ({label_prefix}, Corrected)')
    plt.grid(True)
    plt.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)

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


def _psd_for_plot(iq: np.ndarray, baseband_fs: float, target_fs: float, nperseg: int, smooth_window: int = 10):
    complex_signal = iq[:, 0] + 1j * iq[:, 1]
    
    # 1. Calculate exact integer up/down ratios for polyphase resampling
    up = int(round(target_fs / 1000))
    down = int(round(baseband_fs / 1000))
    g = math.gcd(up, down)
    up //= g
    down //= g
    
    # 2. Upsample the baseband signal so it physically spans the wider bandwidth
    complex_up = scipy.signal.resample_poly(complex_signal, up, down)
    
    # 3. Scale nperseg to maintain the exact same Hz/bin resolution
    nperseg_up = int(nperseg * (target_fs / baseband_fs))
    
    # 4. Calculate PSD using the target_fs (e.g., 200 MHz)
    freq, psd = metrics.power_spectrum(complex_up.reshape(1, -1), fs=target_fs, nperseg=nperseg_up, axis=-1)
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
    baseband_fs = spec["acquisition"]["fs_bb_hz"] 
    wideband_fs_spec = spec["input_signal_fs"] 
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
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(csv_path))[0]

    if psd_mode == "wideband" and is_triband:
        fc_list = _get_fc_list(spec)
        psd_fs = wideband_fs if wideband_fs is not None else wideband_fs_spec
        target_iq_full = original_iq_full.copy()
        
        for idx in range(3):
            band_gain = target_gains[idx]
            target_iq_full[:, idx * 2: idx * 2 + 2] *= band_gain
            
        # Catch the newly returned wb_target and wb_dpd arrays
        freq_ref, psd_ref, wb_target = _wideband_psd_for_plot(target_iq_full, baseband_fs=baseband_fs, wideband_fs=psd_fs, nperseg=nperseg, smooth_window=smooth_window, fc_list=fc_list)
        freq_dpd, psd_dpd, wb_dpd = _wideband_psd_for_plot(pa_after_dpd_full, baseband_fs=baseband_fs, wideband_fs=psd_fs, nperseg=nperseg, smooth_window=smooth_window, fc_list=fc_list)
        psd_label = "wideband"
        
        # --- WIDEBAND AM-AM / AM-PM ---
        am_pm_path = os.path.join(output_dir, f"{dataset_name}__{base_name}__ampm_wideband.png")
        _plot_am_am_pm(wb_target, wb_dpd, am_pm_path, label_prefix="Wideband")
        
        # --- WIDEBAND CONSTELLATION ---
        # Extract and combine symbols from ALL 3 bands into one master array
        all_syms_in, all_syms_out = [], []
        for i in range(3):
            b_target = target_iq_full[:, i*2] + 1j * target_iq_full[:, i*2+1]
            b_dpd = pa_after_dpd_full[:, i*2] + 1j * pa_after_dpd_full[:, i*2+1]
            all_syms_in.append(extract_constellation_symbols(b_target, spec))
            all_syms_out.append(extract_constellation_symbols(b_dpd, spec))
            
        syms_in = np.concatenate(all_syms_in)
        syms_out = np.concatenate(all_syms_out)
        if len(syms_in) > 0 and len(syms_out) > 0:
            const_path = os.path.join(output_dir, f"{dataset_name}__{base_name}__const_wideband.png")
            _plot_constellation(syms_in, syms_out, const_path, label_prefix="Wideband (All Bands)")
            
    else:
        freq_ref, psd_ref = _psd_for_plot(target_iq, baseband_fs=baseband_fs, target_fs=fs, nperseg=nperseg, smooth_window=smooth_window)
        freq_dpd, psd_dpd = _psd_for_plot(pa_after_dpd_iq, baseband_fs=baseband_fs, target_fs=fs, nperseg=nperseg, smooth_window=smooth_window)
        psd_label = f"band{band}"

        # --- SINGLE BAND AM-AM / AM-PM ---
        complex_target = target_iq[:, 0] + 1j * target_iq[:, 1]
        complex_dpd_out = pa_after_dpd_iq[:, 0] + 1j * pa_after_dpd_iq[:, 1]
        
        am_pm_path = os.path.join(output_dir, f"{dataset_name}__{base_name}__ampm_band{band}.png")
        _plot_am_am_pm(complex_target, complex_dpd_out, am_pm_path, label_prefix=f"Band {band}")

        # --- SINGLE BAND CONSTELLATION ---
        syms_in = extract_constellation_symbols(complex_target, spec)
        syms_out = extract_constellation_symbols(complex_dpd_out, spec)
        if len(syms_in) > 0 and len(syms_out) > 0:
            const_path = os.path.join(output_dir, f"{dataset_name}__{base_name}__const_band{band}.png")
            _plot_constellation(syms_in, syms_out, const_path, label_prefix=f"Band {band}")


    # --- PSD PLOTTING (Applies to both) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(freq_ref / 1e6, psd_ref, label="Target (gain * input)", linewidth=1.5)
    ax.plot(freq_dpd / 1e6, psd_dpd, label="PA output after DPD", linestyle="--", linewidth=1.5)

    if psd_mode == "wideband" and is_triband:
        ax.set_xlim(-100, 100)
    else:
        ax.set_xlim(-30, 30)
        ax.set_ylim(-60, 0) 

    ax.set_title(f"Normalized PSD ({psd_label}): {os.path.basename(csv_path)}")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Normalized PSD (dB)")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

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
