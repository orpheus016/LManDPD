import argparse
import json
import os
import shutil
import numpy as np
from scipy import signal

def load_wideband_iq(csv_path):
    # CSV must have columns I,Q
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    return data[:, 0] + 1j * data[:, 1]

def nco_mix_to_dc(x, fs, f_offset_hz):
    n = np.arange(x.size)
    t = n / fs
    return x * np.exp(-1j * 2.0 * np.pi * f_offset_hz * t)

def design_lowpass_fir(fs, bw_hz, numtaps=129, window="hamming"):
    # Passband: 0..bw/2; set cutoff slightly above bw/2 to keep band
    cutoff_hz = 0.55 * bw_hz
    nyq = 0.5 * fs
    return signal.firwin(numtaps, cutoff_hz / nyq, window=window)

def zero_phase_filter(x, b):
    return signal.filtfilt(b, [1.0], x)

def decimate_to_fs(x, fs_wide, fs_base):
    ratio = fs_wide / fs_base
    if abs(ratio - round(ratio)) > 1e-6:
        # Use resample_poly for non-integer ratios
        p = int(round(fs_base))
        q = int(round(fs_wide))
        y = signal.resample_poly(x, p, q)
        return y, fs_base
    ratio = int(round(ratio))
    y = signal.decimate(x, ratio, ftype="fir", zero_phase=True)
    return y, fs_base

def isolate_band(x_wide, fs_wide, f_offset, bw, fs_base, fir_taps=129):
    x_shift = nco_mix_to_dc(x_wide, fs_wide, f_offset)
    b = design_lowpass_fir(fs_wide, bw, numtaps=fir_taps)
    x_filt = zero_phase_filter(x_shift, b)
    x_dec, _ = decimate_to_fs(x_filt, fs_wide, fs_base)
    return x_dec.astype(np.complex128)

def align_lengths(*signals):
    min_len = min(sig.size for sig in signals)
    return [sig[:min_len] for sig in signals]

def isolate_bands(args):
    spec_path = os.path.join(args.input_dir, "spec.json")
    with open(spec_path, "r", encoding="utf-8") as f:
        spec = json.load(f)
    fs_wide = float(spec["input_signal_fs"])

    # Process Input Bands (Features)
    x_wide = load_wideband_iq(os.path.join(args.input_dir, "train_input.csv"))
    x1 = isolate_band(x_wide, fs_wide, args.f1, args.bw, args.fs_base, fir_taps=args.fir_taps)
    x2 = isolate_band(x_wide, fs_wide, args.f2, args.bw, args.fs_base, fir_taps=args.fir_taps)
    x3 = isolate_band(x_wide, fs_wide, args.f3, args.bw, args.fs_base, fir_taps=args.fir_taps)

    # Process Output Bands (Labels/Targets)
    y_wide = load_wideband_iq(os.path.join(args.input_dir, "train_output.csv"))
    y1 = isolate_band(y_wide, fs_wide, args.f1, args.bw, args.fs_base, fir_taps=args.fir_taps)
    y2 = isolate_band(y_wide, fs_wide, args.f2, args.bw, args.fs_base, fir_taps=args.fir_taps)
    y3 = isolate_band(y_wide, fs_wide, args.f3, args.bw, args.fs_base, fir_taps=args.fir_taps)

    # Sync all 6 streams to the same length
    x1, x2, x3, y1, y2, y3 = align_lengths(x1, x2, x3, y1, y2, y3)

    output_path = args.output_npz
    if output_path is None:
        output_path = os.path.join(args.output_dir, "isolated_bands.npz")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, x1=x1, x2=x2, x3=x3, y1=y1, y2=y2, y3=y3)
    print(f"Isolated bands saved to {output_path}")


def build_triband_dataset_from_csv(args):
    input_dir = args.input_dir
    output_dir = args.output_dir or input_dir
    os.makedirs(output_dir, exist_ok=True)

    mappings = {
        "train_input_triband.csv": "train_input.csv",
        "train_output_triband.csv": "train_output.csv",
        "val_input_triband.csv": "val_input.csv",
        "val_output_triband.csv": "val_output.csv",
        "test_input_triband.csv": "test_input.csv",
        "test_output_triband.csv": "test_output.csv",
    }

    for src_name, dst_name in mappings.items():
        src_path = os.path.join(input_dir, src_name)
        dst_path = os.path.join(output_dir, dst_name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Missing required file: {src_path}")
        shutil.copyfile(src_path, dst_path)

    for extra in ("spec.json", "acquisition_log.csv"):
        src_path = os.path.join(input_dir, extra)
        dst_path = os.path.join(output_dir, extra)
        if os.path.exists(src_path):
            shutil.copyfile(src_path, dst_path)

    print(f"Tri-band dataset created at {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Band isolation and tri-band dataset builder.")
    parser.add_argument("--mode", choices=["auto", "isolate", "triband_csv"], default="auto",
                        help="auto: detect triband CSVs; isolate: create isolated_bands.npz; triband_csv: copy *_triband.csv into split CSV layout")
    parser.add_argument("--input_dir", default="datasets/RFWebLab_PA_200MHz",
                        help="Input dataset directory")
    parser.add_argument("--output_dir", default=None,
                        help="Output dataset directory")

    # Isolation parameters
    parser.add_argument("--f1", type=float, default=-60e6, help="Band 1 center frequency (Hz)")
    parser.add_argument("--f2", type=float, default=0.0, help="Band 2 center frequency (Hz)")
    parser.add_argument("--f3", type=float, default=60e6, help="Band 3 center frequency (Hz)")
    parser.add_argument("--bw", type=float, default=20e6, help="Per-band bandwidth (Hz)")
    parser.add_argument("--fs_base", type=float, default=30.72e6, help="Baseband sample rate (Hz)")
    parser.add_argument("--fir_taps", type=int, default=129, help="FIR tap count")
    parser.add_argument("--output_npz", default=None, help="Path for isolated_bands.npz")

    args = parser.parse_args()

    if args.mode == "auto":
        candidate = os.path.join(args.input_dir, "train_input_triband.csv")
        args.mode = "triband_csv" if os.path.exists(candidate) else "isolate"

    if args.mode == "triband_csv":
        if args.output_dir is None:
            args.output_dir = args.input_dir
        build_triband_dataset_from_csv(args)
    else:
        if args.output_dir is None:
            args.output_dir = args.input_dir
        isolate_bands(args)

if __name__ == "__main__":
    main()