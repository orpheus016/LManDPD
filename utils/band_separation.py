import json
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

def main():
    # Load fs_wide from spec.json
    with open("datasets/RFWebLab_PA_200MHz/spec.json", "r", encoding="utf-8") as f:
        spec = json.load(f)
    fs_wide = float(spec["input_signal_fs"])

    # TODO: set these
    f1 = -60e6      # Hz
    f2 = 0.0        # Hz
    f3 = 60e6       # Hz
    bw = 20e6      # Hz, per-band bandwidth
    fs_base = 30.72e6 # Hz

    # Process Input Bands (Features)
    x_wide = load_wideband_iq("datasets/RFWebLab_PA_200MHz/train_input.csv")
    x1 = isolate_band(x_wide, fs_wide, f1, bw, fs_base)
    x2 = isolate_band(x_wide, fs_wide, f2, bw, fs_base)
    x3 = isolate_band(x_wide, fs_wide, f3, bw, fs_base)

    # Process Output Bands (Labels/Targets)
    y_wide = load_wideband_iq("datasets/RFWebLab_PA_200MHz/train_output.csv")
    y1 = isolate_band(y_wide, fs_wide, f1, bw, fs_base)
    y2 = isolate_band(y_wide, fs_wide, f2, bw, fs_base)
    y3 = isolate_band(y_wide, fs_wide, f3, bw, fs_base)

    # Sync all 6 streams to the same length
    x1, x2, x3, y1, y2, y3 = align_lengths(x1, x2, x3, y1, y2, y3)

    # Now compatible with generate_dictionary_matrix_H(x1, x2, x3, M)
    print(x1.shape, x2.shape, x3.shape)
    print(y1.shape, y2.shape, y3.shape)
    output_path = "datasets/RFWebLab_PA_200MHz/isolated_bands.npz"
    np.savez(output_path, x1=x1, x2=x2, x3=x3, y1=y1, y2=y2, y3=y3)
    print(f"Isolated bands saved to {output_path}")

if __name__ == "__main__":
    main()