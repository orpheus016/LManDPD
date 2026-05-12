import json
import numpy as np
import pandas as pd
import metrics

def segment_iq(iq, nperseg):
    n_total = iq.shape[0]
    segments = []
    for start in range(0, n_total, nperseg):
        seg = iq[start:start + nperseg]
        if seg.shape[0] < nperseg:
            pad = np.zeros((nperseg - seg.shape[0], 2), dtype=seg.dtype)
            seg = np.vstack((seg, pad))
        segments.append(seg)
    return np.asarray(segments)

spec = json.load(open("datasets/RFWebLab_PA_200MHz/spec.json", "r"))

nperseg = spec["nperseg"]
fs = spec["input_signal_fs"]
bw_main_ch = spec["bw_main_ch"]
n_sub_ch = spec["n_sub_ch"]

x = pd.read_csv("datasets/RFWebLab_PA_200MHz/train_input.csv")[["I","Q"]].to_numpy()
y = pd.read_csv("datasets/RFWebLab_PA_200MHz/train_output.csv")[["I","Q"]].to_numpy()

x_seg = segment_iq(x, nperseg)
y_seg = segment_iq(y, nperseg)

nmse_out_vs_in_db = metrics.NMSE(y_seg, x_seg)
nmse_in_vs_out_db = metrics.NMSE(x_seg, y_seg)

evm_out_vs_in_db = metrics.EVM(
    y_seg,
    x_seg,
    sample_rate=fs,
    bw_main_ch=bw_main_ch,
    n_sub_ch=n_sub_ch,
    nperseg=nperseg,
)
evm_in_vs_out_db = metrics.EVM(
    x_seg,
    y_seg,
    sample_rate=fs,
    bw_main_ch=bw_main_ch,
    n_sub_ch=n_sub_ch,
    nperseg=nperseg,
)

aclr_in_l_db, aclr_in_r_db = metrics.ACLR(
    x_seg,
    fs=fs,
    nperseg=nperseg,
    bw_main_ch=bw_main_ch,
    n_sub_ch=n_sub_ch,
)
aclr_out_l_db, aclr_out_r_db = metrics.ACLR(
    y_seg,
    fs=fs,
    nperseg=nperseg,
    bw_main_ch=bw_main_ch,
    n_sub_ch=n_sub_ch,
)

print(f"NMSE (output vs input): {nmse_out_vs_in_db:.3f} dB")
print(f"NMSE (input vs output): {nmse_in_vs_out_db:.3f} dB")
print(f"EVM (output vs input): {evm_out_vs_in_db:.3f} dB")
print(f"EVM (input vs output): {evm_in_vs_out_db:.3f} dB")
print(f"ACLR (input): L={aclr_in_l_db:.3f} dB, R={aclr_in_r_db:.3f} dB")
print(f"ACLR (output): L={aclr_out_l_db:.3f} dB, R={aclr_out_r_db:.3f} dB")