the current flow of our training per 10 may 2026
### Get RFWebLab PA data with Matlab

use matlab to get the PA data from RFWebLab
[acquire_pa_dataset_rfweblab.m](https://github.com/orpheus016/LManDPD/blob/main/Matlab/rfweblab/acquire_pa_dataset_rfweblab.m "acquire_pa_dataset_rfweblab.m")

### Extract Bands and Select Basis

run band separation like usual run python
```bash
python utils/band_separation.py --mode triband_csv --input_dir datasets/RFWebLab_PA_200MHz_Isolated
```

run this on the basis_selection.py
```bash
python scripts/basis_selection.py --dataset datasets/RFWebLab_PA_200MHz_Isolated/H_matrix_and_Targets_M4.npz --fs 200e6 --stopbands="-70e6,-50e6;-10e6,10e6;50e6,70e6" --nmse_threshold -45.0
```

### Train PA

```bash
python main.py
  --dataset_name RFWebLab_PA_200MHz_Isolated
  --step train_pa
  --accelerator cuda
  --devices 0
  --PA_backbone triband_qgru
  --PA_hidden_size 23
  --PA_num_layers 1
  --n_epochs 150
  --batch_size 64
  --batch_size_eval 256
  --lr 5e-4
  --opt_type adamw
```

### Train DPD

```bash
python main.py  --dataset_name RFWebLab_PA_200MHz_Isolated  --step train_dpd  --accelerator cuda  --devices 0  --PA_backbone triband_qgru  --PA_hidden_size 23  --PA_num_layers 1  --DPD_backbone triband_bdomp_tdnn  --DPD_hidden_size 59  --DPD_num_layers 1  --frame_length 200  --frame_stride 1  --loss_type l2  --opt_type adamw  --batch_size 64  --batch_size_eval 256  --n_epochs 240  --lr_schedule 1  --lr 5e-3  --lr_end 1e-4  --decay_factor 0.5  --patience 10  --quant  --n_bits_w 14  --n_bits_a 14  --quant_dir_label "q14"
```

### Run DPD

```bash
python main.py  --dataset_name RFWebLab_PA_200MHz_Isolated  --step run_dpd  --accelerator cuda  --devices 0  --PA_backbone triband_qgru  --PA_hidden_size 23  --PA_num_layers 1  --DPD_backbone triband_bdomp_tdnn  --DPD_hidden_size 59  --DPD_num_layers 1  --frame_length 200  --frame_stride 1  --quant  --n_bits_w 14  --n_bits_a 14
```

### Analysis

i copied the dpd_out folder q14 from q14 to [dataset name]/q14
**!NEED REVISION SO THAT THE QUANT TRAIN IMMEDIATELY GOES INSIDE THE DATASET**

```bash
python utils/compare_dpd_outputs.py --inputs "dpd_out/q14/DPD_*.csv" --output_dir "dpd_out/analysis/tdnn_after_dpd" --PA_backbone triband_qgru --PA_hidden_size 23 --PA_num_layers 1 --dataset_name RFWebLab_PA_200MHz_Isolated --band all
```

if only want one band (ex: band 3) then

```bash
python utils/compare_dpd_outputs.py --inputs "dpd_out/q14/DPD_*.csv" --output_dir "dpd_out/analysis/tdnn_after_dpd" --PA_backbone triband_qgru --PA_hidden_size 23 --PA_num_layers 1 --dataset_name RFWebLab_PA_200MHz_Isolated --band 3
```

check the analysis in dpd_out/analysis/tdnn_after_dpd