## Activation on vscode terminal

`.\\.venv\\Scripts\\Activate.ps1 `
## Step-by-step procedure

### Step 1: Train PA model on DPA_200MHz

````bash
python main.py --dataset_name RFWebLab_PA_200MHz --step train_pa --accelerator cuda --PA_backbone gru --PA_hidden_size 23 --n_epochs 150 --batch_size 64 --lr 5e-4
````

### Step 2: Train DPD model on DPA_200MHz

Use the same structure as OpenDPDv2.sh but pointed at `DPA_200MHz` with GPU:

````bash
python main.py
  --dataset_name RFWebLab_PA_200MHz
  --step train_dpd
  --accelerator cuda
  --devices 0
  --PA_backbone gru
  --PA_hidden_size 23
  --PA_num_layers 1
  --DPD_backbone deltagru_tcnskip
  --DPD_hidden_size 15
  --DPD_num_layers 1
  --frame_length 200
  --frame_stride 1
  --loss_type l2
  --opt_type adamw
  --batch_size 64
  --batch_size_eval 256
  --n_epochs 240
  --lr_schedule 1
  --lr 5e-3
  --lr_end 1e-4
  --decay_factor 0.5
  --patience 10
  --thx 0.01
  --thh 0.05
````

Note the trained DPD model checkpoint. It will be saved under:
```
save/DPA_200MHz/train_dpd/DPD_S_0_M_DELTAGRU_TCNSKIP_H_15_F_200*.pt
```

### Step 3: Run DPD evaluation on dataset

````bash
python main.py
  --dataset_name RFWebLab_PA_200MHz
  --step run_dpd
  --accelerator cuda
  --devices 0
  --PA_backbone gru
  --PA_hidden_size 23
  --PA_num_layers 1
  --DPD_backbone deltagru_tcnskip
  --DPD_hidden_size 15
  --DPD_num_layers 1
  --frame_length 200
  --frame_stride 1
  --loss_type l2
  --opt_type adamw
  --batch_size 64
  --batch_size_eval 256
  --n_epochs 240
  --lr_schedule 1
  --lr 5e-3
  --lr_end 1e-4
  --decay_factor 0.5
  --patience 10
  --thx 0.01
  --thh 0.05
````

**Test with one epoch first for sanity test**
### Check output
You can check the output of the validation in dpd_out folder
### Check model weights
You can check the model weights in save/[dataset name] folder
### Check output after run dpd
You can check the model weights in dpd_out folder then folder it into [dataset name] folder
### Run Analysis

````bash
python utils/compare_dpd_outputs.py --inputs "dpd_out/RFWebLab_PA_200MHz/DPD_S_0_M_DELTAGRU_TCNSKIP_H_15_F_200_P_999_THX_0.010_THH_0.050.csv" --output_dir "dpd_out/analysis/rfweblab_pa_after_dpd" --PA_backbone gru --PA_hidden_size 23 --PA_num_layers 1
````

check the output analysis on dpd_out/analysis/rfweblab_pa_after_dpd
### Dataset Signal Analysis

````bash
python utils/dataset_signal_analysis.py --dataset_name RFWebLab_PA_200MHz
````

check the output in datasets/[dataset_name]/signal_analysis to view the output