## TriBand_BDOMP_TDNN Training Pipeline

### Step 1: Train PA Model

Use GRU as the PA backbone (unchanged or you can change to dgru if u want to see how it performs with the TDNN DPD):

```bash
python main.py \
  --dataset_name RFWebLab_PA_200MHz \
  --step train_pa \
  --accelerator cuda \
  --devices 0 \
  --PA_backbone gru \
  --PA_hidden_size 23 \
  --PA_num_layers 1 \
  --n_epochs 150 \
  --batch_size 64 \
  --batch_size_eval 256 \
  --lr 5e-4 \
  --opt_type adamw
```

### Step 2: Train DPD Model with TriBand_BDOMP_TDNN

Use `--DPD_backbone triband_bdomp_tdnn` and set `--DPD_hidden_size ≤ 59, start with 14 first for a smaller model`:

```bash
python main.py \
  --dataset_name RFWebLab_PA_200MHz \
  --step train_dpd \
  --accelerator cuda \
  --devices 0 \
  --PA_backbone gru \
  --PA_hidden_size 23 \
  --PA_num_layers 1 \
  --DPD_backbone triband_bdomp_tdnn \
  --DPD_hidden_size 59 \
  --DPD_num_layers 1 \
  --frame_length 200 \
  --frame_stride 1 \
  --loss_type l2 \
  --opt_type adamw \
  --batch_size 64 \
  --batch_size_eval 256 \
  --n_epochs 240 \
  --lr_schedule 1 \
  --lr 5e-3 \
  --lr_end 1e-4 \
  --decay_factor 0.5 \
  --patience 10
```

**Note:** Input must be 6-channel (3 bands × I/Q) for tri-band training. If your dataset only has 2 channels, you will need a preprocessing step to replicate or generate the tri-band input.

### Step 2b: Train with QAT (Quantization-Aware Training)

Add these flags to enable 14-bit QAT on the TDNN (weights and activations):

```bash
python main.py \
  --dataset_name RFWebLab_PA_200MHz \
  --step train_dpd \
  --accelerator cuda \
  --devices 0 \
  --PA_backbone gru \
  --PA_hidden_size 23 \
  --PA_num_layers 1 \
  --DPD_backbone triband_bdomp_tdnn \
  --DPD_hidden_size 59 \
  --DPD_num_layers 1 \
  --frame_length 200 \
  --frame_stride 1 \
  --loss_type l2 \
  --opt_type adamw \
  --batch_size 64 \
  --batch_size_eval 256 \
  --n_epochs 240 \
  --lr_schedule 1 \
  --lr 5e-3 \
  --lr_end 1e-4 \
  --decay_factor 0.5 \
  --patience 10 \
  --quant \
  --n_bits_w 14 \
  --n_bits_a 14 \
  --quant_dir_label "q14"
```

**QAT Mechanism (Automatic):** When `--quant` is enabled, `get_quant_model()` wraps:
- `fc_hidden` and `fc_out` → `INT_Linear` with 14-bit fixed-point (Q1.13 via power-of-2 scale)
- `Add`, `Mul`, `Pow` in feature extraction → quantized ops
- `Tanh` activation → `Quant_tanh`

Reason: the TDNN uses only repo-known ops, so QAT replacement is automatic. Source: quant_envs.py.

### Step 2c: Train with Sparsity (Optional)

If you also want architectural sparsity, add this to your Python training script before the epoch loop:

```python
from backbones.triband_bdomp_tdnn import TriBand_BDOMP_TDNN

# After net_dpd is instantiated and wrapped by get_quant_model():
TriBand_BDOMP_TDNN.apply_unstructured_pruning(net_dpd.backbone, amount=0.75)
```

This prunes 75% of `fc_hidden` weights (L1 unstructured). Reason: architectural sparsity is applied post-creation; can combine with QAT for quantized+sparse training. Source: triband_bdomp_tdnn.py.

### Step 3: Run DPD Evaluation

Use the same backbone and hidden size from training:

```bash
python main.py \
  --dataset_name RFWebLab_PA_200MHz \
  --step run_dpd \
  --accelerator cuda \
  --devices 0 \
  --PA_backbone gru \
  --PA_hidden_size 23 \
  --PA_num_layers 1 \
  --DPD_backbone triband_bdomp_tdnn \
  --DPD_hidden_size 59 \
  --DPD_num_layers 1 \
  --frame_length 200 \
  --frame_stride 1 \
  --quant \
  --n_bits_w 14 \
  --n_bits_a 14
```

**Note:** If you trained with QAT, add `--quant` here too to match the model signature.

### Step 4: Analysis

```bash
python utils/compare_dpd_outputs.py \
  --inputs "dpd_out/RFWebLab_PA_200MHz/DPD_*.csv" \
  --output_dir "dpd_out/analysis/tdnn_after_dpd" \
  --PA_backbone gru \
  --PA_hidden_size 23 \
  --PA_num_layers 1
```

---

**QAT Clarification:** You do not need to manually activate the QAT class function. The repo's `get_quant_model()` does it automatically when `--quant` is passed. Your TriBand_BDOMP_TDNN is already compatible because it uses `Add`, `Mul`, `Pow` and `nn.Linear` layers—all of which QAT knows how to replace. The `apply_unstructured_pruning()` method is a separate, optional step for sparsity if you want it after training.