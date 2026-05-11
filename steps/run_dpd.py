__author__ = "Yizhuo Wu, Chang Gao"
__license__ = "Apache-2.0 License"
__email__ = "yizhuo.wu@tudelft.nl, chang.gao@tudelft.nl"

import os
import numpy as np
import pandas as pd
import torch
import models as model
from modules.paths import create_folder
from project import Project
from utils.util import count_net_params
from modules.data_collector import load_dataset

import sys
sys.path.append('../..')
from quant import get_quant_model
from quant.utlis import register_activation_hooks

def main(proj: Project):
    ###########################################################################################################
    # Initialization
    ###########################################################################################################
    # Set Accelerator Device
    proj.set_device()

    # Load Dataset
    _, _, _, _, X_test, _ = load_dataset(dataset_name=proj.dataset_name)
    input_size = X_test.shape[1]

    # Create DPD Output Folder
    create_folder(['dpd_out'])

    ###########################################################################################################
    # Network Settings
    ###########################################################################################################
    # Instantiate DPD Model
    net_pa = model.CoreModel(input_size=input_size,
                             hidden_size=proj.PA_hidden_size,
                             num_layers=proj.PA_num_layers,
                             backbone_type=proj.PA_backbone,
                             num_dvr_units=proj.num_dvr_units)
    n_net_pa_params = count_net_params(net_pa)
    print("::: Number of PA Model Parameters: ", n_net_pa_params)
    pa_model_id = proj.gen_pa_model_id(n_net_pa_params)
    net_dpd = model.CoreModel(input_size=input_size,
                              hidden_size=proj.DPD_hidden_size,
                              num_layers=proj.DPD_num_layers,
                              backbone_type=proj.DPD_backbone)

    net_dpd = get_quant_model(proj, net_dpd)
    
    n_net_dpd_params = count_net_params(net_dpd)
    print("::: Number of DPD Model Parameters: ", n_net_dpd_params)
    dpd_model_id = proj.gen_dpd_model_id(n_net_dpd_params)

    # Load Pretrained DPD Model
    pa_model_folder = pa_model_id.split('_P_')[0]
    base_path = os.path.join('save', proj.dataset_name, 'train_dpd', pa_model_folder)
    if not os.path.exists(base_path):
        save_root = os.path.join('save')
        candidates = []
        if os.path.exists(save_root):
            for dataset_dir in os.listdir(save_root):
                candidate = os.path.join(save_root, dataset_dir, 'train_dpd', pa_model_folder)
                if os.path.isdir(candidate):
                    candidates.append(candidate)
        if len(candidates) == 1:
            base_path = candidates[0]
            print(f"::: Using detected base path: {base_path}")
        elif len(candidates) > 1:
            raise FileNotFoundError(
                f"Base path not found: {base_path}. Multiple candidates detected: {candidates}"
            )
    
    quant_dir_label = None
    path_dpd_model = os.path.join(base_path, dpd_model_id + '.pt')

    if proj.args.quant:
        if proj.args.quant_dir_label:
            quant_dir_label = proj.args.quant_dir_label
            path_dpd_model = os.path.join(base_path, quant_dir_label, dpd_model_id + '.pt')
        else:
            # Auto-detect quant directory if quant_dir_label is not specified
            if os.path.exists(base_path):
                subdirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
                if subdirs:
                    quant_dir_label = subdirs[0]
                    path_dpd_model = os.path.join(base_path, quant_dir_label, dpd_model_id + '.pt')
                    print(f"::: Auto-detected quant directory: {quant_dir_label}")
                else:
                    raise FileNotFoundError(f"No quantization subdirectories found in {base_path}. Available contents: {os.listdir(base_path)}")
            else:
                raise FileNotFoundError(f"Base path not found: {base_path}")
        print("::: Loading Quantized DPD Model: ", path_dpd_model)
    
    if not os.path.exists(path_dpd_model):
        raise FileNotFoundError(f"DPD model not found at {path_dpd_model}")
    
    state_dict = torch.load(path_dpd_model)
    remap = {
        "backbone.add.": "backbone.feature_extractor.add.",
        "backbone.mul.": "backbone.feature_extractor.mul.",
    }
    # Remap quantized op keys after feature extractor refactor.
    remapped_state = {}
    for key, value in state_dict.items():
        new_key = key
        for old_prefix, new_prefix in remap.items():
            if key.startswith(old_prefix):
                new_key = new_prefix + key[len(old_prefix):]
                break
        remapped_state[new_key] = value
    net_dpd.load_state_dict(remapped_state)

    # Get parameter count
    n_net_params = count_net_params(net_dpd)
    print("::: Number of Network Parameters: ", n_net_params)

    # Move the network to the proper device
    net_dpd = net_dpd.to(proj.device)

    ###########################################################################################################
    # Run DPD
    ###########################################################################################################
    net_dpd = net_dpd.eval()
    with torch.no_grad():
        # Move test set data to the proper device
        dpd_in = torch.Tensor(X_test).unsqueeze(dim=0).to(proj.device)
        # DPD Model Forward Propagation
        dpd_out = net_dpd(dpd_in)
        # Remove the Batch Dimension
        dpd_out = torch.squeeze(dpd_out)
        # Move dpd_out to CPU
        dpd_out = dpd_out.cpu()

    ###########################################################################################################
    # Export Pre-distorted PA Inputs using the Test Set Data
    ###########################################################################################################
    if X_test.shape[1] == 6 and dpd_out.shape[1] == 6:
        in_cols = ["I1", "Q1", "I2", "Q2", "I3", "Q3"]
        out_cols = ["I1_dpd", "Q1_dpd", "I2_dpd", "Q2_dpd", "I3_dpd", "Q3_dpd"]
        pa_in = pd.DataFrame(
            np.hstack([X_test, dpd_out.numpy()]),
            columns=in_cols + out_cols,
        )
    else:
        pa_in = pd.DataFrame({'I': X_test[:, 0], 'Q': X_test[:, 1], 'I_dpd': dpd_out[:, 0], 'Q_dpd': dpd_out[:, 1]})
    path_file_pa_in = os.path.join('dpd_out', dpd_model_id + '.csv')
    if proj.args.quant and quant_dir_label:
        path_file_pa_in = os.path.join('dpd_out', quant_dir_label, dpd_model_id + '.csv')
        if not os.path.exists(os.path.join('dpd_out', quant_dir_label)):
            os.makedirs(os.path.join('dpd_out', quant_dir_label))
    pa_in.to_csv(path_file_pa_in, index=False)
    print("DPD outputs saved to the ./dpd_out folder.")
