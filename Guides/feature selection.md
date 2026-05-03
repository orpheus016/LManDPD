## Band Isolation
run the band isolation to acquire the .npz file containing the isolated bands. This will be used for feature selection in the next step.
```bash
python band_isolation.py --dataset /path/to/your/dataset.npz --output_dir /path/to/output
```
for now the dataset path is hardcoded will need refactor later

## Dictionary Matrix Generation
run the generate_dictionary_matrix_H to get the dictionary matrix H for the isolated bands. This will be used for feature selection in the next step.
```bash
python generate_dictionary_matrix_H.py --input_path /path/to/isolated_bands.npz --output_path /path/to/H_matrix.npz
```
for now the dataset path is hardcoded will need refactor later

## Feature Selection
Run the basis_selection script to perform feature selection using BOMP. This will output a blueprint of the selected features for each band. You can specify the stopbands according to your requirements. The example below assumes three bands centered at -60 MHz, 0 MHz, and 60 MHz with a bandwidth of 20 MHz each.
```bash
python scripts/basis_selection.py --dataset datasets/RFWebLab_PA_200MHz/H_matrix_and_Targets_M4.npz --fs 200e6 --stopbands="-70e6,-50e6;-10e6,10e6;50e6,70e6" --nmse_threshold -45.0
```
This will output a blueprint file and pareto graph for each band, which can be found in the `dpd_out/` directory. The blueprint files will be named like `hardware_blueprint.json`

### Note
Some notes on the feature selection step:
- The `--stopbands` argument should be set according to the specific bands you want to isolate. The example above assumes three bands centered at -60 MHz, 0 MHz, and 60 MHz with a bandwidth of 20 MHz each. Adjust the stopbands as needed for your specific dataset and requirements.
- you need to remove the hardware blueprint.json then generate new H dict if u decided to make a new feature to be analyzed