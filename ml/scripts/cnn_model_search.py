from itertools import product
import yaml
from ml.utils.generate_experiment_name import hash_yaml
from pathlib import Path
from ml.scripts.scripts_util import run_fit

CONFIG_PATH = 'ml/scripts/configs/cnn_config.yaml'

focused_grid = {
    'kernel_size': [3, 5],        # Most common effective sizes
    'pool_kernel_size': [2],      # Standard pooling
    'stride': [1, 2],             # No stride vs downsampling
    'pool_stride': [2],           # Standard pooling stride
    'padding': [1, 2],            # Valid for kernel sizes 3,5
    'dropout_rate': [0.1, 0.2, 0.3, 0.4]  # Common regularization range
}

def generate_combinations(grid):
    keys = grid.keys()
    values = grid.values()
    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))
    return combinations

all_combinations = generate_combinations(focused_grid)

for combination in all_combinations:
    with open(CONFIG_PATH, 'r') as file:
        yaml_data = yaml.safe_load(file)
    
    yaml_data['model']['init_args']['kernel_size'] = combination['kernel_size']
    yaml_data['model']['init_args']['pool_kernel_size'] = combination['pool_kernel_size']
    yaml_data['model']['init_args']['stride'] = combination['stride']
    yaml_data['model']['init_args']['pool_stride'] = combination['pool_stride']
    yaml_data['model']['init_args']['padding'] = combination['padding']
    yaml_data['model']['init_args']['dropout_rate'] = combination['dropout_rate']

    with open(CONFIG_PATH, "w") as file:
        yaml.dump(yaml_data, file, default_flow_style=False)
    
    hash = hash_yaml(CONFIG_PATH)
    experiment_path = Path('ml/experiments') / hash
    if experiment_path.exists():
        print(f"Skipping combination: {combination} since experiment with {hash} already exists")
        continue

    print("Starting run for following hyperparameter combination", combination)

    try:
        run_fit('fit_cnn')
    except Exception as e:
        print(e)
        success = False
