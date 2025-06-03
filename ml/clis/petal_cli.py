from lightning.pytorch.cli import LightningArgumentParser, LightningCLI, SaveConfigCallback
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.utilities import rank_zero_only
import wandb
import yaml
from pathlib import Path
import os


class CustomSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:    
        config = self.parser.dump(self.config, skip_none=False)
        yaml_data = yaml.safe_load(config)

        path = Path('ml/experiments') / yaml_data['experiment_hash'] / yaml_data['experiment_version'] 
        print(path)
        os.makedirs(path, mode=0o777, exist_ok=True)

        with open(path / 'config.yaml', "w") as file:
            yaml.dump(yaml_data, file, default_flow_style=False)


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument('--auto_lr_find', default=False)
        parser.add_argument('--experiment_hash')
        parser.add_argument('--experiment_version')
    
    def before_fit(self):
        if not self.config['fit']['auto_lr_find']:
            return
        
        tuner = Tuner(self.trainer)
        lr_finder = tuner.lr_find(self.model, datamodule=self.datamodule)
        assert lr_finder is not None, "LR finder is None!"
        new_lr = lr_finder.suggestion()
        
        # Update model learning rate on all processes
        self.model.learning_rate = new_lr
        self.model.save_hyperparameters()
        
        # Only perform file operations on rank 0 to avoid race conditions
        self._update_config_file_rank_zero_only(new_lr)
        
        print(f"Using suggested learning rate: {new_lr}")
    
    @rank_zero_only
    def _update_config_file_rank_zero_only(self, new_lr: float):
        """Update config file only on rank 0 to prevent race conditions."""
        path = Path('ml/experiments') / self.config['fit']['experiment_hash'] / self.config['fit']['experiment_version']
        config_path = path / 'config.yaml'
        
        try:
            with open(config_path, 'r') as file:
                yaml_data = yaml.safe_load(file)
            
            if yaml_data is not None and 'model' in yaml_data and 'init_args' in yaml_data['model']:
                yaml_data['model']['init_args']['lr'] = new_lr
                
                with open(config_path, "w") as file:
                    yaml.dump(yaml_data, file, default_flow_style=False)
                
                print(f"Updated config file with learning rate: {new_lr}")
            else:
                print("Warning: Invalid YAML structure, skipping config file update")
                
        except FileNotFoundError:
            print(f"Warning: Config file not found at {config_path}")
        except yaml.YAMLError as e:
            print(f"Warning: Error parsing YAML file: {e}")
        except Exception as e:
            print(f"Warning: Unexpected error updating config file: {e}")
            
def cli_main() -> None:
    cli = CustomLightningCLI(
        save_config_kwargs={"save_to_log_dir": False},
        save_config_callback=CustomSaveConfigCallback
    )

if __name__ == '__main__':
    wandb.login()
    cli_main()