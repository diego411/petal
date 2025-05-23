from lightning.pytorch.cli import LightningArgumentParser, LightningCLI, SaveConfigCallback
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.tuner.tuning import Tuner
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


class CustomLightninCLI(LightningCLI):
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
        self.model.learning_rate = new_lr 
        self.model.save_hyperparameters()

        path = Path('ml/experiments') / self.config['fit']['experiment_hash'] / self.config['fit']['experiment_version']

        with open(path / 'config.yaml', 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        yaml_data['model']['init_args']['lr'] = new_lr
        
        with open(path / 'config.yaml', "w") as file:
            yaml.dump(yaml_data, file, default_flow_style=False)

        print(f"Using suggested learning rate: {new_lr}")
            
def cli_main() -> None:
    cli = CustomLightninCLI(
        save_config_kwargs={"save_to_log_dir": False},
        save_config_callback=CustomSaveConfigCallback
    )

if __name__ == '__main__':
    wandb.login()
    cli_main()