
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI, SaveConfigCallback
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.tuner.tuning import Tuner
from ml.models.VGGish import VGGish
from ml.data.VGGishDataModule import VGGishDataModule
import wandb
import yaml
from pathlib import Path
from ml.utils.generate_experiment_name import hash_yaml
from ml.utils.generate_experiment_version import get_experiment_version, get_latest_version
import os


class CustomSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:    
        config = self.parser.dump(self.config, skip_none=False)
        yaml_data = yaml.safe_load(config)

        config_path = 'ml/scripts/vision_cnn/config.yaml'
        experiment_path = Path('ml/experiments') / hash_yaml(config_path) 
        path = experiment_path / get_experiment_version(str(experiment_path))
        os.makedirs(path, mode=0o777, exist_ok=True)

        with open(path / 'config.yaml', "w") as file:
            yaml.dump(yaml_data, file, default_flow_style=False)


class CustomLightninCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument('--auto_lr_find', default=False) 

    def before_fit(self):
        if not self.config['fit']['auto_lr_find']:
            return
        
        tuner = Tuner(self.trainer)
        lr_finder = tuner.lr_find(self.model, datamodule=self.datamodule)

        assert lr_finder is not None, "LR finder is None!"
        
        new_lr = lr_finder.suggestion()
        self.model.learning_rate = new_lr 
        self.model.save_hyperparameters()

        config_path = 'ml/scripts/vision_cnn/config.yaml'
        experiment_path = Path('ml/experiments') / hash_yaml(config_path) 
        path = experiment_path / get_latest_version(str(experiment_path))

        with open(path / 'config.yaml', 'r') as file:
            yaml_data = yaml.safe_load(file)
        
        yaml_data['model']['lr'] = new_lr
        
        with open(path / 'config.yaml', "w") as file:
            yaml.dump(yaml_data, file, default_flow_style=False)

        print(f"Using suggested learning rate: {new_lr}")
            

# TODO: maybe one cli for everything will work?
def cli_main() -> None:
    cli = CustomLightninCLI(
        model_class=VGGish,
        datamodule_class=VGGishDataModule,
        save_config_kwargs={"save_to_log_dir": False},
        save_config_callback=CustomSaveConfigCallback
    )

if __name__ == '__main__':
    wandb.login()
    cli_main()