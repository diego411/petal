from lightning.pytorch.cli import LightningArgumentParser, LightningCLI, SaveConfigCallback
from lightning.pytorch import LightningModule, Trainer
from ml.models.VisionCNN import VisionCNN 
from ml.data.PetalDataModule import PetalDataModule
import wandb
import yaml
from pathlib import Path
from ml.utils.generate_experiment_name import hash_yaml
import os

class CustomSaveConfigCallback(SaveConfigCallback):
    def save_config(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:    
        config = self.parser.dump(self.config, skip_none=False)  # Required for proper reproducibility
        yaml_data = yaml.safe_load(config)

        path = Path('ml/experiments') / hash_yaml('ml/scripts/vision_cnn/config.yaml')
        os.makedirs(path, mode=0o777, exist_ok=True)

        with open(path / 'config.yaml', "w") as file:
            yaml.dump(yaml_data, file, default_flow_style=False)


# TODO: maybe one cli for everything will work?
def cli_main() -> None:
    cli = LightningCLI(
        model_class=VisionCNN,
        datamodule_class=PetalDataModule,
        save_config_kwargs={"save_to_log_dir": False},
        save_config_callback=CustomSaveConfigCallback
    )

if __name__ == '__main__':
    wandb.login()
    cli_main()