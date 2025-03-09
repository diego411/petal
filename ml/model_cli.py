from lightning.pytorch.cli import LightningCLI
from ml.models.ResCNNetLightning import ResCNNetLightning
from ml.data.PlantAudioDataModule import PlantAudioDataModule
from lightning.pytorch.loggers import WandbLogger
import wandb

def cli_main() -> None:
    wandb.login()
    cli = LightningCLI(
        model_class=ResCNNetLightning,
        datamodule_class=PlantAudioDataModule,
        save_config_kwargs={"overwrite": True}
    )

if __name__ == '__main__':
    cli_main()