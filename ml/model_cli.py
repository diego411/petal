from lightning.pytorch.cli import LightningCLI
from ml.models.VisionCNNet import VisionCNN 
from ml.data.PlantAudioDataModule import PlantAudioDataModule
import wandb

def cli_main() -> None:
    wandb.login()
 
    cli = LightningCLI(
        model_class=VisionCNN,
        datamodule_class=PlantAudioDataModule,
        save_config_kwargs={"overwrite": True},
    )

if __name__ == '__main__':
    cli_main()