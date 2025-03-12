from lightning.pytorch.cli import LightningCLI
from ml.models.VisionCNNet import VisionCNN 
from ml.data.PetalDataModule import PetalDataModule
import wandb

# TODO: maybe one cli for everything will work?
def cli_main() -> None:
    wandb.login()
 
    cli = LightningCLI(
        model_class=VisionCNN,
        datamodule_class=PetalDataModule,
        save_config_kwargs={"overwrite": True},
    )

if __name__ == '__main__':
    cli_main()