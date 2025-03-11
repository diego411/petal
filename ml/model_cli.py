from lightning.pytorch.cli import LightningCLI
from ml.models.ResCNNetLightning import ResCNNetLightning
from ml.data.PlantAudioDataModule import PlantAudioDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
import os
import uuid

def cli_main() -> None:
    wandb.login()

    #exp_dir = f"./experiments/{uuid.uuid4()}"
    #os.makedirs(exp_dir, exist_ok=True)

    #wandb_logger = WandbLogger(
    #    project="protein-rapppid",
    #    offline=False,
    #    save_dir=exp_dir  # Store W&B logs in the same experiment folder
    #)

    ## Set up ModelCheckpoint to save in the correct directory
    #checkpoint_callback = ModelCheckpoint(
    #    dirpath=os.path.join(exp_dir, "checkpoints"),  # Save checkpoints here
    #    #filename="{epoch:02d}-{validation_f1:.2f}"
    #    save_top_k=3,
    #    monitor="validation_f1",
    #    mode="max"
    #)
    
    cli = LightningCLI(
        model_class=ResCNNetLightning,
        datamodule_class=PlantAudioDataModule,
        save_config_kwargs={"overwrite": True},
        #trainer_defaults={
        #    "default_root_dir": exp_dir,
        #    "logger": {
        #        "class_path": "lightning.pytorch.loggers.WandbLogger",
        #        "init_args": {
        #            "project": "petal",
        #            "offline": False,
        #            "save_dir": exp_dir  # Store W&B logs in the same experiment folder
        #        }
        #    },
        #    "callbacks": [
        #        {
        #            "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
        #            "init_args": {
        #                "dirpath": os.path.join(exp_dir, "checkpoints"),  # Save checkpoints here
        #                "save_top_k": 3,
        #                "monitor": "validation_f1",
        #                "mode": "max"
        #            }
        #        }
        #    ]
        #}
    )

if __name__ == '__main__':
    cli_main()