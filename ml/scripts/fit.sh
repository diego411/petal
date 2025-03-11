#!/bin/bash
CONFIG_FILE="ml/scripts/config.yaml"
EXPERIMENT_HASH=$(python -m ml.scripts.generate_experiment_name "$CONFIG_FILE")
EXPERIMENT_PATH="ml/experiments/$EXPERIMENT_HASH"
EXPERIMENT_VERSION=$(python -m ml.scripts.generate_experiment_version $EXPERIMENT_PATH)

python -m ml.model_cli fit --config $CONFIG_FILE --trainer.logger.save_dir=$EXPERIMENT_PATH --trainer.callbacks.dirpath=ml/experiments/$EXPERIMENT_HASH/$EXPERIMENT_VERSION/checkpoints
