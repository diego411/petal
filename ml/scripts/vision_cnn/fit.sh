#!/bin/bash
CONFIG_FILE="ml/scripts/vision_cnn/config.yaml"
EXPERIMENT_HASH=$(python -m ml.utils.generate_experiment_name "$CONFIG_FILE")
EXPERIMENTS_PATH="ml/experiments"
EXPERIMENT_PATH="$EXPERIMENTS_PATH/$EXPERIMENT_HASH"
EXPERIMENT_VERSION=$(python -m ml.utils.generate_experiment_version $EXPERIMENT_PATH)

python -m ml.clis.vision_cnn_cli fit \
    --config $CONFIG_FILE \
    --trainer.logger.save_dir=$EXPERIMENTS_PATH \
    --trainer.logger.name=$EXPERIMENT_HASH \
    --trainer.logger.version=$EXPERIMENT_VERSION \
    --trainer.callbacks.dirpath=$EXPERIMENT_PATH/$EXPERIMENT_VERSION/checkpoints
