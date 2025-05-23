#!/bin/bash
BASE_CONFIG_FILE="ml/scripts/configs/base_config.yaml"
CONFIG_FILE="ml/scripts/configs/$1_config.yaml"
EXPERIMENT_HASH=$(python -m ml.utils.generate_experiment_name "$CONFIG_FILE")
EXPERIMENTS_PATH="ml/experiments"
EXPERIMENT_PATH="$EXPERIMENTS_PATH/$EXPERIMENT_HASH"
EXPERIMENT_VERSION=$(python -m ml.utils.generate_experiment_version $EXPERIMENT_PATH)

echo Fitting experiment $EXPERIMENT_HASH with version $EXPERIMENT_VERSION

python -m ml.clis.petal_cli fit \
    --config $CONFIG_FILE \
    --config $BASE_CONFIG_FILE \
    --trainer.logger.save_dir=$EXPERIMENTS_PATH \
    --trainer.logger.name=$EXPERIMENT_HASH \
    --trainer.logger.version=$EXPERIMENT_VERSION \
    --trainer.callbacks.dirpath=$EXPERIMENT_PATH/$EXPERIMENT_VERSION/checkpoints \
    --experiment_hash=$EXPERIMENT_HASH \
    --experiment_version=$EXPERIMENT_VERSION
