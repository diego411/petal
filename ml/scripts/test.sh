# Configure this
EXPERIMENT_NAME=0507e4b4a2e5c20bb956a621abd1c801
EXPERIMENT_VERSION=version_0
CHECKPOINT_NAME="epoch=20-validation_macro_f1=0.38"

EXPERIMENTS_PATH=ml/experiments_to_test
EXPERIMENT_PATH=$EXPERIMENTS_PATH/$EXPERIMENT_NAME/$EXPERIMENT_VERSION
CHECKPOINT_PATH=$EXPERIMENT_PATH/checkpoints/$CHECKPOINT_NAME.ckpt

python -m ml.clis.petal_cli test \
    --config=$EXPERIMENT_PATH/config.yaml \
    --ckpt_path=$CHECKPOINT_PATH
