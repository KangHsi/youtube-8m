#!/bin/bash

MODEL="LogisticModel"
YT8M_ROOT_DIR="/data01/lab_data/liyinghong/youtube-8m"
DATA_DIR="$YT8M_ROOT_DIR/feature/video_level"
TRAIN_DATA_PATTERN="$DATA_DIR/train/train*.tfrecord"
VAL_DATA_PATTERN="$DATA_DIR/validate/validate*.tfrecord"
TRAIN_MODEL_DIR="$YT8M_ROOT_DIR/models/logistic"

FEATURE_NAMES="mean_rgb"
FRAME_FEATURES=False
FEATURE_SIZES=1024

LABEL_LOSS="CrossEntropyLoss"
OPTIMIZER="AdamOptimizer"

GPUS="1"
NUM_EPOCHS=10
DISP_BATCHES=100
EXPORT_MODEL_STEPS=10000

SOURCE_DIR=$(dirname "$BASH_SOURCE")

cd $SOURCE_DIR/../youtube-8m/

python train.py \
--train_data_pattern=$TRAIN_DATA_PATTERN \
--feature_sizes=$FEATURE_SIZES \
--feature_names=$FEATURE_NAMES \
--train_dir=$TRAIN_MODEL_DIR \
--frame_features=$FRAME_FEATURES \
--model=$MODEL \
--batch_size=128 \
--label_loss=$LABEL_LOSS \
--gpus=$GPUS \
--num_epochs=$NUM_EPOCHS \
--optimizer=$OPTIMIZER \
--disp_batches=$DISP_BATCHES \
--export_model_steps=$EXPORT_MODEL_STEPS

# validation
python eval.py \
--eval_data_pattern=$VAL_DATA_PATTERN \
--frame_features=$FRAME_FEATURES \
--feature_names=$FEATURE_NAMES \
--feature_sizes=$FEATURE_SIZES \
--model=$MODEL \
--label_loss=$LABEL_LOSS \
--train_dir=$TRAIN_MODEL_DIR \
--gpus=$GPUS \
--batch_size=1024 \
--run_once=True

