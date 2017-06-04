#!/bin/bash
#export PATH=/data00/home/liyinghong/miniconda2/bin:$PATH
MODEL="DbofModel"
YT8M_ROOT_DIR="/data01/lab_data/liyinghong/youtube-8m"
DATA_DIR="$YT8M_ROOT_DIR/feature/frame_level"
TRAIN_DATA_PATTERN="$DATA_DIR/train/train*.tfrecord"
VAL_DATA_PATTERN="$DATA_DIR/validate/validate*.tfrecord"
TRAIN_MODEL_DIR="/data01/home/shikang/kaggle/models/dbof"

FEATURE_NAMES="rgb"
FRAME_FEATURES=True
FEATURE_SIZES=1024

LABEL_LOSS="CrossEntropyLoss"
OPTIMIZER="AdamOptimizer"

GPUS="0"
NUM_EPOCHS=10
DISP_BATCHES=10
EXPORT_MODEL_STEPS=100

SOURCE_DIR=$(dirname "$BASH_SOURCE")

cd $SOURCE_DIR/../youtube-8m/

python eval.py \
--eval_data_pattern=$TRAIN_DATA_PATTERN \
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
--start_new_model=False \
--export_model_steps=$EXPORT_MODEL_STEPS

