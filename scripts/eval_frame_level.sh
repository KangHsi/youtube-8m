#!/bin/bash
export PATH=/data00/home/liyinghong/miniconda2/bin:$PATH
MODEL="my_DbofModel"
YT8M_ROOT_DIR="/data01/home/shikang/"
DATA_DIR="$YT8M_ROOT_DIR/kaggle/data"
TRAIN_DATA_PATTERN="$DATA_DIR/train/train*.tfrecord"
VAL_DATA_PATTERN="$DATA_DIR/validate*.tfrecord"
TRAIN_MODEL_DIR="/data01/home/shikang/kaggle/tf-youtube-8m/dbof_hidden2"

FEATURE_NAMES="rgb,audio"
FRAME_FEATURES=True

LABEL_LOSS="CrossEntropyLoss"

GPUS="1"
BATCH_SIZE=1024


SOURCE_DIR=$(dirname "$BASH_SOURCE")

cd $SOURCE_DIR/../youtube-8m/

# validation
python eval.py \
--eval_data_pattern=$VAL_DATA_PATTERN \
--frame_features=$FRAME_FEATURES \
--feature_names=$FEATURE_NAMES \
--feature_sizes='1024, 128' \
--model=$MODEL \
--label_loss=$LABEL_LOSS \
--train_dir=$TRAIN_MODEL_DIR \
--gpus=$GPUS \
--batch_size=$BATCH_SIZE \
--iterations=60 \
--moe_num_mixtures=4 \
--run_once=True

