#!/bin/bash
#export PATH=/data00/home/liyinghong/miniconda2/bin:$PATH
MODEL="LstmModel"
YT8M_ROOT_DIR="/data01/lab_data/liyinghong/youtube-8m"
DATA_DIR="$YT8M_ROOT_DIR/feature/frame_level"
TRAIN_DATA_PATTERN="$DATA_DIR/train/train*.tfrecord"
VAL_DATA_PATTERN="$DATA_DIR/validate/validate*.tfrecord"
TRAIN_MODEL_DIR="/data01/home/shikang/kaggle/models/lstm"

FEATURE_NAMES="rgb"
FRAME_FEATURES=True
FEATURE_SIZES=1024

LABEL_LOSS="CrossEntropyLoss"
OPTIMIZER="AdamOptimizer"

GPUS="0"
NUM_EPOCHS=10
DISP_BATCHES=10
EXPORT_MODEL_STEPS=1000

SOURCE_DIR=$(dirname "$BASH_SOURCE")

cd $SOURCE_DIR/../youtube-8m/

CUDA_VISIBLE_DEVICES=0 python inference.py \
--input_data_pattern="$VAL_DATA_PATTERN" \
--feature_names="$FEATURE_NAMES" \
--feature_sizes="$FEATURE_SIZES" \
--train_dir="$TRAIN_MODEL_DIR" \
--output_file="$TRAIN_MODEL_DIR/predictions.csv" \
--frame_features=True \
--batch_size=256


