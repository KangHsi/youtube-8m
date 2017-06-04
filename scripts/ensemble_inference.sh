#!/bin/bash
#export PATH=/data00/home/liyinghong/miniconda2/bin:$PATH
MODEL="DbofModel"
YT8M_ROOT_DIR="/data01/home/shikang"
DATA_DIR="$YT8M_ROOT_DIR/kaggle/data/test"
TEST_DATA_PATTERN="$DATA_DIR/*"
TRAIN_DIR="$YT8M_ROOT_DIR/kaggle/tf-youtube-8m/ensemble"

FEATURE_NAMES="rgb,audio"
FRAME_FEATURES=True
FEATURE_SIZES="1024,128"

LABEL_LOSS="CrossEntropyLoss"

GPUS="0"
NUM_EPOCHS=10
DISP_BATCHES=10
EXPORT_MODEL_STEPS=1000

SOURCE_DIR=$(dirname "$BASH_SOURCE")

cd $SOURCE_DIR/../youtube-8m/

CUDA_VISIBLE_DIVICES=0 python ensemble_inference.py \
--input_data_pattern="$TEST_DATA_PATTERN" \
--feature_names="$FEATURE_NAMES" \
--feature_sizes="$FEATURE_SIZES" \
--train_dir="$TRAIN_DIR" \
--output_file="$TRAIN_DIR/predictions.csv" \
--frame_features=True \
--batch_size=512


