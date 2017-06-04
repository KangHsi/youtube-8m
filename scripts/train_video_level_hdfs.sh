#!/bin/bash

MODEL="Ensemble_lstm"
YT8M_ROOT_DIR="/data01/home/shikang/kaggle/tf-youtube-8m/youtube-8m/"
TRAIN_DATA_PATTERN="hdfs://haruna/user/lab/open/youtube8m/data/frame_level/train/train*.tfrecord"
TRAIN_MODEL_DIR="/data01/home/shikang/kaggle/models/lstm"

FEATURE_NAMES="rgb, audio"
FRAME_FEATURES=True
FEATURE_SIZES="1024, 128"

LABEL_LOSS="CrossEntropyLoss"
OPTIMIZER="AdamOptimizer"

DOWNLOAD_TMP_DIR="/data01/home/shikang/kaggle/models/lstm"
NUM_DOWNLOADERS=1
ENQUEUE_SIZE=4

GPUS="2"
NUM_EPOCHS=10
DISP_BATCHES=100
EXPORT_MODEL_STEPS=10000

SOURCE_DIR=$(dirname "$BASH_SOURCE")

cd $SOURCE_DIR/../youtube-8m/

python train_hdfs.py \
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
--export_model_steps=$EXPORT_MODEL_STEPS \
--data_tmp_dir=$DOWNLOAD_TMP_DIR \
--num_downloaders=$NUM_DOWNLOADERS \
--enqueue_size=$ENQUEUE_SIZE
