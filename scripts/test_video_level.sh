MODEL="LogisticModel"

YT8M_ROOT_DIR="/data01/lab_data/liyinghong/youtube-8m"

DATA_DIR="$YT8M_ROOT_DIR/feature/video_level"

TEST_DATA_PATTERN="$DATA_DIR/validate/validate*.tfrecord"

TRAIN_MODEL_DIR="$YT8M_ROOT_DIR/models/logistic"



FEATURE_NAMES="mean_rgb"

FRAME_FEATURES=False



#LABEL_LOSS="CrossEntropyLoss"



GPUS="1"

BATCH_SIZE=4096

FEATURE_SIZES=1024



SOURCE_DIR=$(dirname "$BASH_SOURCE")



cd $SOURCE_DIR/../youtube-8m/



# inference

python inference.py \

	—input_data_pattern=$TEST_DATA_PATTERN \

	--frame_features=$FRAME_FEATURES \

	--feature_names=$FEATURE_NAMES \

	--feature_sizes=$FEATURE_SIZES \

	--model=$MODEL \

	--train_dir=$TRAIN_MODEL_DIR \

	--gpus=$GPUS \

	--batch_size=$BATCH_SIZE \

	--output_file=$TRAIN_MODEL_DIR/predictions.csv

