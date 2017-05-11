UnicodeDecodeError: 'utf8' codec can't decode byte 0xb4 in position 69: invalid start byte

BUCKET_NAME=gs://${USER}_yt8m_train_bucket

tensorboard --logdir=$BUCKET_NAME --port=8080


JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-4gpu.yaml \
-- --train_data_pattern='gs://youtube8m-ml-us-east1/1/frame_level/train/train*.tfrecord' \
--train_dir=$BUCKET_NAME/yt8m_train_frame_level_LSTM_model \
--frame_features=True --model=LstmModel --num_epochs=15 \
--base_learning_rate=0.0002 --learning_rate_decay=0.6 --learning_rate_decay_examples=11400000 \
--feature_names="rgb, audio" --feature_sizes="1024, 128" --batch_size=128


#the conv version, with one modal
JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --train_data_pattern='gs://youtube8m-ml-us-east1/1/frame_level/train/train*.tfrecord' \
--train_dir=$BUCKET_NAME/yt8m_train_frame_level_DbofModel_model_two_Modal \
--base_learning_rate=0.1 --regularization_penalty=50000.0 --learning_rate_decay=0.8 --learning_rate_decay_examples=2800000 --optimizer=AdagradOptimizer \
--frame_features=True --model=DbofModel --num_epochs=10 \
--feature_names="rgb, audio" --feature_sizes="1024, 128" --batch_size=128

(if you don't want to train from scratch, don't use start_new_model)



JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --train_data_pattern='gs://youtube8m-ml-us-east1/1/frame_level/train/train*.tfrecord' \
--train_dir=$BUCKET_NAME/yt8m_train_frame_level_CNN_model \
--frame_features=True --model=FrameLevelLogisticModel --num_epochs=10 --start_new_model=True \
--base_learning_rate=0.1 --learning_rate_decay=0.6 --regularization_penalty=10000.0 --learning_rate_decay_examples=2800000 --optimizer=AdagradOptimizer \
--feature_names="rgb, audio" --feature_sizes="1024, 128" --batch_size=128

#--optimizer=MomentumOptimizer --learning_rate_decay_examples=5786881
--------------------------eval--------------
JOB_TO_EVAL=yt8m_train_frame_level_DbofModel_model_two_Modal/
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://youtube8m-ml-us-east1/1/frame_level/validate/validate[0,1,2]*.tfrecord' \
--frame_features=True --model=DbofModel --moe_num_mixtures=4 --feature_names="rgb, audio" --feature_sizes="1024, 128" \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL}


JOB_TO_EVAL=yt8m_train_video_level_my_Moe_two_modal/
JOB_NAME=yt8m_eval_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.eval \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --eval_data_pattern='gs://youtube8m-ml-us-east1/1/video_level/validate/validate[0,1,2]*.tfrecord' \
--model=my_MoeModel --feature_names="mean_rgb, mean_audio" --feature_sizes="1024, 128" \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL}









-----------------------video level----------------------------

JOB_NAME=yt8m_train_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.train \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --train_data_pattern='gs://youtube8m-ml-us-east1/1/video_level/train/train*.tfrecord' \
--train_dir=$BUCKET_NAME/yt8m_train_video_level_my_Moe_two_modal --start_new_model=True \
--model=my_MoeModel \
--feature_names="mean_rgb, mean_audio" --feature_sizes="1024, 128" --batch_size=128







-------------inference-------
JOB_TO_EVAL=yt8m_train_frame_level_DbofModel_deq_model_two_Modal

JOB_TO_EVAL=yt8m_train_frame_level_CNNb_model

JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.inference \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --input_data_pattern='gs://youtube8m-ml/1/frame_level/test/test*.tfrecord' \
--frame_features=True --model=DbofModel --feature_names="rgb, audio" --feature_sizes="1024, 128" \
--batch_size=2048 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions.csv


---------------
JOB_TO_EVAL=yt8m_train_video_level_my_Moe_two_modal

JOB_NAME=yt8m_inference_$(date +%Y%m%d_%H%M%S); gcloud --verbosity=debug ml-engine jobs \
submit training $JOB_NAME \
--package-path=youtube-8m --module-name=youtube-8m.inference \
--staging-bucket=$BUCKET_NAME --region=us-east1 \
--config=youtube-8m/cloudml-gpu.yaml \
-- --input_data_pattern='gs://youtube8m-ml/1/video_level/test/test*.tfrecord' \
--model=my_MoeModel --feature_names="mean_rgb, mean_audio" --feature_sizes="1024, 128" \
--batch_size=4096 \
--train_dir=$BUCKET_NAME/${JOB_TO_EVAL} \
--output_file=$BUCKET_NAME/${JOB_TO_EVAL}/predictions.csv