# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Binary for generating predictions over a set of videos."""

import os
import time

import numpy
import tensorflow as tf

from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
import embedding_models
import video_level_models
import eval_util
import losses
import readers
import utils

FLAGS = flags.FLAGS

if __name__ == '__main__':
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to load the model files from.")
  flags.DEFINE_string("model", "my_DbofModel,my_LstmModel",
                      "The directory to load the model files from.")
  flags.DEFINE_string("output_file", "",
                      "The file to save the predictions to.")
  flags.DEFINE_string(
      "input_data_pattern", "",
      "File glob defining the evaluation dataset in tensorflow.SequenceExample "
      "format. The SequenceExamples are expected to have an 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", False,
      "If set, then --eval_data_pattern must be frame-level features. "
      "Otherwise, --eval_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_integer(
      "batch_size", 8192,
      "How many examples to process per batch.")
  flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")


  # Other flags.
  flags.DEFINE_integer("num_readers", 1,
                       "How many threads to use for reading input files.")
  flags.DEFINE_integer("top_k", 20,
                       "How many predictions to output per video.")

def format_lines(video_ids, predictions, top_k):
  batch_size = len(video_ids)
  for video_index in range(batch_size):
    top_indices = numpy.argpartition(predictions[video_index], -top_k)[-top_k:]
    line = [(class_index, predictions[video_index][class_index])
            for class_index in top_indices]
    line = sorted(line, key=lambda p: -p[1])
    yield video_ids[video_index].decode('utf-8') + "," + " ".join("%i %f" % pair
                                                  for pair in line) + "\n"


def get_input_data_tensors(reader, data_pattern, batch_size, num_readers=1):
  """Creates the section of the graph which reads the input data.

  Args:
    reader: A class which parses the input data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  with tf.name_scope("input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find input files. data_pattern='" +
                    data_pattern + "'")
    logging.info("number of input files: " + str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=1, shuffle=False)
    examples_and_labels = [reader.prepare_reader(filename_queue)
                           for _ in range(num_readers)]

    video_id_batch, video_batch, unused_labels, num_frames_batch = (
        tf.train.batch_join(examples_and_labels,
                            batch_size=batch_size,
                            allow_smaller_final_batch=True,
                            enqueue_many=True))
    return video_id_batch, video_batch, num_frames_batch

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def inference(reader, train_dir, data_pattern, out_file_location, batch_size, top_k):
  model_names = FLAGS.model
  if len(model_names.split(',')) > 1:
    model = []
    for name in model_names.split(','):
      modules = find_class_by_name(name,
                                   [embedding_models, video_level_models])()
      model.append(modules)
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess, gfile.Open(out_file_location, "w+") as out_file:
    video_id_batch, model_input_raw, num_frames = get_input_data_tensors(reader, data_pattern, batch_size)

    feature_dim = len(model_input_raw.get_shape()) - 1

    # Normalize input features.
    model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
    # with tf.variable_scope("net1"):
    with tf.variable_scope("tower"):

      result1 = model[0].create_model(model_input,
                                           num_frames=num_frames,
                                           vocab_size=4716,
                                           is_training=False)
      #####

      result1 = tf.stop_gradient(result1)
      result2 = model[1].create_model(model_input,
                                           num_frames=num_frames,
                                           vocab_size=4716,
                                           is_training=False)
      result2 = tf.stop_gradient(result2)
      all_vars = tf.global_variables()
      # for v in all_vars:
      #   print v.name
      # for i in v_vars:
      #   logging.info(str(i))
      for i, v in enumerate(all_vars):
        logging.info(str(v.name))
        if 'rnn' in v.name:
          vars1 = all_vars[:i]
          vars2 = all_vars[i:]
          break

      result1 = tf.nn.l2_normalize(result1, dim=1)
      result2 = tf.nn.l2_normalize(result2, dim=1)
      embeddings = tf.concat([result1, result2], axis=1)
      model_concat = find_class_by_name('MoeModel',
                                        [video_level_models])()
      result = model_concat.create_model(embeddings, vocab_size=4716,
                                         num_mixtures=4)
      predictions = result["predictions"]
      # predictions=(result1["predictions"]+result2["predictions"])/2
      tf.summary.histogram("model_activations", predictions)
      # if "loss" in result.keys():
      #   label_loss = result["loss"]
      # else:
      # label_loss = losses.CrossEntropyLoss().calculate_loss(predictions, labels_batch)
      # tf.summary.scalar("label_loss", label_loss)
      # if "regularization_loss" in result.keys():
      #   reg_loss = result["regularization_loss"]
      # reg_losses = tf.losses.get_regularization_losses()
      # if "regularization_loss" in result.keys():
      #   reg_loss = result["regularization_loss"]
      # else:
      #   reg_loss = tf.constant(0.0)
      # final_loss = FLAGS.regularization_penalty * reg_loss + label_loss
      #
      # optimizer = optimizer_class(learning_rate)
      # gradients = optimizer.compute_gradients(final_loss,
      #                                         colocate_gradients_with_ops=False)
      #
      # with tf.name_scope('clip_grads'):
      #   merged_gradients = utils.clip_gradient_norms(gradients, 1.0)
      # train_op = optimizer.apply_gradients(merged_gradients, global_step=global_step)
      #
      # tf.add_to_collection("global_step", global_step)
      # tf.add_to_collection("loss", label_loss)
      tf.add_to_collection("input_batch_raw", model_input_raw)
      tf.add_to_collection("predictions", predictions)
      tf.add_to_collection("video_id_batch", video_id_batch)
      tf.add_to_collection("num_frames", num_frames)
      # tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
      tf.add_to_collection("summary_op", tf.summary.merge_all())
      # tf.add_to_collection("train_op", train_op)


      video_id_batch = tf.get_collection("video_id_batch")[0]
      # prediction_batch = tf.get_collection("predictions")[0]
      # label_batch = tf.get_collection("labels")[0]
      # loss = tf.get_collection("loss")[0]

    saver=tf.train.Saver()
    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if latest_checkpoint is None:
      raise Exception("unable to find a checkpoint at location: %s" % train_dir)
    else:
      meta_graph_location = latest_checkpoint + ".meta"
      logging.info("loading meta-graph: " + meta_graph_location)
    # saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
    logging.info("restoring variables from " + latest_checkpoint)
    saver.restore(sess, latest_checkpoint)
    input_tensor = tf.get_collection("input_batch_raw")[0]
    num_frames_tensor = tf.get_collection("num_frames")[0]
    predictions_tensor = tf.get_collection("predictions")[0]

    # Workaround for num_epochs issue.
    def set_up_init_ops(variables):
      init_op_list = []
      for variable in list(variables):
        logging.info("train_input:")
        logging.info(str(variable.name))
        if "train_input" in variable.name:

          init_op_list.append(tf.assign(variable, 1))
          variables.remove(variable)
      init_op_list.append(tf.variables_initializer(variables))
      return init_op_list

    sess.run(set_up_init_ops(tf.get_collection_ref(
        tf.GraphKeys.LOCAL_VARIABLES)))

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    num_examples_processed = 0
    start_time = time.time()
    out_file.write("VideoId,LabelConfidencePairs\n")

    try:
      while not coord.should_stop():
          video_id_batch_val, video_batch_val, num_frames_batch_val,predictions_val, = sess.run([video_id_batch, model_input_raw, num_frames,predictions_tensor])
          now = time.time()
          num_examples_processed += len(video_batch_val)
          num_classes = predictions_val.shape[1]
          logging.info("num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(now-start_time))
          for line in format_lines(video_id_batch_val, predictions_val, top_k):
            out_file.write(line)
          out_file.flush()


    except tf.errors.OutOfRangeError:
        logging.info('Done with inference. The output file was written to ' + out_file_location)
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()


def main(unused_argv):
  logging.set_verbosity(tf.logging.INFO)

  # convert feature_names and feature_sizes to lists of values
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  if FLAGS.frame_features:
    reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                            feature_sizes=feature_sizes)
  else:
    reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                 feature_sizes=feature_sizes)

  if FLAGS.output_file is "":
    raise ValueError("'output_file' was not specified. "
      "Unable to continue with inference.")

  if FLAGS.input_data_pattern is "":
    raise ValueError("'input_data_pattern' was not specified. "
      "Unable to continue with inference.")

  inference(reader, FLAGS.train_dir, FLAGS.input_data_pattern,
    FLAGS.output_file, FLAGS.batch_size, FLAGS.top_k)


if __name__ == "__main__":
  app.run()
