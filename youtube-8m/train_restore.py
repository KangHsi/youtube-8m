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
"""Binary for training Tensorflow models on the YouTube-8M dataset."""

import json
import os
import time

import eval_util
import export_model
import losses

import embedding_models
import video_level_models
import readers
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow import app
from tensorflow import flags
from tensorflow import gfile
from tensorflow import logging
from tensorflow.python.client import device_lib
import utils

FLAGS = flags.FLAGS

if __name__ == "__main__":
  # Dataset flags.
  flags.DEFINE_string("train_dir", "/tmp/yt8m_model/",
                      "The directory to save the model files in.")
  flags.DEFINE_string("ensemble_dir", "/data01/home/shikang/kaggle/models/ensemble/",
                      "The directory to save the ensemble model.")
  flags.DEFINE_string(
      "train_data_pattern", "",
      "File glob for the training dataset. If the files refer to Frame Level "
      "features (i.e. tensorflow.SequenceExample), then set --reader_type "
      "format. The (Sequence)Examples are expected to have 'rgb' byte array "
      "sequence feature as well as a 'labels' int64 context feature.")
  flags.DEFINE_string("feature_names", "mean_rgb", "Name of the feature "
                      "to use for training.")
  flags.DEFINE_string("feature_sizes", "1024", "Length of the feature vectors.")

  # Model flags.
  flags.DEFINE_bool(
      "frame_features", False,
      "If set, then --train_data_pattern must be frame-level features. "
      "Otherwise, --train_data_pattern must be aggregated video-level "
      "features. The model must also be set appropriately (i.e. to read 3D "
      "batches VS 4D batches.")
  flags.DEFINE_string(
      "model", "DbofModel",
      "Which architecture to use for the model. Models are defined "
      "in models.py.")
  flags.DEFINE_bool(
      "start_new_model", False,
      "If set, this will not resume from a checkpoint and will instead create a"
      " new model instance.")

  # Training flags.
  flags.DEFINE_integer("batch_size", 1024,
                       "How many examples to process per batch for training.")
  flags.DEFINE_string("label_loss", "CrossEntropyLoss",
                      "Which loss function to use for training the model.")
  flags.DEFINE_float(
      "regularization_penalty", 1.0,
      "How much weight to give to the regularization loss (the label loss has "
      "a weight of 1).")
  flags.DEFINE_float("base_learning_rate", 0.01,
                     "Which learning rate to start with.")
  flags.DEFINE_float("learning_rate_decay", 0.95,
                     "Learning rate decay factor to be applied every "
                     "learning_rate_decay_examples.")
  flags.DEFINE_float("learning_rate_decay_examples", 4000000,
                     "Multiply current learning rate by learning_rate_decay "
                     "every learning_rate_decay_examples.")
  flags.DEFINE_integer("num_epochs", 5,
                       "How many passes to make over the dataset before "
                       "halting training.")
  flags.DEFINE_integer("max_steps", None,
                       "The maximum number of iterations of the training loop.")
  flags.DEFINE_integer("export_model_steps", 1000,
                       "The period, in number of steps, with which the model "
                       "is exported for batch prediction.")

  # Other flags.
  flags.DEFINE_string("gpus", "2",
                      "GPU ids to use.")
  flags.DEFINE_integer("num_readers", 8,
                       "How many threads to use for reading input files.")
  flags.DEFINE_integer("disp_batches", 100,
                       "Display losses and metrics each disp_batches step")
  flags.DEFINE_string("optimizer", "AdamOptimizer",
                      "What optimizer class to use.")
  flags.DEFINE_float("clip_gradient_norm", 1.0, "Norm to clip gradients to.")
  flags.DEFINE_bool(
      "log_device_placement", False,
      "Whether to write the device on which every op will run into the "
      "logs on startup.")

def validate_class_name(flag_value, category, modules, expected_superclass):
  """Checks that the given string matches a class of the expected type.

  Args:
    flag_value: A string naming the class to instantiate.
    category: A string used further describe the class in error messages
              (e.g. 'model', 'reader', 'loss').
    modules: A list of modules to search for the given class.
    expected_superclass: A class that the given class should inherit from.

  Raises:
    FlagsError: If the given class could not be found or if the first class
    found with that name doesn't inherit from the expected superclass.

  Returns:
    True if a class was found that matches the given constraints.
  """
  candidates = [getattr(module, flag_value, None) for module in modules]
  for candidate in candidates:
    if not candidate:
      continue
    if not issubclass(candidate, expected_superclass):
      raise flags.FlagsError("%s '%s' doesn't inherit from %s." %
                             (category, flag_value,
                              expected_superclass.__name__))
    return True
  raise flags.FlagsError("Unable to find %s '%s'." % (category, flag_value))

def get_input_data_tensors(reader,
                           data_pattern,
                           batch_size=1000,
                           num_epochs=None,
                           num_readers=1):
  """Creates the section of the graph which reads the training data.

  Args:
    reader: A class which parses the training data.
    data_pattern: A 'glob' style path to the data files.
    batch_size: How many examples to process at a time.
    num_epochs: How many passes to make over the training data. Set to 'None'
                to run indefinitely.
    num_readers: How many I/O threads to use.

  Returns:
    A tuple containing the features tensor, labels tensor, and optionally a
    tensor containing the number of frames per video. The exact dimensions
    depend on the reader being used.

  Raises:
    IOError: If no files matching the given pattern were found.
  """
  logging.info("Using batch size of " + str(batch_size) + " for training.")
  with tf.name_scope("train_input"):
    files = gfile.Glob(data_pattern)
    if not files:
      raise IOError("Unable to find training files. data_pattern='" +
                    data_pattern + "'.")
    logging.info("Number of training files: %s.", str(len(files)))
    filename_queue = tf.train.string_input_producer(
        files, num_epochs=num_epochs, shuffle=True)
    training_data = [
        reader.prepare_reader(filename_queue) for _ in range(num_readers)
    ]

    return tf.train.shuffle_batch_join(
        training_data,
        batch_size=batch_size,
        capacity=batch_size * 5,
        min_after_dequeue=batch_size,
        allow_smaller_final_batch=True,
        enqueue_many=True)


def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def build_graph_retrain(reader,
                model,
                train_data_pattern,
                label_loss_fn,
                batch_size=1024,
                num_readers=1):
  """Creates the Tensorflow graph for evaluation.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    eval_data_pattern: glob path to the evaluation data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    num_readers: How many threads to use for I/O operations.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")
  video_id_batch, model_input_raw, labels_batch, num_frames = get_input_data_tensors(  # pylint: disable=g-line-too-long
      reader,
      train_data_pattern,
      batch_size=batch_size,
      num_readers=num_readers)
  tf.summary.histogram("model_input_raw", model_input_raw)

  feature_dim = len(model_input_raw.get_shape()) - 1

  # Normalize input features.
  model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

  with tf.variable_scope("tower"):
    result = model.create_model(model_input,
                                num_frames=num_frames,
                                vocab_size=reader.num_classes,
                                labels=labels_batch,
                                is_training=False)

    predictions = result["predictions"]
    tf.summary.histogram("model_activations", predictions)
    if "loss" in result.keys():
      label_loss = result["loss"]
    else:
      label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("loss", label_loss)
  tf.add_to_collection("predictions", predictions)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("video_id_batch", video_id_batch)
  tf.add_to_collection("num_frames", num_frames)
  tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
  tf.add_to_collection("summary_op", tf.summary.merge_all())

def build_graph(reader,
                model,
                train_data_pattern,
                label_loss_fn=losses.CrossEntropyLoss(),
                batch_size=1000,
                base_learning_rate=0.01,
                learning_rate_decay_examples=1000000,
                learning_rate_decay=0.95,
                optimizer_class=tf.train.AdamOptimizer,
                clip_gradient_norm=1.0,
                regularization_penalty=1,
                num_readers=1,
                num_epochs=None):
  """Creates the Tensorflow graph.

  This will only be called once in the life of
  a training model, because after the graph is created the model will be
  restored from a meta graph file rather than being recreated.

  Args:
    reader: The data file reader. It should inherit from BaseReader.
    model: The core model (e.g. logistic or neural net). It should inherit
           from BaseModel.
    train_data_pattern: glob path to the training data files.
    label_loss_fn: What kind of loss to apply to the model. It should inherit
                from BaseLoss.
    batch_size: How many examples to process at a time.
    base_learning_rate: What learning rate to initialize the optimizer with.
    optimizer_class: Which optimization algorithm to use.
    clip_gradient_norm: Magnitude of the gradient to clip to.
    regularization_penalty: How much weight to give the regularization loss
                            compared to the label loss.
    num_readers: How many threads to use for I/O operations.
    num_epochs: How many passes to make over the data. 'None' means an
                unlimited number of passes.
  """

  global_step = tf.Variable(0, trainable=False, name="global_step")

  local_device_protos = device_lib.list_local_devices()
  gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
  num_gpus = len(gpus)

  if num_gpus > 0:
    logging.info("Using the following GPUs to train: " + str(gpus))
    num_towers = num_gpus
    device_string = '/gpu:%d'
  else:
    logging.info("No GPUs found. Training on CPU.")
    num_towers = 1
    device_string = '/cpu:%d'

  learning_rate = tf.train.exponential_decay(
      base_learning_rate,
      global_step * batch_size * num_towers,
      learning_rate_decay_examples,
      learning_rate_decay,
      staircase=True)
  tf.summary.scalar('learning_rate', learning_rate)

  optimizer = optimizer_class(learning_rate)
  unused_video_id, model_input_raw, labels_batch, num_frames = (
      get_input_data_tensors(
          reader,
          train_data_pattern,
          batch_size=batch_size * num_towers,
          num_readers=num_readers,
          num_epochs=num_epochs))
  tf.summary.histogram("model/input_raw", model_input_raw)

  feature_dim = len(model_input_raw.get_shape()) - 1

  model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)

  tower_inputs = tf.split(model_input, num_towers)
  tower_labels = tf.split(labels_batch, num_towers)
  tower_num_frames = tf.split(num_frames, num_towers)
  tower_gradients = []
  tower_predictions = []
  tower_label_losses = []
  tower_reg_losses = []
  for i in range(num_towers):
    # For some reason these 'with' statements can't be combined onto the same
    # line. They have to be nested.
    with tf.device(device_string % i):
      with (tf.variable_scope(("tower"), reuse=True if i > 0 else None)):
        with (slim.arg_scope([slim.model_variable, slim.variable], device="/cpu:0" if num_gpus!=1 else "/gpu:0")):
          result = model.create_model(
              tower_inputs[i],
              num_frames=tower_num_frames[i],
              vocab_size=reader.num_classes,
              labels=tower_labels[i])


          for variable in slim.get_model_variables():
            tf.summary.histogram(variable.op.name, variable)

          predictions = result["predictions"]
          logging.info("pred:%s",str(predictions))
          tower_predictions.append(predictions)

          if "loss" in result.keys():
            label_loss = result["loss"]
          else:
            label_loss = label_loss_fn.calculate_loss(predictions, tower_labels[i])

          if "regularization_loss" in result.keys():
            reg_loss = result["regularization_loss"]
          else:
            reg_loss = tf.constant(0.0)

          reg_losses = tf.losses.get_regularization_losses()
          if reg_losses:
            reg_loss += tf.add_n(reg_losses)

          tower_reg_losses.append(reg_loss)

          # Adds update_ops (e.g., moving average updates in batch normalization) as
          # a dependency to the train_op.
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
          if "update_ops" in result.keys():
            update_ops += result["update_ops"]
          if update_ops:
            with tf.control_dependencies(update_ops):
              barrier = tf.no_op(name="gradient_barrier")
              with tf.control_dependencies([barrier]):
                label_loss = tf.identity(label_loss)

          tower_label_losses.append(label_loss)

          # Incorporate the L2 weight penalties etc.
          final_loss = regularization_penalty * reg_loss + label_loss
          gradients = optimizer.compute_gradients(final_loss,
              colocate_gradients_with_ops=False)
          tower_gradients.append(gradients)
  label_loss = tf.reduce_mean(tf.stack(tower_label_losses))
  tf.summary.scalar("label_loss", label_loss)
  if regularization_penalty != 0:
    reg_loss = tf.reduce_mean(tf.stack(tower_reg_losses))
    tf.summary.scalar("reg_loss", reg_loss)
  merged_gradients = utils.combine_gradients(tower_gradients)

  if clip_gradient_norm > 0:
    with tf.name_scope('clip_grads'):
      merged_gradients = utils.clip_gradient_norms(merged_gradients, clip_gradient_norm)

  train_op = optimizer.apply_gradients(merged_gradients, global_step=global_step)

  tf.add_to_collection("global_step", global_step)
  tf.add_to_collection("loss", label_loss)
  logging.info("towerconcat:%s",str(tf.concat(tower_predictions, 0)))
  tf.add_to_collection("predictions", tf.concat(tower_predictions, 0))
  tf.add_to_collection("input_batch_raw", model_input_raw)
  tf.add_to_collection("input_batch", model_input)
  tf.add_to_collection("num_frames", num_frames)
  tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
  tf.add_to_collection("train_op", train_op)


class Trainer(object):
  """A Trainer to train a Tensorflow graph."""

  def __init__(self, cluster, task, train_dir, model, reader, model_exporter,
               log_device_placement=True, max_steps=None,
               export_model_steps=1000, disp_batches=100):
    """"Creates a Trainer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task
    self.is_master = (task.type == "master" and task.index == 0)
    self.train_dir = train_dir
    self.config = tf.ConfigProto(
        allow_soft_placement=True,log_device_placement=log_device_placement)
    self.config.gpu_options.allow_growth = True
    self.model = model
    self.reader = reader
    self.model_exporter = model_exporter
    self.max_steps = max_steps
    self.max_steps_reached = False
    self.export_model_steps = export_model_steps
    self.last_model_export_step = 0
    self.disp_batches = disp_batches

#     if self.is_master and self.task.index > 0:
#       raise StandardError("%s: Only one replica of master expected",
#                           task_as_string(self.task))

  def run(self, start_new_model=False):
    """Performs training on the currently defined Tensorflow graph.

    Returns:
      A tuple of the training Hit@1 and the training PERR.
    """
    if self.is_master and start_new_model:
        self.remove_training_directory(self.train_dir)

    target, device_fn = self.start_server_if_distributed()

    meta_filename = []
    for filename in self.train_dir.split(','):
      logging.info("filename:%s",str(filename))
      meta_filename.append(self.get_meta_filename(start_new_model, filename))

    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
    optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

    local_device_protos = device_lib.list_local_devices()
    gpus = [x.name for x in local_device_protos if x.device_type == 'GPU']
    num_gpus = len(gpus)

    if num_gpus > 0:
      logging.info("Using the following GPUs to train: " + str(gpus))
      num_towers = num_gpus
      device_string = '/gpu:%d'
    else:
      logging.info("No GPUs found. Training on CPU.")
      num_towers = 1
      device_string = '/cpu:%d'
    # build_graph_retrain(
    #     reader=self.reader,
    #     model=self.model,
    #     train_data_pattern=FLAGS.train_data_pattern,
    #     label_loss_fn=label_loss_fn,
    #     num_readers=FLAGS.num_readers,
    #     batch_size=FLAGS.batch_size)


    # with tf.variable_scope("net2"):

    ####

    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(
      FLAGS.base_learning_rate,
      global_step * FLAGS.batch_size * num_towers,
      FLAGS.learning_rate_decay_examples,
      FLAGS.learning_rate_decay,
      staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    video_id_batch, model_input_raw, labels_batch, num_frames = get_input_data_tensors(
      # pylint: disable=g-line-too-long
      self.reader,
      FLAGS.train_data_pattern,
      batch_size=FLAGS.batch_size,
      num_readers=FLAGS.num_readers)
    tf.summary.histogram("model_input_raw", model_input_raw)

    feature_dim = len(model_input_raw.get_shape()) - 1

    # Normalize input features.
    model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
    # with tf.variable_scope("net1"):
    with tf.variable_scope("tower"):

      result1 = self.model[0].create_model(model_input,
                                    num_frames=num_frames,
                                    vocab_size=self.reader.num_classes,
                                    is_training=False)
    #####

      result1=tf.stop_gradient(result1)
      result2= self.model[1].create_model(model_input,
                                    num_frames=num_frames,
                                    vocab_size=self.reader.num_classes,
                                    labels=labels_batch,
                                    is_training=False)
      result2=tf.stop_gradient(result2)
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
      # v_vars0 = [v for v in all_vars if v.name == 'tower/input_bn/beta:0'
      #           or v.name == 'tower/input_bn/gamma:0'
      #           or v.name == 'tower/input_bn/beta:0'
      #           or v.name == 'tower/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biases:0']
      # v_vars = [v for v in all_vars if v.name == 'tower/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/weights:0'
      #           or v.name == 'tower/rnn/multi_rnn_cell/cell_0/basic_lstm_cell/biases:0'
      #           or v.name == 'tower/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/weights:0'
      #           or v.name == 'tower/rnn/multi_rnn_cell/cell_1/basic_lstm_cell/biases:0']

      result1=tf.nn.l2_normalize(result1,dim=1)
      result2=tf.nn.l2_normalize(result2,dim=1)
      embeddings=tf.concat([result1,result2],axis=1)
      model_concat = find_class_by_name('MoeModel',
                                        [video_level_models])()
      result = model_concat.create_model(embeddings, vocab_size=self.reader.num_classes,
                                         num_mixtures=4)
      predictions = result["predictions"]
      # predictions=(result1["predictions"]+result2["predictions"])/2
      tf.summary.histogram("model_activations", predictions)
      # if "loss" in result.keys():
      #   label_loss = result["loss"]
      # else:
      label_loss = label_loss_fn.calculate_loss(predictions, labels_batch)
      tf.summary.scalar("label_loss", label_loss)
      if "regularization_loss" in result.keys():
        reg_loss = result["regularization_loss"]
      reg_losses = tf.losses.get_regularization_losses()
      if "regularization_loss" in result.keys():
        reg_loss = result["regularization_loss"]
      else:
        reg_loss = tf.constant(0.0)
      final_loss = FLAGS.regularization_penalty * reg_loss + label_loss

      optimizer = optimizer_class(learning_rate)
      gradients = optimizer.compute_gradients(final_loss,
                                              colocate_gradients_with_ops=False)

      with tf.name_scope('clip_grads'):
        merged_gradients = utils.clip_gradient_norms(gradients, 1.0)
      train_op = optimizer.apply_gradients(merged_gradients, global_step=global_step)


      tf.add_to_collection("global_step", global_step)
      tf.add_to_collection("loss", label_loss)
      tf.add_to_collection("predictions", predictions)
      tf.add_to_collection("input_batch", model_input)
      tf.add_to_collection("video_id_batch", video_id_batch)
      tf.add_to_collection("num_frames", num_frames)
      tf.add_to_collection("labels", tf.cast(labels_batch, tf.float32))
      tf.add_to_collection("summary_op", tf.summary.merge_all())
      tf.add_to_collection("train_op", train_op)


      video_id_batch = tf.get_collection("video_id_batch")[0]
      prediction_batch = tf.get_collection("predictions")[0]
      label_batch = tf.get_collection("labels")[0]
      loss = tf.get_collection("loss")[0]
      summary_op = tf.get_collection("summary_op")[0]
      # saver = tf.train.Saver(tf.global_variables())
      # saver=tf.train.Saver(result1)
      summary_writer = tf.summary.FileWriter(
        FLAGS.ensemble_dir, graph=tf.get_default_graph())

      config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False)
      config.gpu_options.allow_growth = True

      with tf.Session(config=config) as sess:
        train_dirs=FLAGS.train_dir.split(',')
        latest_checkpoint0 = tf.train.latest_checkpoint(train_dirs[0])
        latest_checkpoint1=tf.train.latest_checkpoint(train_dirs[1])
        sess.run(tf.global_variables_initializer())

        if latest_checkpoint0:
          logging.info("Loading checkpoint for eval: " + latest_checkpoint0)
          saver1=tf.train.Saver(vars1)

          saver1.restore(sess, latest_checkpoint0)


        if latest_checkpoint1:
          saver2=tf.train.Saver(vars2)
          logging.info("Loading checkpoint for eval: " + latest_checkpoint1)

          saver2.restore(sess, latest_checkpoint1)

        saver=tf.train.Saver()
        fetches = [learning_rate,global_step,train_op,video_id_batch, prediction_batch, label_batch, loss, summary_op]

        coord = tf.train.Coordinator()

        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(
            sess, coord=coord, daemon=True,
            start=True))



        while not coord.should_stop():
          # batch_start_time = time.time()
          learning_rate_val,global_step_val,_,vid_val, predictions_val, labels_val, loss_val, summary_val = sess.run(
              fetches)
          # hit_at_one = eval_util.calculate_hit_at_one(predictions_val, labels_val)
          # perr = eval_util.calculate_precision_at_equal_recall_rate(predictions_val,
          #                                                           labels_val)
          # gap = eval_util.calculate_gap(predictions_val, labels_val)
          # logging.info( "training step " + str(global_step_val)+" | Loss: " + ("%.2f" % loss_val) +" | Hit@1: " +
          #              ("%.4f" % hit_at_one) + " PERR: " + ("%.4f" % perr) +
          #              " GAP: " + ("%.4f" % gap))

          if self.is_master and global_step_val % self.disp_batches == 0 and self.train_dir:
            eval_start_time = time.time()
            hit_at_one = eval_util.calculate_hit_at_one(predictions_val, labels_val)
            perr = eval_util.calculate_precision_at_equal_recall_rate(predictions_val,
                                                                      labels_val)
            gap = eval_util.calculate_gap(predictions_val, labels_val)
            eval_end_time = time.time()
            eval_time = eval_end_time - eval_start_time
            logging.info("training step " + str(global_step_val)+"| learning rate: " + ("%.4f" % learning_rate_val)  + " | Loss: " + ("%.2f" % loss_val) + " | Hit@1: " +
                         ("%.4f" % hit_at_one) + " PERR: " + ("%.4f" % perr) +
                         " GAP: " + ("%.4f" % gap))
            summary_writer.add_summary(
              utils.MakeSummary("model/Training_Hit@1", hit_at_one),
              global_step_val)
            summary_writer.add_summary(
              utils.MakeSummary("model/Training_Perr", perr), global_step_val)
            summary_writer.add_summary(
              utils.MakeSummary("model/Training_GAP", gap), global_step_val)
            summary_writer.add_summary(
              utils.MakeSummary("model/loss", loss_val), global_step_val)
            summary_writer.add_summary(
              utils.MakeSummary("model/lr", learning_rate_val), global_step_val)
            summary_writer.flush()
            if global_step_val % FLAGS.export_model_steps==0:
              saver.save(sess, FLAGS.ensemble_dir, global_step=global_step_val)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


  def export_model(self, global_step_val, saver, save_path, session):

    # If the model has already been exported at this step, return.
    if global_step_val == self.last_model_export_step:
      return

    last_checkpoint = saver.save(session, save_path, global_step_val)

    model_dir = "{0}/export/step_{1}".format(self.train_dir, global_step_val)
    logging.info("%s: Exporting the model at step %s to %s.",
                 task_as_string(self.task), global_step_val, model_dir)

    self.model_exporter.export_model(
        model_dir=model_dir,
        global_step_val=global_step_val,
        last_checkpoint=last_checkpoint)

  def start_server_if_distributed(self):
    """Starts a server if the execution is distributed."""

    if self.cluster:
      logging.info("%s: Starting trainer within cluster %s.",
                   task_as_string(self.task), self.cluster.as_dict())
      server = start_server(self.cluster, self.task)
      target = server.target
      device_fn = tf.train.replica_device_setter(
          ps_device="/job:ps",
          worker_device="/job:%s/task:%d" % (self.task.type, self.task.index),
          cluster=self.cluster)
    else:
      target = ""
      device_fn = ""
    return (target, device_fn)

  def remove_training_directory(self, train_dir):
    """Removes the training directory."""
    try:
      logging.info(
          "%s: Removing existing train directory.",
          task_as_string(self.task))
      gfile.DeleteRecursively(train_dir)
    except:
      logging.error(
          "%s: Failed to delete directory " + train_dir +
          " when starting a new model. Please delete it manually and" +
          " try again.", task_as_string(self.task))

  def get_meta_filename(self, start_new_model, train_dir):
    if start_new_model:
      logging.info("%s: Flag 'start_new_model' is set. Building a new model.",
                   task_as_string(self.task))
      return None

    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if not latest_checkpoint:
      logging.info("%s: No checkpoint file found. Building a new model.",
                   task_as_string(self.task))
      return None

    meta_filename = latest_checkpoint + ".meta"
    if not gfile.Exists(meta_filename):
      logging.info("%s: No meta graph file found. Building a new model.",
                     task_as_string(self.task))
      return None
    else:
      return meta_filename

  def recover_model(self, meta_filename):
    logging.info("%s: Restoring from meta graph file %s",
                 task_as_string(self.task), meta_filename)
    return tf.train.import_meta_graph(meta_filename)

  def build_model(self, model, reader):
    """Find the model and build the graph."""

    label_loss_fn = find_class_by_name(FLAGS.label_loss, [losses])()
    optimizer_class = find_class_by_name(FLAGS.optimizer, [tf.train])

    build_graph(reader=reader,
                 model=model,
                 optimizer_class=optimizer_class,
                 clip_gradient_norm=FLAGS.clip_gradient_norm,
                 train_data_pattern=FLAGS.train_data_pattern,
                 label_loss_fn=label_loss_fn,
                 base_learning_rate=FLAGS.base_learning_rate,
                 learning_rate_decay=FLAGS.learning_rate_decay,
                 learning_rate_decay_examples=FLAGS.learning_rate_decay_examples,
                 regularization_penalty=FLAGS.regularization_penalty,
                 num_readers=FLAGS.num_readers,
                 batch_size=FLAGS.batch_size,
                 num_epochs=FLAGS.num_epochs)

    return tf.train.Saver(max_to_keep=0, keep_checkpoint_every_n_hours=0.25)


def get_reader():
  # Convert feature_names and feature_sizes to lists of values.
  feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
      FLAGS.feature_names, FLAGS.feature_sizes)

  if FLAGS.frame_features:
    reader = readers.YT8MFrameFeatureReader(
        feature_names=feature_names, feature_sizes=feature_sizes)
  else:
    reader = readers.YT8MAggregatedFeatureReader(
        feature_names=feature_names, feature_sizes=feature_sizes)

  return reader


class ParameterServer(object):
  """A parameter server to serve variables in a distributed execution."""

  def __init__(self, cluster, task):
    """Creates a ParameterServer.

    Args:
      cluster: A tf.train.ClusterSpec if the execution is distributed.
        None otherwise.
      task: A TaskSpec describing the job type and the task index.
    """

    self.cluster = cluster
    self.task = task

  def run(self):
    """Starts the parameter server."""

    logging.info("%s: Starting parameter server within cluster %s.",
                 task_as_string(self.task), self.cluster.as_dict())
    server = start_server(self.cluster, self.task)
    server.join()


def start_server(cluster, task):
  """Creates a Server.

  Args:
    cluster: A tf.train.ClusterSpec if the execution is distributed.
      None otherwise.
    task: A TaskSpec describing the job type and the task index.
  """

  if not task.type:
    raise ValueError("%s: The task type must be specified." %
                     task_as_string(task))
  if task.index is None:
    raise ValueError("%s: The task index must be specified." %
                     task_as_string(task))

  # Create and start a server.
  return tf.train.Server(
      tf.train.ClusterSpec(cluster),
      protocol="grpc",
      job_name=task.type,
      task_index=task.index)

def task_as_string(task):
  return "/job:%s/task:%s" % (task.type, task.index)

def main(unused_argv):
  # Load the environment.
  env = json.loads(os.environ.get("TF_CONFIG", "{}"))
  # os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpus
  # os.environ["CUDA_VISIBLE_DEVICES"] = -1
  # Load the cluster data from the environment.
  cluster_data = env.get("cluster", None)
  cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None

  # Load the task data from the environment.
  task_data = env.get("task", None) or {"type": "master", "index": 0}
  task = type("TaskSpec", (object,), task_data)

  # Logging the version.
  logging.set_verbosity(tf.logging.INFO)
  logging.info("%s: Tensorflow version: %s.",
               task_as_string(task), tf.__version__)

  # Dispatch to a master, a worker, or a parameter server.
  if not cluster or task.type == "master" or task.type == "worker":
    ####
    model_names=FLAGS.model
    if len(model_names.split(','))>1:
      model=[]
      for name in model_names.split(','):
        modules = find_class_by_name(name,
                                   [embedding_models, video_level_models])()
        model.append(modules)
    else:
    ####
      model = find_class_by_name(FLAGS.model,
          [embedding_models, video_level_models])()
    logging.info("models:%s",str(model))
    reader = get_reader()

    Trainer(cluster, task, FLAGS.train_dir, model, reader, None,
            FLAGS.log_device_placement, FLAGS.max_steps,
            FLAGS.export_model_steps, FLAGS.disp_batches).run(start_new_model=FLAGS.start_new_model)

  elif task.type == "ps":
    ParameterServer(cluster, task).run()
  else:
    raise ValueError("%s: Invalid task_type: %s." %
                     (task_as_string(task), task.type))

if __name__ == "__main__":
  app.run()
