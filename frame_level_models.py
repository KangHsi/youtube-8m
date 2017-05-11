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

"""Contains a collection of models which operate on variable-length sequences.
"""
import math

import models
import video_level_models
import tensorflow as tf
import model_utils as utils
import utils as tools
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow import flags
from tensorflow import logging
FLAGS = flags.FLAGS
flags.DEFINE_integer("iterations", 30,
                     "Number of frames per batch for DBoF.")
flags.DEFINE_bool("dbof_add_batch_norm", True,
                  "Adds batch normalization to the DBoF model.")
flags.DEFINE_bool(
    "sample_random_frames", True,
    "If true samples random frames (for frame level models). If false, a random"
    "sequence of frames is sampled instead.")
flags.DEFINE_integer("dbof_cluster_size", 8192,
                     "Number of units in the DBoF cluster layer.")
flags.DEFINE_integer("dbof_hidden_size", 1024,
                     "Number of units in the DBoF hidden layer.")
flags.DEFINE_string("dbof_pooling_method", "max",
                    "The pooling method used in the DBoF cluster layer. "
                    "Choices are 'average' and 'max'.")
flags.DEFINE_string("video_level_classifier_model", "MoeModel",
                    "Some Frame-Level models can be decomposed into a "
                    "generalized pooling operation followed by a "
                    "classifier layer")
flags.DEFINE_integer("lstm_cells", 1024, "Number of LSTM cells.")
flags.DEFINE_integer("lstm_layers", 2, "Number of LSTM layers.")






class FrameLevelLogisticModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a logistic classifier over the average of the
    frame-level features.

    This class is intended to be an example for implementors of frame level
    models. If you want to train a model over averaged features it is more
    efficient to average them beforehand rather than on the fly.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
  ##########################################################original logisticModel

    # logging.info("model_input_shape: %s." ,str(model_input))
    #
    # ###(1,300,1024),padding to 300 frames even if the true num_frames not 300.
    # ##if use audio_information, the vector becomes(?,300,1152),since 1152=1024+128
    # num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    # feature_size = model_input.get_shape().as_list()[2]
    #
    # denominators = tf.reshape(
    #     tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    # ##
    # logging.info("denominators: %s.", str(denominators))
    #
    # ##(1,1024)
    # avg_pooled = tf.reduce_sum(model_input,
    #                            axis=[1]) / denominators
    # ##an average 1024 feature
    # output = slim.fully_connected(
    #     avg_pooled, vocab_size, activation_fn=tf.nn.sigmoid,
    #     weights_regularizer=slim.l2_regularizer(1e-8))
    # return {"predictions": output}

  #############################################################



    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    feature_size = model_input.get_shape().as_list()[2]
    extrac_frames=100
    model_input = utils.SampleFramesOrdered(model_input, num_frames,extrac_frames)#
    model_input=tf.expand_dims(model_input,-1)
    logging.info("model_input_after_shape: %s.", str(model_input))
    #batchsize*extrac_frames*feature_size

    ########


    filters = [16, 64, 256, 1024, 4096]
    #dequantize
    model_input=tools.Dequantize(model_input)

    x = self._conv('conv1', model_input, time_stride=30,in_filters=1,out_filters= 100,feature_size=feature_size,
                   strides=[1, 10, 1, 1],padding='VALID')
    logging.info("after_conv1: %s.", str(x))
    #8
    bias = tf.get_variable('bias1', [100], tf.float32, initializer=tf.zeros_initializer())

    x=self._relu(x+bias,0.0)

    x=tf.nn.max_pool(x,ksize=[1,8,1,1],strides=[1,8,1,1],padding='VALID',name="max1")
    #42


    # logging.info("x_after_maxpool1: %s.", str(x))
    # x=self._conv('conv2',x,time_stride=3,in_filters=filters[0],out_filters=filters[2],feature_size=1,
    #              strides=[1,1,1,1],padding='SAME')
    # bias = tf.get_variable('bias2', [filters[2]], tf.float32, initializer=tf.zeros_initializer())
    #
    # x = self._relu(x + bias,0.0)
    #
    # x=tf.nn.max_pool(x,ksize=[1,41,1,1],strides=[1,41,1,1],padding='VALID',name="max2")
    # #21
    #
    # # x=self.group_conv(name='group',x=x,time_stride=21,in_filters=filters[1],out_filters=filters[2],strides=[1,1,1,1])
    #
    # x=tf.nn.relu6(x,name='relu6')



    x=tf.contrib.layers.flatten(x)
    # x=tf.nn.dropout(x,keep_prob=0.5)

    logging.info("output: %s.", x)

    # hidden = slim.fully_connected(
    #     x, 8196, activation_fn=None,
    #     weights_regularizer=slim.l2_regularizer(1e-8))
    # # drop=tf.nn.dropout(hidden,keep_prob=0.5)
    # hidden=tf.nn.relu(hidden,'relu6')


    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    # logging.info("DBoF_activitions:%s", str(activation))
    return aggregated_model().create_model(
        model_input=x,
        vocab_size=vocab_size,
        **unused_params)

    #bs*126*1



    # denominators = tf.reshape(
    #     tf.tile(num_frames, [1, feature_size]), [-1, feature_size])
    # ##
    # logging.info("denominators: %s.", str(denominators))
    #
    # ##(1,1024)
    # avg_pooled = tf.reduce_sum(model_input,
    #                            axis=[1]) / denominators
    ##an average 1024 feature
    # logging.info("vocab_size:%s",str(vocab_size))

    # output = slim.fully_connected(
    #     x, vocab_size, activation_fn=tf.nn.sigmoid,
    #     weights_regularizer=slim.l2_regularizer(1e-8))
    # return {"predictions": output}

  def _conv(self, name, x, time_stride, in_filters,out_filters,feature_size, strides,padding='SAME'):
      """Convolution."""
      with tf.variable_scope(name):
          n = time_stride * feature_size* out_filters
          kernel = tf.get_variable(
              'DW', [time_stride, feature_size, in_filters, out_filters],
              tf.float32, initializer=tf.contrib.layers.xavier_initializer())
          # tf.random_normal_initializer(
          # stddev = np.sqrt(2.0 / n)
          return tf.nn.conv2d(x, kernel, strides, padding=padding)

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""

    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def group_conv(self, name, x, time_stride, in_filters, out_filters, strides):
    """group Convolution."""
    with tf.variable_scope(name):
      channel_multiplier = 4.0
      n = 1 * time_stride * out_filters
      dep_kernel = tf.get_variable(
        'DW'  , [time_stride, 1, in_filters, channel_multiplier],
        tf.float32, initializer=tf.random_normal_initializer(
          stddev=np.sqrt(2.0*out_filters/n)))
      pnt_kernel = tf.get_variable(
        'DW'+'pntwise' , [1, 1, channel_multiplier*in_filters , out_filters],
        tf.float32, initializer=tf.random_normal_initializer(
          stddev=np.sqrt(2.0 / out_filters)))

      return tf.nn.separable_conv2d(x, depthwise_filter=dep_kernel, pointwise_filter=pnt_kernel, strides=strides,
                                   padding='SAME')



class DbofModel(models.BaseModel):
  """Creates a Deep Bag of Frames model.

  The model projects the features for each frame into a higher dimensional
  'clustering' space, pools across frames in that space, and then
  uses a configurable video-level model to classify the now aggregated features.

  The model will randomly sample either frames or sequences of frames during
  training to speed up convergence.

  Args:
    model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                 input features.
    vocab_size: The number of classes in the dataset.
    num_frames: A vector of length 'batch' which indicates the number of
         frames for each video (before padding).

  Returns:
    A dictionary with a tensor containing the probability predictions of the
    model in the 'predictions' key. The dimensions of the tensor are
    'batch_size' x 'num_classes'.
  """

  def create_model(self,
                   model_input,
                   vocab_size,
                   num_frames,
                   iterations=None,
                   add_batch_norm=None,
                   sample_random_frames=None,
                   cluster_size=None,
                   hidden_size=None,
                   is_training=True,
                   **unused_params):
    iterations = iterations or FLAGS.iterations
    add_batch_norm = add_batch_norm or FLAGS.dbof_add_batch_norm
    random_frames = sample_random_frames or FLAGS.sample_random_frames
    cluster_size = cluster_size or FLAGS.dbof_cluster_size
    hidden1_size = hidden_size or FLAGS.dbof_hidden_size

    num_frames = tf.cast(tf.expand_dims(num_frames, 1), tf.float32)
    if random_frames:
      model_input = utils.SampleRandomFrames(model_input, num_frames,
                                             iterations)
    else:
      model_input = utils.SampleRandomSequence(model_input, num_frames,
                                               iterations)
    max_frames = model_input.get_shape().as_list()[1]
    feature_size = model_input.get_shape().as_list()[2]
    reshaped_input = tf.reshape(model_input, [-1, feature_size])
    ##dequantize
    logging.info("max:%s",str(tf.arg_max(reshaped_input,1)))
    logging.info("min:%s", str(tf.arg_min(reshaped_input,1)))
    reshaped_input=tools.Dequantize(reshaped_input,255,0)

    tf.summary.histogram("input_hist", reshaped_input)

    if add_batch_norm:
      reshaped_input = slim.batch_norm(
          reshaped_input,
          center=True,
          scale=True,
          is_training=is_training,
          scope="input_bn")

    cluster_weights = tf.get_variable("cluster_weights",
      [feature_size, cluster_size],
      initializer = tf.random_normal_initializer(stddev=1 / math.sqrt(feature_size)))
    tf.summary.histogram("cluster_weights", cluster_weights)
    activation = tf.matmul(reshaped_input, cluster_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="cluster_bn")
    else:
      cluster_biases = tf.get_variable("cluster_biases",
        [cluster_size],
        initializer = tf.random_normal(stddev=1 / math.sqrt(feature_size)))
      tf.summary.histogram("cluster_biases", cluster_biases)
      activation += cluster_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("cluster_output", activation)

    activation = tf.reshape(activation, [-1, max_frames, cluster_size])
    activation = utils.FramePooling(activation, FLAGS.dbof_pooling_method)

    hidden1_weights = tf.get_variable("hidden1_weights",
      [cluster_size, hidden1_size],
      initializer=tf.random_normal_initializer(stddev=1 / math.sqrt(cluster_size)))
    tf.summary.histogram("hidden1_weights", hidden1_weights)
    activation = tf.matmul(activation, hidden1_weights)
    if add_batch_norm:
      activation = slim.batch_norm(
          activation,
          center=True,
          scale=True,
          is_training=is_training,
          scope="hidden1_bn")
    else:
      hidden1_biases = tf.get_variable("hidden1_biases",
        [hidden1_size],
        initializer = tf.random_normal_initializer(stddev=0.01))
      tf.summary.histogram("hidden1_biases", hidden1_biases)
      activation += hidden1_biases
    activation = tf.nn.relu6(activation)
    tf.summary.histogram("hidden1_output", activation)
    #dropout
    activation=tf.nn.dropout(activation,0.5)


    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)
    logging.info("DBoF_activitions:%s", str(activation))
    return aggregated_model().create_model(
        model_input=activation,
        vocab_size=vocab_size,
        **unused_params)

class LstmModel(models.BaseModel):

  def create_model(self, model_input, vocab_size, num_frames, **unused_params):
    """Creates a model which uses a stack of LSTMs to represent the video.

    Args:
      model_input: A 'batch_size' x 'max_frames' x 'num_features' matrix of
                   input features.
      vocab_size: The number of classes in the dataset.
      num_frames: A vector of length 'batch' which indicates the number of
           frames for each video (before padding).

    Returns:
      A dictionary with a tensor containing the probability predictions of the
      model in the 'predictions' key. The dimensions of the tensor are
      'batch_size' x 'num_classes'.
    """
    lstm_size = FLAGS.lstm_cells
    number_of_layers = FLAGS.lstm_layers

    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [
                tf.contrib.rnn.BasicLSTMCell(
                    lstm_size+128, forget_bias=1.0)
                for _ in range(number_of_layers)
                ])

    loss = 0.0

    outputs, state = tf.nn.dynamic_rnn(stacked_lstm, model_input,
                                       sequence_length=num_frames,
                                       dtype=tf.float32)

    aggregated_model = getattr(video_level_models,
                               FLAGS.video_level_classifier_model)

    return aggregated_model().create_model(
        model_input=state[-1].h,
        vocab_size=vocab_size,
        **unused_params)
