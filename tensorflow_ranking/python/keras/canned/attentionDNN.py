# Copyright 2020 The TensorFlow Ranking Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""DNN Ranking network in Keras."""

import tensorflow.compat.v2 as tf
import tensorflow_addons as tfa

from tensorflow_ranking.python.keras import network as network_lib
from tensorflow_ranking.python.keras.canned import custom_layers


class AttentionDNNRankingNetwork(network_lib.MultivariateAttentionRankingNetwork):
  """Deep Neural Network (DNN) scoring based univariate ranking network."""

  def __init__(self,
               context_feature_columns=None,
               example_feature_columns=None,
               num_cnn_filter=256,
               head_size=64,
               num_head=4,
               hidden_layer_dims=None,
               activation=None,
               use_batch_norm=True,
               batch_norm_moment=0.999,
               dropout=0.5,
               name='attention_dnn_ranking_network',
               **kwargs):
    
    if not example_feature_columns or not hidden_layer_dims:
      raise ValueError('example_feature_columns or hidden_layer_dims must not '
                       'be empty.')
    super(AttentionDNNRankingNetwork, self).__init__(
        context_feature_columns=context_feature_columns,
        example_feature_columns=example_feature_columns,
        name=name,
        **kwargs)
    self._num_cnn_filter = num_cnn_filter
    self._hidden_layer_dims = [int(d) for d in hidden_layer_dims]
    self._batch_norm_moment = batch_norm_moment
    self._activation = activation
    self._use_batch_norm = use_batch_norm
    self._dropout = dropout
    self._num_head = num_head
    self._head_size = head_size

    self._cnn_layer = tf.keras.layers.Conv1D(
      filters=self._num_cnn_filter,
      kernel_size=1,
      padding='same')

    layers = []
    if self._use_batch_norm:
      layers.append(
          tf.keras.layers.BatchNormalization(momentum=self._batch_norm_moment))
    for _, layer_width in enumerate(self._hidden_layer_dims):
      layers.append(tf.keras.layers.Dense(units=layer_width))
      if self._use_batch_norm:
        layers.append(
            tf.keras.layers.BatchNormalization(
                momentum=self._batch_norm_moment))
      layers.append(tf.keras.layers.Activation(activation=self._activation))
      # layers.append(tf.keras.layers.Dropout(rate=self._dropout))

    self._attention_layer = tfa.layers.MultiHeadAttention(head_size=self._head_size, num_heads=self._num_head)

    self._scoring_layers = layers
    self._output_score_layer = tf.keras.layers.Dense(units=1, activation='relu')


  def score(self, context_features=None, example_features=None, training=True):
    """Univariate scoring of context and one example to generate a score.

    Args:
      context_features: (dict) context feature names to 2D tensors of shape
        [batch_size, 1].
      example_features: (dict) example feature names to 2D tensors of shape
        [batch_size, list_size].
      training: (bool) whether in training or inference mode.

    Returns:
      (tf.Tensor) A score tensor of shape [batch_size, 1].
    """

    context_input = []
    example_input = []
	
	# we assume context and example features are same for now
    for name in self.example_feature_columns:
      context_input.append(context_features[name])
      example_input.append(example_features[name])

    context_input = tf.concat(context_input, -1)
    example_input = tf.concat(example_input, -1)

    context_projection = self._cnn_layer(context_input)
    example_projection = self._cnn_layer(example_input)

    # query_value_attention_seq = tf.keras.layers.Attention()([context_projection, example_projection])
    # query_value_attention_seq = custom_layers.MultiHeadSelfAttention(self._num_cnn_filter, self._num_head)([context_projection, example_projection])
    query_value_attention_seq = self._attention_layer([context_projection, example_projection])

    score_layer_input = tf.keras.layers.Flatten()(query_value_attention_seq)

    outputs = score_layer_input
    for layer in self._scoring_layers:
      outputs = layer(outputs, training=training)

    return self._output_score_layer(outputs, training=training)

  def get_config(self):
    config = super(AttentionDNNRankingNetwork, self).get_config()
    config.update({
        'num_cnn_filter': self._num_cnn_filter,
        'hidden_layer_dims': self._hidden_layer_dims,
        'activation': self._activation,
        'use_batch_norm': self._use_batch_norm,
        'batch_norm_moment': self._batch_norm_moment,
        'dropout': self._dropout,
    })
    return config
