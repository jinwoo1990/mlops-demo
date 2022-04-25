# Copyright 2020 Google LLC. All Rights Reserved.
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


import tensorflow as tf
import tensorflow_transform as tft
from models import features  # 여기서 constants들 불러옴


def _fill_in_missing(x):
    """Replace missing values in a SparseTensor.

    Fills in missing values of `x` with '' or 0, and converts to a dense tensor.

    Args:
    x: A `SparseTensor` of rank 2.  Its dense shape should have size at most 1
      in the second dimension.

    Returns:
    A rank 1 tensor where missing values of `x` have been filled in.
    """
    if not isinstance(x, tf.sparse.SparseTensor):
        return x

    default_value = '' if x.dtype == tf.string else 0
    return tf.squeeze(
      tf.sparse.to_dense(
          tf.SparseTensor(x.indices, x.values, [x.dense_shape[0], 1]),
          default_value), axis=1)


def preprocessing_fn(inputs):
    """
    tf.transform's callback function for preprocessing inputs.
     Args:
       inputs: map from feature keys to raw not-yet-transformed features.
     Returns:
       Map from string feature key to transformed feature operations.
     """
    outputs = {}

    # Transform numeric feature
    for key in features.DENSE_FLOAT_FEATURE_KEYS:
        scaled_dense = tft.scale_to_z_score(
            _fill_in_missing(inputs[key]))
        outputs[features.transformed_name(key)] = tf.reshape(scaled_dense, [-1])

    # Transform categorical feature
    for key in features.VOCAB_FEATURE_KEYS:
        indices_dense = tft.compute_and_apply_vocabulary(
            _fill_in_missing(inputs[key]),
            top_k=features.VOCAB_SIZE,
            num_oov_buckets=features.OOV_SIZE)
        one_hot = tf.one_hot(indices_dense, features.VOCAB_SIZE + features.OOV_SIZE)
        outputs[features.transformed_name(key)] = tf.reshape(one_hot, [-1, features.VOCAB_SIZE + features.OOV_SIZE])

    # Transform target
    # 이름만 바뀜
    label_dense = _fill_in_missing(inputs[features.LABEL_KEY])
    outputs[features.transformed_name(features.LABEL_KEY)] = tf.reshape(label_dense, [-1])

    return outputs
