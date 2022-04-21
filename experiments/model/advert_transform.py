import tensorflow as tf
import tensorflow_transform as tft
import advert_constants as constants


# Constants 불러오기
_DENSE_FLOAT_FEATURE_KEYS = constants.DENSE_FLOAT_FEATURE_KEYS
_LABEL_KEY = constants.LABEL_KEY
_VOCAB_FEATURE_KEYS = constants.VOCAB_FEATURE_KEYS
_VOCAB_SIZE = constants.VOCAB_SIZE
_OOV_SIZE = constants.OOV_SIZE
# 함수 불러오기
_transformed_name = constants.transformed_name


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
    for key in _DENSE_FLOAT_FEATURE_KEYS:
        # Preserve this feature as a dense float, setting nan's to the mean.
        # outputs[_transformed_name(key)] = tft.scale_to_z_score(
        #    inputs[key])
        scaled = tft.scale_to_z_score(inputs[key])
        scaled_dense = tf.sparse.to_dense(scaled)
        outputs[_transformed_name(key)] = tf.reshape(scaled_dense, [-1])

    # Transform categorical feature
    for key in _VOCAB_FEATURE_KEYS:
        # Build a vocabulary for this feature.
        # outputs[_transformed_name(key)] = tft.compute_and_apply_vocabulary(
        #    inputs[key],
        #    top_k=_VOCAB_SIZE,
        #    num_oov_buckets=_OOV_SIZE)
        indices = tft.compute_and_apply_vocabulary(inputs[key],
                                                   top_k=_VOCAB_SIZE,
                                                   num_oov_buckets=_OOV_SIZE)
        indices_dense = tf.sparse.to_dense(indices)
        one_hot = tf.one_hot(indices_dense, _VOCAB_SIZE + _OOV_SIZE)
        outputs[_transformed_name(key)] = tf.reshape(one_hot, [-1, _VOCAB_SIZE + _OOV_SIZE])

    # Transform target
    # 이름만 바뀜
    # outputs[_transformed_name(_LABEL_KEY)] = inputs[_LABEL_KEY]
    label_dense = tf.sparse.to_dense(inputs[_LABEL_KEY])
    outputs[_transformed_name(_LABEL_KEY)] = tf.reshape(label_dense, [-1])
    # outputs[_transformed_name(_LABEL_KEY)] = tf.cast(inputs[_LABEL_KEY], tf.float32)

    return outputs
