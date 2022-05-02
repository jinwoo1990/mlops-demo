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


from absl import logging
from tfx.components.trainer.fn_args_utils import FnArgs
import tensorflow as tf
from tensorflow import keras
import tensorflow_transform as tft
from models import features
from models.keras_model import constants


def _gzip_reader_fn(filenames):
    """Load compressed dataset
    Args:
      filenames - filenames of TFRecords to load
    Returns:
      TFRecordDataset loaded from the filenames
    """

    # Load the dataset. Specify the compression type since it is saved as `.gz`
    return tf.data.TFRecordDataset(filenames, compression_type='GZIP')


def _input_fn(file_pattern,
              tf_transform_output,
              num_epochs=None,
              batch_size=32) -> tf.data.Dataset:
    """Create batches of features and labels from TF Records
    Args:
      file_pattern - List of files or patterns of file paths containing Example records.
      tf_transform_output - transform output graph
      num_epochs - Integer specifying the number of times to read through the dataset.
              If None, cycles through the dataset forever.
      batch_size - An int representing the number of records to combine in a single batch.
    Returns:
      A dataset of dict elements, (or a tuple of dict elements and label).
      Each dict maps feature keys to Tensor or SparseTensor objects.
    """
    transformed_feature_spec = (
        tf_transform_output.transformed_feature_spec().copy())

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=_gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=features.transformed_name(features.LABEL_KEY))

    return dataset


def _get_tf_examples_serving_signature(model, tf_transform_output):
    """Returns a serving signature that accepts `tensorflow.Example`."""

    model.tft_layer_inference = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def serve_tf_examples_fn(serialized_tf_example):
        """Returns the output to be used in the serving signature."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        # Remove label feature since these will not be present at serving time.
        raw_feature_spec.pop(features.LABEL_KEY)
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_inference(raw_features)
        logging.info('serve_transformed_features = %s', transformed_features)

        outputs = model(transformed_features)
        return {'outputs': outputs}

    return serve_tf_examples_fn


def _get_transform_features_signature(model, tf_transform_output):
    """Returns a serving signature that applies tf.Transform to features."""

    model.tft_layer_eval = tf_transform_output.transform_features_layer()

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.string, name='examples')
    ])
    def transform_features_fn(serialized_tf_example):
        """Returns the transformed_features to be fed as input to evaluator."""
        raw_feature_spec = tf_transform_output.raw_feature_spec()
        raw_features = tf.io.parse_example(serialized_tf_example, raw_feature_spec)
        transformed_features = model.tft_layer_eval(raw_features)
        logging.info('eval_transformed_features = %s', transformed_features)
        return transformed_features

    return transform_features_fn


def _model_builder(hp):
    """
    Builds the model and sets up the hyperparameters to tune.
    Args:
      hp - Keras tuner object
    Returns:
      model with hyperparameters to tune
    """
    num_dnn_layers = 3  # 일단 복잡하니까 작은 수로 지정하고 optimize도 제외
    hp_first_units = hp.get('units')
    hp_decay_factor = hp.get('decay')

    hidden_units = [
        max(2, int(hp_first_units * hp_decay_factor ** i))
        for i in range(num_dnn_layers)
    ]

    model = _wide_and_deep_classifier(hidden_units)

    return model


def _wide_and_deep_classifier(hidden_units):
    input_numeric = [
        tf.keras.layers.Input(name=colname, shape=(1,), dtype=tf.float32)
        for colname in features.transformed_names(features.DENSE_FLOAT_FEATURE_KEYS)
    ]
    input_categorical = [
        tf.keras.layers.Input(name=colname, shape=(features.VOCAB_SIZE + features.OOV_SIZE,), dtype=tf.float32)
        for colname in features.transformed_names(features.VOCAB_FEATURE_KEYS)
    ]

    # deep 부분 정의
    deep = tf.keras.layers.concatenate(input_numeric)
    for numnodes in hidden_units:
        deep = tf.keras.layers.Dense(numnodes, name='deep_' + str(numnodes))(deep)
    # wide 부분 정의
    wide = tf.keras.layers.concatenate(input_categorical)
    # concat
    combined = keras.layers.concatenate([wide, deep], name='combined')
    # output 만들기
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(combined)
    # output = tf.squeeze(output, -1)
    # input layers와 output을 넣어 model 만들기
    # 이런 식으로 만들면 input layer에서 분리되서 wide, deep 나뉘고 다시 output으로 합쳐지게 깔끔하게 나옴
    input_layers = input_numeric + input_categorical
    model = tf.keras.Model(input_layers, output)

    # Setup model for training
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=constants.LEARNING_RATE),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    # Print the model summary
    model.summary()

    return model


def run_fn(fn_args: FnArgs) -> None:
    # Callback for TensorBoard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=fn_args.model_run_dir, update_freq='batch')

    # Load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = _input_fn(fn_args.train_files[0], tf_transform_output, batch_size=constants.TRAIN_BATCH_SIZE)
    val_set = _input_fn(fn_args.eval_files[0], tf_transform_output, batch_size=constants.EVAL_BATCH_SIZE)

    # Load best hyperparameters
    if fn_args.hyperparameters:
        hp = fn_args.hyperparameters.get('values')  # TODO: Tuner
    else:
        hp = {'units': constants.FIRST_UNITS,
              'decay': constants.DECAY_FACTOR}

    # Build the model
    model = _model_builder(hp)

    # Train the model
    model.fit(
        train_set,
        epochs=constants.NUM_EPOCHS,  # tensorflow 버전 업하면서 batch마다 tensorboard 그래프가 안 나와서 epoch 지정
        steps_per_epoch=fn_args.train_steps,  # 이렇게 해야 num_steps을 받아옴
        validation_data=val_set,
        validation_steps=fn_args.eval_steps,  # 이렇게 해야 num_steps을 받아옴
        callbacks=[tensorboard_callback])

    # Define default serving signature
    # TODO: 조금 더 제대로 이해해보기
    # SavedModel에 있는 하나의 특성으로 이를 바탕으로 raw data부터 전처리까지 한 후 추론이 가능하도록 만들 수 있음
    # _get_serve_tf_examples_fn 내에 serve_tf_example_fn(tf.function)으로 get_concrete_function을 통해 examples 데이터를 전달해     # 추론 가능하도록 함
    signatures = {
        'serving_default':
            _get_tf_examples_serving_signature(model, tf_transform_output),
        'transform_features':
            _get_transform_features_signature(model, tf_transform_output),
    }

    # Save the model
    model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)
