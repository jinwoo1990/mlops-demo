from kerastuner.engine import base_tuner
import kerastuner as kt
from tensorflow import keras
from typing import NamedTuple, Dict, Text, Any, List
from tfx.components.trainer.fn_args_utils import FnArgs, DataAccessor
import tensorflow as tf
import tensorflow_transform as tft
import advert_constants as constants

# Declare namedtuple field names
TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])
# Callback for the search strategy
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

_DENSE_FLOAT_FEATURE_KEYS = constants.DENSE_FLOAT_FEATURE_KEYS
_VOCAB_FEATURE_KEYS = constants.VOCAB_FEATURE_KEYS
_VOCAB_SIZE = constants.VOCAB_SIZE
_OOV_SIZE = constants.OOV_SIZE
_LABEL_KEY = constants.LABEL_KEY
_transformed_name = constants.transformed_name


def _transformed_names(keys):
    return [_transformed_name(key) for key in keys]


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
        label_key=_transformed_name(_LABEL_KEY))

    return dataset


def _model_builder(hp):
    """
    Builds the model and sets up the hyperparameters to tune.

    Args:
      hp - Keras tuner object

    Returns:
      model with hyperparameters to tune
    """
    
    num_dnn_layers = 3  # ?????? ??????????????? ?????? ?????? ???????????? optimize??? ??????
    hp_first_units = hp.Int('units', min_value=128, max_value=256, step=32)
    hp_decay_factor = hp.Choice('decay', values=[0.7, 0.5, 0.3])
    
    hidden_units = [
         max(2, int(hp_first_units * hp_decay_factor**i))
         for i in range(num_dnn_layers)
    ]
    
    model = _wide_and_deep_classifier(hidden_units)
    
    return model


def _wide_and_deep_classifier(hidden_units):    
    input_numeric = [
        tf.keras.layers.Input(name=colname, shape=(1,), dtype=tf.float32)
        for colname in _transformed_names(_DENSE_FLOAT_FEATURE_KEYS)
    ]
    input_categorical = [
        tf.keras.layers.Input(name=colname, shape=(_VOCAB_SIZE + _OOV_SIZE,), dtype=tf.float32)
        for colname in _transformed_names(_VOCAB_FEATURE_KEYS)
    ]
    
    # TODO: Tuner hyperparameters??? ???????????? ??????
    # deep ?????? ??????
    deep = tf.keras.layers.concatenate(input_numeric)
    for numnodes in hidden_units:
        deep = tf.keras.layers.Dense(numnodes, name='deep_'+str(numnodes))(deep)
    # wide ?????? ??????
    wide = tf.keras.layers.concatenate(input_categorical)
    # concat
    combined = keras.layers.concatenate([wide, deep], name='combined')
    # output ?????????
    output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(combined)
    # input layers??? output??? ?????? model ?????????
    # ?????? ????????? ????????? input layer?????? ???????????? wide, deep ????????? ?????? output?????? ???????????? ???????????? ??????
    input_layers = input_numeric + input_categorical
    model = tf.keras.Model(input_layers, output)
    
    # TODO: learning rate Tuner hyperparameters??? ??????
    # Setup model for training
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss="binary_crossentropy",
                  metrics=[tf.keras.metrics.BinaryAccuracy()])

    # Print the model summary
    model.summary()
    
    return model


def tuner_fn(fn_args: FnArgs) -> None:
    # Define tuner search strategy
    tuner = kt.Hyperband(_model_builder,
                     objective='binary_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory=fn_args.working_dir,
                     project_name='kt_hyperband')
    
    # Load transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    
    # epochs ????????? ??? ????????? steps ?????? steps??? epochs*???????????? ?????? ??? ?????? ???????????? None??? ??????????????? ??? ?????? ??????????????? ???
    # train_steps, eval_steps??? ??? ?????? epochs?????? ???????????? ????????? ??????
    train_set = _input_fn(fn_args.train_files[0], tf_transform_output)  
    val_set = _input_fn(fn_args.eval_files[0], tf_transform_output)
    
    return TunerFnResult(
    tuner=tuner,
    fit_kwargs={ 
      "callbacks":[stop_early],
      'x': train_set,
      'validation_data': val_set,
      'steps_per_epoch': fn_args.train_steps,
      'validation_steps': fn_args.eval_steps}
    )   
