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


import os
from absl import logging

from tfx import v1 as tfx
from pipeline import configs
from pipeline import pipeline

# TFX pipeline produces many output files and metadata. All output data will be
# stored under this OUTPUT_DIR.
# NOTE: It is recommended to have a separated OUTPUT_DIR which is *outside* of
#       the source code structure. Please change OUTPUT_DIR to other location
#       where we can store outputs of the pipeline.
HOME_DIR = os.path.expanduser('~')
OUTPUT_DIR = os.path.join(HOME_DIR, 'temp/tfx-output')
try:
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
except OSError:
    print('Failed to creat directory.')

# TFX produces two types of outputs, files and metadata.
# - Files will be created under PIPELINE_ROOT directory.
# - Metadata will be written to SQLite database in METADATA_PATH.
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output',
                             configs.PIPELINE_NAME)
METADATA_PATH = os.path.join(OUTPUT_DIR, 'tfx_metadata', configs.PIPELINE_NAME,
                             'metadata.db')

# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')

# Specifies data file directory. DATA_PATH should be a directory containing CSV
# files for CsvExampleGen in this example. By default, data files are in the
# `data` directory.
# NOTE: If you upload data files to GCS(which is recommended if you use
#       Kubeflow), you can use a path starting "gs://YOUR_BUCKET_NAME/path" for
#       DATA_PATH. For example,
#       DATA_PATH = 'gs://bucket/chicago_taxi_trips/csv/'.
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def run():
    tfx.orchestration.LocalDagRunner().run(
      pipeline.create_pipeline(
          pipeline_name=configs.PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          data_path=DATA_PATH,
          preprocessing_fn=configs.PREPROCESSING_FN,
          run_fn=configs.RUN_FN,
          train_args=tfx.proto.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
          eval_args=tfx.proto.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
          eval_config=configs.EVAL_CONFIG,
          serving_model_dir=SERVING_MODEL_DIR,
          metadata_connection_config=tfx.orchestration.metadata
          .sqlite_metadata_connection_config(METADATA_PATH)))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()