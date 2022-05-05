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
                             configs.PIPELINE_NAME_BASE)
METADATA_PATH = os.path.join(OUTPUT_DIR, 'tfx_metadata', configs.PIPELINE_NAME_BASE,
                             'metadata.db')


# The last component of the pipeline, "Pusher" will produce serving model under
# SERVING_MODEL_DIR.
SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')
# SCHEMA_PATH = None  # 처음에 None으로 Schema 생성하면서 파악
SCHEMA_PATH = './schema/schema.pbtxt'  # gs://bucket/tfx_pipeline_output/advert_pipeline/SchemaGen/schema/116  # 정해진 Schema 불러오기
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data/csv')


def run():
    metadata_config = tfx.orchestration.metadata.sqlite_metadata_connection_config(
        METADATA_PATH)  # local 경로로 할 때
    # metadata_config = tfx.orchestration.metadata.mysql_metadata_connection_config(
    #     host=configs.MYSQL_HOST,
    #     port=configs.MYSQL_PORT,
    #     database=configs.MYSQL_DATABASE,
    #     username=configs.MYSQL_USERNAME,
    #     password=configs.MYSQL_PASSWORD)

    tfx.orchestration.LocalDagRunner().run(
        pipeline.create_pipeline(
          pipeline_name=configs.PIPELINE_NAME_DAILY,
          pipeline_root=PIPELINE_ROOT,
          query='None',
          schema_path=SCHEMA_PATH,
          data_path=DATA_PATH,
          preprocessing_fn=configs.PREPROCESSING_FN,
          run_fn=configs.RUN_FN,
          train_args=tfx.proto.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
          eval_args=tfx.proto.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
          eval_config=configs.EVAL_CONFIG,
          serving_model_dir=SERVING_MODEL_DIR,
          metadata_connection_config=metadata_config)
    )


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()
