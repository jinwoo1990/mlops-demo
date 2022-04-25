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

import ml_metadata as mlmd
from ml_metadata.metadata_store import metadata_store
from ml_metadata.proto import metadata_store_pb2


OUTPUT_DIR = os.path.join('gs://', configs.GCS_BUCKET_NAME)
PIPELINE_ROOT = os.path.join(OUTPUT_DIR, 'tfx_pipeline_output',
                             configs.PIPELINE_NAME)
# TODO: 경로 변경
# SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')
SERVING_MODEL_DIR = os.path.join('gs://tfx-project-348009-bucket', 'serving_model')  # Deploy 환경이 참조할 경로
DATA_PATH = 'gs://{}/tfx-template/data/taxi/'.format(configs.GCS_BUCKET_NAME)


def run():
    # Metadata config. The defaults works work with the installation of
    # KF Pipelines using Kubeflow. If installing KF Pipelines using the
    # lightweight deployment option, you may need to override the defaults.
    # If you use Kubeflow, metadata will be written to MySQL database inside
    # Kubeflow cluster.
    # metadata_config = tfx.orchestration.experimental.get_default_kubeflow_metadata_config(
    # )
    # TODO: 작동 확인 및 암호화 생각
    metadata_config = tfx.orchestration.metadata.mysql_metadata_connection_config(
             host="35.194.41.63", 
             port=3306, 
             database="ml_metadata", 
             username="root", 
             password="0000")

    runner_config = tfx.orchestration.experimental.KubeflowDagRunnerConfig(
      kubeflow_metadata_config=metadata_config,
      tfx_image=configs.PIPELINE_IMAGE)
    pod_labels = {
      'add-pod-env': 'true',
      tfx.orchestration.experimental.LABEL_KFP_SDK_ENV: 'tfx-test'
    }
    tfx.orchestration.experimental.KubeflowDagRunner(
      config=runner_config, pod_labels_to_attach=pod_labels
    ).run(
      pipeline.create_pipeline(
          pipeline_name=configs.PIPELINE_NAME,
          pipeline_root=PIPELINE_ROOT,
          data_path=DATA_PATH,
          # TODO(step 7): (Optional) Uncomment below to use BigQueryExampleGen.
          # query=configs.BIG_QUERY_QUERY,
          # TODO(step 5): (Optional) Set the path of the customized schema.
          # schema_path=generated_schema_path,
          preprocessing_fn=configs.PREPROCESSING_FN,
          run_fn=configs.RUN_FN,
          train_args=tfx.proto.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
          eval_args=tfx.proto.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
          eval_accuracy_threshold=configs.EVAL_ACCURACY_THRESHOLD,
          serving_model_dir=SERVING_MODEL_DIR,
          # TODO(step 7): (Optional) Uncomment below to use provide GCP related
          #               config for BigQuery with Beam DirectRunner.
          # beam_pipeline_args=configs
    ))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()
