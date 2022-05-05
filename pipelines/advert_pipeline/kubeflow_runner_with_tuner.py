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

PIPELINE_ROOT = os.path.join('gs://', configs.GCS_PIPELINE_BUCKET_NAME, 'tfx_pipeline_output',
                             configs.PIPELINE_NAME_BASE)
# SERVING_MODEL_DIR = os.path.join(PIPELINE_ROOT, 'serving_model')
SERVING_MODEL_NAME = 'advert_classifier'
SERVING_MODEL_DIR = os.path.join('gs://', configs.GCS_SERVING_BUCKET_NAME, 'serving_model', SERVING_MODEL_NAME)  # Deploy 환경이 참조할 경로
# SCHEMA_PATH = None  # 처음에 None으로 Schema 생성하면서 파악
SCHEMA_PATH = './schema/schema.pbtxt'  # gs://bucket/tfx_pipeline_output/advert_pipeline/SchemaGen/schema/116  # 정해진 Schema 불러오기
# DATA_PATH = 'gs://{}/data/advertising/'.format(configs.GCS_DATA_BUCKET_NAME)


def run():
    # Metadata config. The defaults works work with the installation of
    # KF Pipelines using Kubeflow. If installing KF Pipelines using the
    # lightweight deployment option, you may need to override the defaults.
    # If you use Kubeflow, metadata will be written to MySQL database inside
    # Kubeflow cluster.
    # kubeflow pipeline 만들 때 설정한 metadata config 가져옴
    # e.g. prefix='xxx'이면 xxx-metadata database로 메타데이터가 저장됨
    # kubernetes configMap에 해당 파일에 username, password db 등 저장됨
    # metadata-mysql-configmap, ml-pipeline-install-config-snapshot (여기 대부분 managed 정보들어있음)
    # # https://github.com/kubeflow/pipelines/tree/master/manifests/kustomize/base/metadata/overlays/db
    metadata_config = tfx.orchestration.experimental.get_default_kubeflow_metadata_config()  # default
    
    # 직접 메타데이터 설정하는 방법 
    # managed storage를 설정하고 kubeflow pipeline 만들면 override가 안 됨
    # managed storage 옵션 없이 하면 적용은 되는데 kubeflow ui 상에서 metadata가 보이지는 않음 (kubeflow가 DB 접근 불가)
    # from tfx.orchestration.kubeflow.proto import kubeflow_pb2   
    # metadata_config = kubeflow_pb2.KubeflowMetadataConfig()
    # 아래 코드가 configMap을 업데이트하지 못 함
    # _MYSQL_HOST = kubeflow_pb2.ConfigValue()
    # _MYSQL_HOST.value = configs.MYSQL_HOST
    # _MYSQL_PORT = kubeflow_pb2.ConfigValue()
    # _MYSQL_PORT.value = configs.MYSQL_PORT
    # _MYSQL_DATABASE = kubeflow_pb2.ConfigValue()
    # _MYSQL_DATABASE.value = configs.MYSQL_DATABASE
    # _MYSQL_USERNAME = kubeflow_pb2.ConfigValue()
    # _MYSQL_USERNAME.value = configs.MYSQL_USERNAME
    # _MYSQL_PASSWORD = kubeflow_pb2.ConfigValue()
    # _MYSQL_PASSWORD.value = configs.MYSQL_PASSWORD
    # metadata_config.mysql_db_service_host.CopyFrom(_MYSQL_HOST)
    # metadata_config.mysql_db_service_port.CopyFrom(_MYSQL_PORT)
    # metadata_config.mysql_db_name.CopyFrom(_MYSQL_DATABASE)
    # metadata_config.mysql_db_user.CopyFrom(_MYSQL_USERNAME)
    # metadata_config.mysql_db_password.CopyFrom(_MYSQL_PASSWORD)

    runner_config = tfx.orchestration.experimental.KubeflowDagRunnerConfig(
        kubeflow_metadata_config=metadata_config,
        tfx_image=configs.PIPELINE_IMAGE_TUNER)
    pod_labels = {
        'add-pod-env': 'true',
        tfx.orchestration.experimental.LABEL_KFP_SDK_ENV: 'tfx-test-1'
    }
    tfx.orchestration.experimental.KubeflowDagRunner(
        config=runner_config, pod_labels_to_attach=pod_labels
    ).run(
        pipeline.create_pipeline(
            pipeline_name=configs.PIPELINE_NAME_TUNER,
            pipeline_root=PIPELINE_ROOT,
            # data_path=DATA_PATH,
            query=configs.BIG_QUERY_QUERY,
            schema_path=SCHEMA_PATH,
            tuner_flag=True,
            preprocessing_fn=configs.PREPROCESSING_FN,
            tuner_fn=configs.TUNER_FN,
            run_fn=configs.RUN_FN,
            train_args=tfx.proto.TrainArgs(num_steps=configs.TRAIN_NUM_STEPS),
            eval_args=tfx.proto.EvalArgs(num_steps=configs.EVAL_NUM_STEPS),
            eval_config=configs.EVAL_CONFIG,
            serving_model_dir=SERVING_MODEL_DIR,
            beam_pipeline_args=configs.BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS
        ))


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    run()
