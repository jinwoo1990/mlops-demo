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
import re
from ast import literal_eval
from google.cloud import storage
from ml_metadata import metadata_store
from ml_metadata.proto import metadata_store_pb2
from tfx import v1 as tfx
from dotenv import load_dotenv


load_dotenv('.env')


def get_latest_hp():
    # # TODO: 테스트 환경에 따라 변경
    # # kubeflow
    # connection_config = metadata_store_pb2.ConnectionConfig()
    # connection_config.mysql.host = os.environ.get('MYSQL_HOST')
    # connection_config.mysql.port = 3306
    # connection_config.mysql.database = os.environ.get('MYSQL_DATABASE')
    # connection_config.mysql.user = os.environ.get('MYSQL_USERNAME')
    # connection_config.mysql.password = os.environ.get('MYSQL_PASSWORD')
    # # store = metadata_store.MetadataStore(connection_config, enable_upgrade_migration=True)
    # store = metadata_store.MetadataStore(connection_config)
    # # local
    # # metadata_path = os.environ.get('LOCAL_METADATA_PATH')
    # # connection_config = tfx.orchestration.metadata.sqlite_metadata_connection_config(metadata_path)
    # # store = metadata_store.MetadataStore(connection_config, enable_upgrade_migration=True)
    
    # artifact_types = store.get_artifact_types()
    # artifact_types = [artifact_type.name for artifact_type in artifact_types]
    # if 'HyperParameters' not in artifact_types:
    #     return None
    # hp_list = store.get_artifacts_by_type('HyperParameters')
    
    # mlmd.downgrade_schema(
    # config=connection_config,
    # downgrade_to_schema_version=6)  # 버전 호환 문제로 작성  # TODO: 없애고 kubeflow 버전 이상한 것 바꾸기, ExampleGen에서는 6이고, 여기서는 8 요구
    
    # return hp_list[-1].uri
    
    # 버전 에러로 잠시 코드 변경
    # TODO: 원하는 경로 설정
    hp_uri = ''
    return hp_uri


def load_hp_from_uri(uri):
    if uri.startswith('gs://'):
        pat = re.compile('gs:\/\/([\w\-\_]+)/(\S+)')
        mat = pat.search(uri)
        bucket_name, uri_prefix = mat.group(1), mat.group(2)
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        blob = bucket.get_blob(os.path.join(uri_prefix, 'best_hyperparameters.txt'))
        downloaded_file = blob.download_as_text(encoding="utf-8")
    else:
        uri_with_filename = os.path.join(uri, 'best_hyperparameters.txt')
        downloaded_file = open(uri_with_filename, "r").readline()
    modified_file = downloaded_file.replace('null', '"null"').replace('true', "True")
    hp_dict = literal_eval(modified_file)
    
    return hp_dict


# try:
#     hp_uri = get_latest_hp(METADATA_PATH)
# except:
#     hp_uri = None
hp_uri = get_latest_hp()

if hp_uri:
    hp = load_hp_from_uri(hp_uri).get('values')
    FIRST_UNITS = hp.get('units')
    DECAY_FACTOR = hp.get('decay')
else:
    FIRST_UNITS = 128
    DECAY_FACTOR = 0.7

LEARNING_RATE = 0.001

NUM_EPOCHS = 2
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 32
