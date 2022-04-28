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
from dotenv import load_dotenv


PIPELINE_NAME = 'advert_pipeline'

# 기본 GCP 설정
# Project 이름 불러오기
try:
    import google.auth  # pylint: disable=g-import-not-at-top  # pytype: disable=import-error
    try:
        _, GOOGLE_CLOUD_PROJECT = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError:
        GOOGLE_CLOUD_PROJECT = ''
except ImportError:
    GOOGLE_CLOUD_PROJECT = ''
# Storage 이름 설정
# GCS_PIPELINE_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-kubeflowpipelines-default'  # default
GCS_PIPELINE_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-pipelines'
GCS_DATA_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-data'
GCS_SERVING_BUCKET_NAME = GOOGLE_CLOUD_PROJECT + '-serving'
# Region 설정
GOOGLE_CLOUD_REGION = 'us-central1-a'
# 파이프라인 이미지 빌드할 registry 경로
# This image will be automatically built by CLI if we use --build-image flag.
PIPELINE_IMAGE = f'gcr.io/{GOOGLE_CLOUD_PROJECT}/{PIPELINE_NAME}'

# Metadata 설정
# 비밀번호가 포함되어 따로 관리
load_dotenv()  # load secret env
MYSQL_HOST = os.environ.get('MYSQL_HOST')
MYSQL_PORT = os.environ.get('MYSQL_PORT')
MYSQL_DATABASE = os.environ.get('MYSQL_DATABASE')
MYSQL_USERNAME = os.environ.get('MYSQL_USERNAME')
MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD')

# TODO: csv -> mysql로 변경하면서 수정
# 데이터 쿼리 설정
# Beam args to use BigQueryExampleGen with Beam DirectRunner.
# TODO(step 7): (Optional) Uncomment here to provide GCP related configs for
#               BigQuery.
# BIG_QUERY_WITH_DIRECT_RUNNER_BEAM_PIPELINE_ARGS = [
#    '--project=' + GOOGLE_CLOUD_PROJECT,
#    '--temp_location=' + os.path.join('gs://', GCS_BUCKET_NAME, 'tmp'),
#    ]

# The rate at which to sample rows from the Chicago Taxi dataset using BigQuery.
# The full taxi dataset is > 120M record.  In the interest of resource
# savings and time, we've set the default for this example to be much smaller.
# Feel free to crank it up and process the full dataset!
_query_sample_rate = 0.0001  # Generate a 0.01% random sample.

# The query that extracts the examples from BigQuery.  The Chicago Taxi dataset
# used for this example is a public dataset available on Google AI Platform.
# https://console.cloud.google.com/marketplace/details/city-of-chicago-public-data/chicago-taxi-trips
# TODO(step 7): (Optional) Uncomment here to use BigQuery.
# BIG_QUERY_QUERY = """
#         SELECT
#           pickup_community_area,
#           fare,
#           EXTRACT(MONTH FROM trip_start_timestamp) AS trip_start_month,
#           EXTRACT(HOUR FROM trip_start_timestamp) AS trip_start_hour,
#           EXTRACT(DAYOFWEEK FROM trip_start_timestamp) AS trip_start_day,
#           UNIX_SECONDS(trip_start_timestamp) AS trip_start_timestamp,
#           pickup_latitude,
#           pickup_longitude,
#           dropoff_latitude,
#           dropoff_longitude,
#           trip_miles,
#           pickup_census_tract,
#           dropoff_census_tract,
#           payment_type,
#           company,
#           trip_seconds,
#           dropoff_community_area,
#           tips,
#           IF(tips > fare * 0.2, 1, 0) AS big_tipper
#         FROM `bigquery-public-data.chicago_taxi_trips.taxi_trips`
#         WHERE (ABS(FARM_FINGERPRINT(unique_key)) / 0x7FFFFFFFFFFFFFFF)
#           < {query_sample_rate}""".format(
#    query_sample_rate=_query_sample_rate)

# TFX 설정
# Function 경로
PREPROCESSING_FN = 'models.preprocessing.preprocessing_fn'
RUN_FN = 'models.keras_model.model.run_fn'
# Trainer 파라미터
TRAIN_NUM_STEPS = 1000
EVAL_NUM_STEPS = 150
# Evaluator 설정
EVAL_CONFIG = """
  ## Model information
  model_specs {
    # This assumes a serving model with signature 'serving_default'.
    signature_name: "serving_default",
    label_key: "ClickedOnAd"  # raw feature name or transformed feature name 둘 다 가능
  }

  ## Post training metric information
  metrics_specs {
    metrics { class_name: "ExampleCount" }
    metrics {
      class_name: "BinaryAccuracy"
      threshold {
        # Ensure that metric is always > 0.5
        value_threshold {
          lower_bound { value: 0.5 }  # TODO: 수정
        }
        # Ensure that metric does not drop by more than a small epsilon
        # e.g. (candidate - baseline) > -1e-10 or candidate > baseline - 1e-10
        change_threshold {
          direction: HIGHER_IS_BETTER
          absolute { value: -1e-10 }
        }
      }
    }
    metrics { class_name: "BinaryCrossentropy" }
    metrics { class_name: "AUC" }
    metrics { class_name: "AUCPrecisionRecall" }
    metrics { class_name: "Precision" }
    metrics { class_name: "Recall" }
    # ... add additional metrics and plots ...
  }

  ## Slicing information
  slicing_specs {}  # overall slice
  slicing_specs {
    feature_keys: ["Male"]
  }
  slicing_specs {
    feature_values: {
      key: "Country"
      value: "Senegal"}
  }
  # slicing_specs {
  #   feature_keys: ["Male", "Age"]
  # }

"""