import os
import pandas as pd
import datetime
from pytz import timezone
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_data_validation.utils import io_util
from google.cloud import storage
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.proto import metrics_for_slice_pb2
from tensorflow_model_analysis.proto import validation_result_pb2
import gcsfs
import json
import re

        
def parse_bucket_name_prefix(bucket_uri):
    pat = re.compile('gs:\/\/([\w\-\_]+)/(\S+)')
    mat = pat.search(bucket_uri)
    
    return mat.group(1), mat.group(2)        


def list_bucket_uri_dir(bucket_uri):
    bucket_name, uri_prefix = parse_bucket_name_prefix(bucket_uri)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=uri_prefix)

    res = ['gs://' + bucket_name + '/' + item.name for item in blobs]
    
    return res


def print_sample_tfrecord_from_uri(uri, sample_number, compression_type="GZIP"):
    if uri.startswith('gs://'):
        tfrecord_filenames = list_bucket_uri_dir(uri)
    else:
        tfrecord_filenames = [os.path.join(uri, name)
                             for name in os.listdir(uri)]
    dataset = tf.data.TFRecordDataset(tfrecord_filenames, compression_type)
    
    for tfrecord in dataset.take(sample_number):
        serialized_example = tfrecord.numpy()
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        print(example)


def load_stats_from_uri(uri):
    """
    Split-train 까지 uri 명시해줘야 함. 그렇지 않으면 google cloud python sdk에서 0번째 인덱스로 파일 경로가 아닌
    자기 자신 경로 아웃풋으로 내보내서 에러 발생
    """
    if uri.startswith('gs://'):
        file_name = list_bucket_uri_dir(uri)[0]
        file_format = file_name.split('.')[1]
        file_path = file_name
    else:
        file_name = os.listdir(uri)[0]
        file_format = file_name.split('.')[1]
        file_path = os.path.join(uri, file_name)
    
    if file_format == 'pb':
        result = tfdv.load_stats_binary(file_path)
    elif file_format == 'pbtxt':
        result = tfdv.load_stats_text(file_path)
    else:
        result = None
    
    return result


def load_schema_from_uri(uri):
    if uri.startswith('gs://'):
        file_name = list_bucket_uri_dir(uri)[1]  # list_bucket_uri_dir 자기 자신 경로 때문에 1로
        file_path = file_name
    else:
        file_name = os.listdir(uri)[0]
        file_path = os.path.join(uri, file_name)
    
    result = tfdv.load_schema_text(file_path)
    
    return result


def load_anomalies_from_uri(uri):
    if uri.startswith('gs://'):
        file_name = list_bucket_uri_dir(uri)[1]  # list_bucket_uri_dir 자기 자신 경로 때문에 1로
        print(file_name)
        file_format = file_name.split('.')[1]
        file_path = file_name
    else:
        file_name = os.listdir(uri)[0]
        file_format = file_name.split('.')[1]
        file_path = os.path.join(uri, file_name)
    
    if file_format == 'pb':
        # Source code
        anomalies_proto = anomalies_pb2.Anomalies()
        anomalies_proto.ParseFromString(io_util.read_file_to_string(
          file_path, binary_mode=True))  # binary mode true로 해서 parsing
        result = anomalies_proto      
    elif file_format == 'pbtxt':
        result = tfdv.load_anomalies_binary(file_path)
        # Source code
        # anomalies = anomalies_pb2.Anomalies()
        # anomalies_text = io_util.read_file_to_string(input_path)
        # text_format.Parse(anomalies_text, anomalies)
        # result = anomalies
    else:
        result = None
    
    return result


def load_eval_attribution_from_uri(uri):
    data = tf.data.TFRecordDataset(uri)
    
    for item in data:
        attribution = metrics_for_slice_pb2.AttributionsForSlice.FromString(item.numpy())
        print(attribution)

        
def load_eval_result_from_uri(uri):
    """
    합쳐진 전체 결과 나옴
    개별 파일 uri 아닌 evaluation uri 입력
    개별 요소 확인
    result.attributions
    result.config
    result.data_location
    result.model_location
    result.file_format
    result.plots
    result.slicing_metrics
    참고로 blessed는 splits들의 metrics 결과들을 모두 보고 판단
    """
    result = tfma.load_eval_result(uri)
    # tfma.view.render_slicing_metrics(eval_result, slicing_spec = tfma.slicer.SingleSliceSpec())
    
    return result


def load_eval_config_from_uri(uri):
    gcs_file_system = gcsfs.GCSFileSystem()
    
    with gcs_file_system.open(uri) as f:
        json_dict = json.load(f)
    return json_dict


def load_blessing_from_uri(uri):
    if uri.startswith('gs://'):
        file_name = list_bucket_uri_dir(uri)[1]  # list_bucket_uri_dir 자기 자신 경로 때문에 1로
        file_path = file_name
    else:
        file_name = os.listdir(uri)[0]
        file_path = os.path.join(uri, file_name)
    
    pat = re.compile(uri+'/(\S+)')
    mat = pat.search(a)
    
    return mat.group(1)


def print_eval_attribution_from_uri(uri):
    data = tf.data.TFRecordDataset(uri)
    
    for item in data:
        line = metrics_for_slice_pb2.AttributionsForSlice.FromString(item.numpy())
        print(line)
        

def print_eval_metrics_from_uri(uri):
    """
    같은 metrics라도 candidate, baseline 쌍의 결과가 [값, 차이값, example_weight 적용값] 3가지로 나옴
    example_weight은 설정 안하면 일반 값이랑 똑같이 나옴
    https://www.tensorflow.org/tfx/model_analysis/metrics (metrics 설정 참조)
    """
    data = tf.data.TFRecordDataset(uri)
    
    for item in data:
        line = metrics_for_slice_pb2.MetricsForSlice.FromString(item.numpy())
        print(line)

        
def print_eval_plots_from_uri(uri):
    data = tf.data.TFRecordDataset(uri)
    
    for item in data:
        line = metrics_for_slice_pb2.PlotsForSlice.FromString(item.numpy())
        print(line)
        

def print_eval_validations_from_uri(uri):
    data = tf.data.TFRecordDataset(uri)
    
    for item in data:
        line = validation_result_pb2.ValidationResult.FromString(item.numpy())
        print(line)
        

if __name__ == '__main__':
    print('')
    # csv_to_parquet('./../data')
    # change_parquet_datetime('./../data/parquet')
