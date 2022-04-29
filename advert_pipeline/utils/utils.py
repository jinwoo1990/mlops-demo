import os
import pandas as pd
import datetime
from pytz import timezone
import tensorflow as tf
import tensorflow_data_validation as tfdv
from tensorflow_metadata.proto.v0 import anomalies_pb2
from tensorflow_data_validation.utils import io_util
from google.cloud import storage


def csv_to_parquet(path):
    for item in ['advert_1', 'advert_2', 'advert_3']:
        df = pd.read_csv(os.path.join(path, 'csv/%s.csv' % item))
        df.to_parquet(os.path.join(path, 'parquet/%s.parquet' % item))


# 이 방식으로 바꾸면 문제 생김. 그냥 쿼리에서 처리하는 것이 나아 보임
def change_parquet_datetime(path):
    now = datetime.datetime.now(timezone('Asia/Seoul'))
    print(now)
    first_file = pd.read_parquet(os.path.join(path, 'advert_1.parquet'))
    base = pd.to_datetime(first_file['Timestamp'].loc[0])
    base = datetime.datetime(base.year, base.month, base.day, tzinfo=timezone('Asia/Seoul'))
    print(base)
    print((now - base).days)

    day_delta = now.day - base.day
    print(day_delta)
    for item in ['advert_1', 'advert_2', 'advert_3']:
        df_parquet = pd.read_parquet(os.path.join(path, '%s.parquet' % item))
        print(df_parquet.head())
        df_parquet['Timestamp'] = pd.to_datetime(df_parquet['Timestamp'])
        df_parquet['Timestamp'] = df_parquet['Timestamp'] + datetime.timedelta(days=day_delta)
        # 2022-05-01T23:46:51Z
        # BigQueryExample does not support timestamp type
        df_parquet['Timestamp'] = df_parquet['Timestamp'].astype('str')
        df_parquet.to_parquet(os.path.join(path, '%s.parquet' % item))
        print(df_parquet.head())
        print(df_parquet.info())

        
def parse_bucket_name_prefix(bucket_uri):
    import re
    
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


def load_sample_tfrecord_from_uri(uri, sample_number, compression_type="GZIP"):
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


if __name__ == '__main__':
    print('')
    # csv_to_parquet('./../data')
    # change_parquet_datetime('./../data/parquet')
