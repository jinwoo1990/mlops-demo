from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
from pytz import timezone
from airflow.models import DAG
from airflow.contrib.operators.gcs_to_bq import GoogleCloudStorageToBigQueryOperator


# TODO: connection 관련 에러 수정
now = datetime.now(timezone('Asia/Seoul')).strftime('%Y%m%d')

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2022, 5, 3),
    'email': [''],
    'email_on_faiure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

bucket = 'project-111111-bucket'
filename = 'data/advertising/%s/*.parquet' % now  # project-111111-data/data/advertising

# dag = DAG('gcs_to_bq_dag', default_args=default_args, schedule_interval='0 * * * *')  # 1시간마다
dag = DAG('gcs_to_bq_dag', default_args=default_args, schedule_interval='@daily')

gcsToBigQuery = GoogleCloudStorageToBigQueryOperator(
    bigquery_conn_id='airflow-sa-all',
    google_cloud_storage_conn_id='airflow-sa-all'
    task_id='gcs_to_bq', 
    destination_project_dataset_table='', # project.dataset.table
    bucket=bucket, 
    source_objects=[filename],
    write_disposition='WRITE_APPEND',
    source_format='PARQUET',
    dag=dag)