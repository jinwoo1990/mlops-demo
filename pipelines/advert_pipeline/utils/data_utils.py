import os
import pandas as pd
import datetime
from pytz import timezone


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

        
if __name__ == '__main__':
    print('')