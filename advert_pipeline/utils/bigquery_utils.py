from google.cloud import bigquery


BIGQUERY_TABLE = ''  # TODO: 프로젝트에 맞는 테이블로 변경
NEW_BIGQUERY_TABLE = ''  # TODO: 프로젝트에 맞는 테이블로 변경

# 환경에 따라 table 명 바꾸기
# {2}, %Y 같은 값 때문에 python string formatting이 작동하게 바꾸는 과정에 원래 쿼리랑 다른 str 값 포함 (e.g. {{2}})
# print된 값 참조
NEW_QUERY = f"""
WITH v AS (
  WITH d AS (
    WITH b AS (
      SELECT
        CAST(Timestamp as timestamp) as base,
      FROM `tfx-project-348306.advertising.advert_2022` 
      ORDER BY Timestamp 
      LIMIT 1
    )
    SELECT
      current_date('Asia/Seoul') as day_1,
      TIMESTAMP_ADD(current_date('Asia/Seoul'), INTERVAL 1 DAY) as day_2,
      TIMESTAMP_ADD(current_date('Asia/Seoul'), INTERVAL 2 DAY) as day_3,
      b.base as ori_day_1,
      TIMESTAMP_ADD(b.base, INTERVAL 1 DAY) as ori_day_2,
      TIMESTAMP_ADD(b.base, INTERVAL 2 DAY) as ori_day_3,
    FROM b
    )
  SELECT
    FORMAT_DATE("%Y-%m-%d", d.day_1) AS day_1_str,
    FORMAT_DATE("%Y-%m-%d", d.day_2) AS day_2_str,
    FORMAT_DATE("%Y-%m-%d", d.day_3) AS day_3_str,
    FORMAT_DATE("%Y-%m-%d", d.ori_day_1) AS ori_day_1_str,
    FORMAT_DATE("%Y-%m-%d", d.ori_day_2) AS ori_day_2_str,
    FORMAT_DATE("%Y-%m-%d", d.ori_day_3) AS ori_day_3_str,
    -- CONCAT('2022-', EXTRACT(MONTH FROM d.day_1), '-', EXTRACT(DAY FROM d.day_1)) AS day_1_str,
    -- CONCAT('2022-', EXTRACT(MONTH FROM d.day_2), '-', EXTRACT(DAY FROM d.day_2)) AS day_2_str,
    -- CONCAT('2022-', EXTRACT(MONTH FROM d.day_3), '-', EXTRACT(DAY FROM d.day_3)) AS day_3_str,
  FROM d
  )
SELECT
  DailyTimeSpentOnSite, 
  Age, 
  AreaIncome, 
  DailyInternetUsage, 
  AdTopicLine, 
  City, 
  Male, 
  Country,
  CONCAT(v.day_1_str, regexp_extract(t.Timestamp, 'T[0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}}Z')) AS Timestamp,
  ClickedOnAd
FROM v, `{BIGQUERY_TABLE}` AS t
WHERE Timestamp BETWEEN CONCAT(v.ori_day_1_str, 'T00:00Z') AND CONCAT(v.ori_day_1_str, 'T23:59:59Z')
UNION ALL
SELECT
  DailyTimeSpentOnSite, 
  Age, 
  AreaIncome, 
  DailyInternetUsage, 
  AdTopicLine, 
  City, 
  Male, 
  Country,
  CONCAT(v.day_2_str, regexp_extract(t.Timestamp, 'T[0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}}Z')) AS Timestamp,
  ClickedOnAd
FROM v, `{BIGQUERY_TABLE}` AS t
WHERE Timestamp BETWEEN CONCAT(v.ori_day_2_str, 'T00:00Z') AND CONCAT(v.ori_day_2_str, 'T23:59:59Z')
UNION ALL
SELECT
  DailyTimeSpentOnSite, 
  Age, 
  AreaIncome, 
  DailyInternetUsage, 
  AdTopicLine, 
  City, 
  Male, 
  Country,
  CONCAT(v.day_3_str, regexp_extract(t.Timestamp, 'T[0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}}Z')) AS Timestamp,
  ClickedOnAd
FROM v, `{BIGQUERY_TABLE}` AS t
WHERE Timestamp BETWEEN CONCAT(v.ori_day_3_str, 'T00:00Z') AND CONCAT(v.ori_day_3_str, 'T23:59:59Z')
;
"""


def get_query_result(query, num):
    client = bigquery.Client()
    query_job = client.query(query)
    results = query_job.result()
    
    i = 0
    for row in results:
        if i == num:
            break
        # print(row)
        print(row.Timestamp, row.ClickedOnAd)
        i += 1

        
def create_table_from_query(query, dest_table):
    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(destination=dest_table)
    job_config.write_disposition = "WRITE_TRUNCATE"  # 테이블 존재하면 삭제하고 덮어씌움

    query_job = client.query(query, job_config=job_config)  # Make an API request.
    query_job.result() 

    print("Query results loaded to the table {}".format(dest_table))
    
    
if __name__ == '__main__':
    print('')
    print(NEW_QUERY)
    # get_query_result(VIEW_QUERY, 1)
    # create_table_from_query(VIEW_QUERY, NEW_BIGQUERY_TABLE)
    