import kfp


# 참조
# https://kubeflow-pipelines.readthedocs.io/en/stable/source/kfp.client.html
# 유용한 함수
# create
# client.create_experiment(name)  -> experiment 생성 
# listing
# client.list_experiments() -> id는 name이 아니고 여기서 확인된 id
# client.list_pipelines() -> id는 name이 아니고 여기서 확인된 id
# client.list_pipeline_versions() -> version 확인
# get
# client.get_pipeline(id) -> 특정 파이프라인의 상세 정보
# recurring_runs
# client.list_recurring_runs() -> recurring_run 목록 확인
# disable_job(id) -> id 입력해 recurring_run disable 가능
# delete_job(id)  -> recurring_run 삭제


def connect_to_client(host):
    client = kfp.Client(host)  # url
    return client


def create_custom_recurring_runs(host, 
                                 experiment_id, 
                                 job_name, 
                                 description, 
                                 start_time, 
                                 end_time, 
                                 cron_expression, 
                                 pipeline_id,
                                 version_id, 
                                 no_catchup):
    client = kfp.Client(host)
    client.create_recurring_run(experiment_id,  #  31qaq32-... 과 같은 id 형식
                                job_name,  # recurring_run 이름. recurring_run은 job에 해당
                                description,
                                start_time,  # 2022-05-02T09:00:00Z
                                end_time, # '2022-05-06T09:00:00Z'
                                cron_expression, # '0 0 9 * * *' 매일 아침 9시
                                pipeline_id,
                                version_id,  # version_id가 pipeline_id보다 우선권 가짐
                                no_catchup)