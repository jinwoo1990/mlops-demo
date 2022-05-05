## Docker

- 이미지 MODEL_BASE_PATH 자기 자신 경로에 맞게 수정
- 이미지 만들고 gcr에 저장
    - TFserving용 Dockerfile이 있는 ../mlops-demo/advert_pipeline/deployment 경로로 cd
    - `docker build -t gcr.io/[GOOGLE_PROJECT_ID]/advert-tfserving:latest .`
    - `docker push gcr.io/[GOOGLE_PROJECT_ID]/advert-tfserving:latest`
    - 생성된 이미지는 Container Registry에서 확인 가능하고 파일은 같은 프로젝트에 존재하는 stroage에 저장됨
    - 두번째 push부터는 기존 latest 태그를 없애고 새로 push된 이미지를 latest로 관리
- 가져오기
    - `docker pull gcr.io/[GOOGLE_PROJECT_ID]/advert-tfserving:latest`
- VM 인스턴스 생성
    - Container-Optimized OS 인스턴스로 gcr에 있는 이미지를 불러와서 인스턴스 생성
    - 해당 instance의 ip로 tfserving request하면 정상 작동
    - `gcloud compute instances create-with-container advert-tfserving-container --container-image gcr.io/[GOOGLE_PROJECT_ID]/advert-tfserving:latest --zone us-central1-a`


## GKE

- 환경변수 설정
    - `gcloud config set compute/zone us-central1-a`
    - `PROJECT_ID=$(gcloud config get-value project)`
    - `CLUSTER_NAME=cluster-2`
- 클러스터 생성 (1.22.8-gke.200)
    - `gcloud beta container clusters create $CLUSTER_NAME \
      --cluster-version=latest \
      --machine-type=n1-standard-4 \
      --enable-autoscaling \
      --min-nodes=1 \
      --max-nodes=3 \
      --num-nodes=1`
- Credentials 설정
    - `gcloud container clusters get-credentials $CLUSTER_NAME`
- k8s 설정
    - ../mlops-demo/advert_pipeline/deployment 경로로 cd
    - configmap.yaml MODEL_PATH 수정
    - `kubectl apply -f ./configmap.yaml`
    - `kubectl apply -f ./deployment.yaml`
    - `kubectl apply -f ./service.yaml`
    - `kubectl autoscale deployment advert-classifier --cpu-percent=60 --min=1 --max=4`
- 활용
    - `kubectl get svc advert-classifier`로 LoadBalancer external ip 확인
    - ../mlops-demo 경로로 cd (advert_test.json 파일 있는 경로)
    - `curl -X POST http://<ip>:8501/v1/models/advert_classifier:predict \
    -d @./advert_test.json \
    -H "Content-Type: application/json"`
