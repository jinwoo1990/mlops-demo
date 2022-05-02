import os
import tensorflow as tf
import base64
import json
import numpy as np
import requests
from google.protobuf import text_format


# saved_model serving_default 설정 확인
# !saved_model_cli show --dir {MODEL_DIR} --tag_set serve --signature_def serving_default
# saved_model 모든 구성요소 확인
# !saved_model_cli show --dir {MODEL_DIR} --all

# Google Compute Engine에 docker 설치
# 리눅스 업데이트
# $sudo apt-get update
# $sudo apt-get upgrade$sudo usermod -a -G docker ${USER}udo usermod -a -G docker ${USER}
#Docker 설치
# $sudo curl -fsSL https://get.docker.com/ | sudo sh
#sudo 제외하고 docker 실행 가능 하도록 변경
# $sudo usermod -a -G docker ${USER}

# docker를 활용한 tfserving server 띄우기
# docker pull tensorflow/serving:2.5.1 커맨드를 통해 설치 (2.5.1이 현 시점 latest)
# MODEL_PATH를 원하는 값으로 변경
# serving_model 경로 안에 advert_classifier이라는 경로에 모델에 있어야 작동 (MODEL_NAME 설정 안 하면 기본 값 model인데 디렉토리 없으면 작동 안 함
example_tfservering_init_command = '''
docker run -p 8501:8501 -e MODEL_BASE_PATH=gs://project-111111-serving/serving_model -e MODEL_NAME=advert_classifier -t tensorflow/serving
'''

# command 치는 경로에 test.json 파일 필요
# ip 주소는 cloud compute와 동일 네트워크에서 작업 (노트북)이면 내부/외부 ip 주소 모두 가능하나 그 외에서 접근하려면 외부 주소 활용
# 8501이든 열어놓은 포트에 대해 방화벽 규칙 추가 필요
example_post_command = '''
curl -X POST http://<ip>:8501/v1/models/advert_classifier:predict \
    -d @./advert_test.json \
    -H "Content-Type: application/json"
'''

example_text = '''
features {
  feature {
    key: "AdTopicLine"
    value {
      bytes_list {
        value: "Proactive bandwidth-monitored policy"
      }
    }
  }
  feature {
    key: "Age"
    value {
      int64_list {
        value: 90
      }
    }
  }
  feature {
    key: "AreaIncome"
    value {
      float_list {
        value: 41920.7890625
      }
    }
  }
  feature {
    key: "City"
    value {
      bytes_list {
        value: "West Steven"
      }
    }
  }
  feature {
    key: "ClickedOnAd"
    value {
      int64_list {
        value: 0
      }
    }
  }
  feature {
    key: "Country"
    value {
      bytes_list {
        value: "Guatemala"
      }
    }
  }
  feature {
    key: "DailyInternetUsage"
    value {
      float_list {
        value: 187.9499969482422
      }
    }
  }
  feature {
    key: "DailyTimeSpentOnSite"
    value {
      float_list {
        value: 55.54999923706055
      }
    }
  }
  feature {
    key: "Male"
    value {
      int64_list {
        value: 0
      }
    }
  }
  feature {
    key: "Timestamp"
    value {
      bytes_list {
        value: "2022-04-29T02:35:54Z"
      }
    }
  }
}
'''


def load_saved_model(model_dir):
    """
    loaded.variables  # variables 확인
    """
    loaded = tf.saved_model.load(model_dir)
    print(list(loaded.signatures.keys()))  # ["serving_default", "..."]
    
    return loaded


def create_sample_data(example_text):
    example = text_format.Parse(example_text, tf.train.Example())
    byte_example = example.SerializeToString()
    return byte_example


def create_json_request_data(byte_example, signature):
    encoded_string = base64.b64encode(byte_example).decode('utf-8')
    # signature_name에 따라 다른 output
    request_body = json.dumps({"signature_name": signature, "instances": [{"b64": encoded_string}]})
    
    return request_body


def infer_using_signature(loaded, signature, examples):
    """
    examples: example_pb2를 SerializedToString()으로 바꾼 byte 데이터여야 함
    """
    infer = loaded.signatures[signature]
    res = infer(examples=tf.constant(examples))
    return res


def post_tfserving_request(url, data):
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    print(json_response)
    predictions = np.array(json.loads(json_response.text)["predictions"])
    return predictions


def test_infer_with_examplegen(raw_dataset):
    """
    raw_dataset = tf.data.TFRecordDataset(list_bucket_uri_dir(example_uri + '/Split-train'), compression_type="GZIP")
    추가 참조
    example = tf.train.Example()
    example.ParseFromString(serialized_example)  # example pb 형식으로 된 스트링 확인가능
    serialized_example로 signature concrete function으로 input 들어감 (과정은 내부 소스 참조)
    """
    for tfrecord in raw_dataset.take(3):
        serialized_example = tfrecord.numpy()
        res = infer_using_signature(loaded, 'serving_default', serialized_example)
        print(res)
