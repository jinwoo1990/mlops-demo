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
