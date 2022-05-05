import streamlit as st
import streamlit.components.v1 as components
from collections import OrderedDict
import os
import pickle
import json
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import base64
import numpy as np
from google.protobuf import text_format



# 모델 API endpoint
url = 'http://<ip>:8502'  
predict_endpoint = '/v1/models/advert_classifier:predict'

# 기타 변수 초기화
last_updated = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def get_user_input_features():
    """
    메뉴 및 기타 input 값을 받기 위한 함수
    :return: json 형식의 user input 데이터
    """

    user_features = {"menu_name": st.sidebar.selectbox('Menu', ['predict'])}

    return [user_features]


def get_raw_input_features():
    """
    raw data input 값을 받기 위한 함수
    :return: json 형식의 raw input 데이터
    """
    raw_features = {"AdTopicLine": st.sidebar.text_input('AdTopicLine', 'Proactive bandwidth-monitored policy'),
                    "Age": st.sidebar.slider('Age', 0, 90, 17),
                    "AreaIncome": st.sidebar.slider('AreaIncome', 0.0, 200000.0, 41920.7890625),
                    "City": st.sidebar.text_input('City', 'West Steven'),
                    "Country": st.sidebar.text_input('Country', 'Guatemala'),
                    "DailyInternetUsage": st.sidebar.slider('DailyInternetUsage', 0.0, 1440.0, 187.9499969482422),
                    "DailyTimeSpentOnSite": st.sidebar.slider('DailyTimeSpentOnSite', 0.0, 1440.0, 55.54999923706055),
                    "Male": st.sidebar.selectbox('Male', (1, 0)),
                    }

    return [raw_features]


def create_sample_data_from_json(raw_features):
    raw_json = raw_features[0]
    
    ad_topic_line = raw_json['AdTopicLine']
    age = raw_json['Age']
    area_income = raw_json['AreaIncome']
    city = raw_json['City']
    country = raw_json['Country']
    daily_internet_usage = raw_json['DailyInternetUsage']
    daily_time_spent_on_site = raw_json['DailyTimeSpentOnSite']
    male = raw_json['Male']
    
    example_text = '''
        features {
          feature {
            key: "AdTopicLine"
            value {
              bytes_list {
                value: "%s"
              }
            }
          }
          feature {
            key: "Age"
            value {
              int64_list {
                value: %d
              }
            }
          }
          feature {
            key: "AreaIncome"
            value {
              float_list {
                value: %f
              }
            }
          }
          feature {
            key: "City"
            value {
              bytes_list {
                value: "%s"
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
                value: "%s"
              }
            }
          }
          feature {
            key: "DailyInternetUsage"
            value {
              float_list {
                value: %f
              }
            }
          }
          feature {
            key: "DailyTimeSpentOnSite"
            value {
              float_list {
                value: %f
              }
            }
          }
          feature {
            key: "Male"
            value {
              int64_list {
                value: %d
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
    ''' % (ad_topic_line, age, area_income, city, country, daily_internet_usage, daily_time_spent_on_site, male)  

    example = text_format.Parse(example_text, tf.train.Example())
    byte_example = example.SerializeToString()
    return byte_example


def create_json_request_data(byte_example, signature):
    encoded_string = base64.b64encode(byte_example).decode('utf-8')
    # signature_name에 따라 다른 output
    request_body = json.dumps({"signature_name": signature, "instances": [{"b64": encoded_string}]})
    
    return request_body


def post_tfserving_request(url, data):
    headers = {"content-type": "application/json"}
    json_response = requests.post(url, data=data, headers=headers)
    print(json_response)
    predictions = np.array(json.loads(json_response.text)["predictions"])
    return predictions


def streamlit_main():
    """
    streamlit main 함수
    :return: None
    """
    st.title('Advertising Classifier')
    # 화면 오른쪽에 last updated 표시
    components.html(
        f'''<p style="text-align:right; font-family:'IBM Plex Sans', sans-serif; font-size:0.8rem; color:#585858";>\
            Last Updated: {last_updated}</p>''', height=30)

    # sidebar input 값 선택 UI 생성
    st.sidebar.header('User Menu')
    user_input_data = get_user_input_features()

    st.sidebar.header('Raw Input Features')
    raw_input_data = get_raw_input_features()

    submit = st.sidebar.button('Get predictions')
    if submit:
        sample_data = create_sample_data_from_json(raw_input_data)
        json_body = create_json_request_data(sample_data, signature='serving_default')
        result = post_tfserving_request(url + predict_endpoint, json_body)

        # 예측 결과 표시
        st.subheader('Results')
        prob = np.around(result[0, 0], 2)
        st.write("Prediction: ", prob)


if __name__ == '__main__':
    streamlit_main()