# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List


# Float feature 목록
DENSE_FLOAT_FEATURE_KEYS = ['DailyTimeSpentOnSite', 'Age', 'AreaIncome', 'DailyInternetUsage' ]

# Categorical feature 목록
VOCAB_FEATURE_KEYS = ['City', 'Male', 'Country']
# Number of vocabulary terms used for encoding VOCAB_FEATURES by tf.transform
# Top 1000개의 vocab 각각에 대해 값 생김
VOCAB_SIZE = 1000
# Count of out-of-vocab buckets in which unrecognized VOCAB_FEATURES are hashed.
# Top 1000개 외 vocab을 10개의 임의 공간에 할당
OOV_SIZE = 10

# Target 목록
LABEL_KEY = 'ClickedOnAd'


# Transformed된 feature 이름 재설정 위한 함수
def transformed_name(key):
    return key + '_xf'


def transformed_names(keys: List[str]) -> List[str]:
    return [transformed_name(key) for key in keys]
