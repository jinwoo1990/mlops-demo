# Float feature 목록
DENSE_FLOAT_FEATURE_KEYS = ['DailyTimeSpentOnSite', 'Age', 'AreaIncome', 'DailyInternetUsage' ]

# Categorical feature 목록
VOCAB_FEATURE_KEYS = ['City', 'Male', 'Country' ]
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