---
title:  "[머신러닝] 지도학습 - 3.4. 감성분석"
excerpt: "나이브베이즈 분류 모형을 이용한 문서 감성분석"

categories:
  - MachinLearning
tags:
  - SupervisedLearning
  - 10월
toc: true
toc_sticky: true
toc_label: 페이지 목차
use_math: true
---
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/11.03%20%EA%B0%90%EC%84%B1%20%EB%B6%84%EC%84%9D.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 1. 감성분석
 '감성분석(sentiment analysis)'은문서 데이터에 대해 문서가 나타내는 뉘앙스나 분위기가  **좋다** 혹은 **나쁘다**는 평가를 내리는 것을 말한다. 주식에서도 사용되고, 리뷰나 평판에 대한 것을 판단할 때도 사용할 수 있다. 

### 감성분석 예제
github에 올려져 있는 네이버 영화 감상평에 대한 감성 분석 예제를 이용
- https://github.com/e9t/nsmc (https://github.com/e9t/nsmc)
- 자료를 받아서 python 작업 파일과 같은 디렉토리 내의 저장시켜 놓으면 된다.

- 다항 나이브베이즈 모형을 이용하되, 다양한 전처리 방법을 사용해보도록 하겠다.

### 1) 데이터 로드 및 정보
```py
# codecs 패키지를 활용, 유니코드로 인코딩 / 이를 Streaming Ecoder라고 한다. 
import codecs
with codecs.open("ratings_train.txt", encoding='utf-8') as f:
  data = [line.split('\t') for line in f.read().splitlines()]
  data = data[1:] # header 제외

# 데이터는 번호, 내용, 평점으로 이루어져 있다.
from pprint import pprint  # pprint: 어떤 구조를 맞춰 프린트 한다.
pprint(data[0])
"""
 # |번호      |내용                     | 좋다(1), 나쁘다(0)    
 >['9976970', '아 더빙.. 진짜 짜증나네요 목소리', '0']
"""
```
### 2) 데이터 분석 (CountVectorizer)
---
```py
# 내용을 X, 평점을 y로 지정한다.
X = list(zip(*data))[1]
y = np.array(list(zip(*data))[2], dtype=int)

# 다항 나이브베이즈 모형 생성 (Pipeline이용 - Countvectorizer)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

model1 = Pipeline([
  ('vect', CountVectorizer(),
  'mb', MultinomialNB())
])

# 모델 fitting 
model1.fit(X, y)

# 테스트 데이터 생성
with codecs.open("ratings_test.txt", encoding='utf-8') as f:
  data_test = [line.split('\t') for line in f.read().splitlines()]
  data_test = data_test[1:] # header 제외

X_test = list(zip(*data_test))[1]
y_test = np.array(list(zip(*data_test))[2], dtype=int)

# 분석 결과 보고서 출력
print(classification_report(y_test, model1.predict(X_test)))
```
![](/assets/images/Supervised5_0.jpg)

### 3) 데이터 분석 (TfidfVectorizer)
---
```py
from sklearn.feature_extraction.text import TfidfVectorizer
model2 = Pipeline([
('vect', TfidfVectorizer()),
('mb', MultinomialNB()),
])

# 모델 피팅
model2.fit(X, y)

# 분석 결과 보고서 출력
print(classification_report(y_test, model2.predict(X_test)))
```
![](/assets/images/Supervised5_1.jpg)

### 4) 데이터 분석 (형태소 분석기 이용)
---
```py
from konlpy.tag import Okt
pos_tagger = Okt()
def tokenize_pos(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc)]

model3 = Pipeline([
('vect', CountVectorizer(tokenizer=tokenize_pos)),
('mb', MultinomialNB()),
])

# 모델 피팅
model3.fit(X, y)

# 분석 결과 보고서 출력
print(classification_report(y_test, model3.predict(X_test)))
```
![](/assets/images/Supervised5_2.jpg)

### 5) 데이터 분석 (N-gram)
---
```py
from konlpy.tag import Okt
pos_tagger = Okt()
def tokenize_pos(doc):
    return ['/'.join(t) for t in pos_tagger.pos(doc)]

model4 = Pipeline([
('vect', TfidfVectorizer(tokenizer=tokenize_pos, ngram_range=(1, 2))),
('mb', MultinomialNB()),
])

# 모델 피팅
model4.fit(X, y)

# 분석 결과 보고서 출력
print(classification_report(y_test, model4.predict(X_test)))
```
![](/assets/images/Supervised5_3.jpg)

- N-gram을 사용하면 성능이 더 개선되는 것을 볼 수 있다.

## 2. 정리 (f1-score)
1) 데이터 분석(CountVectorizer)   - 0.83  
2) 데이터 분석(TfidfVectorizer)   - 0.83  
3) 데이터 분석(형태소 분석기 이용) - 0.85  
4) 데이터 분석(N-gram)            - 0.87  

- 어느 전처리기를 사용하느냐에 따라 성능에 차이가 난다.