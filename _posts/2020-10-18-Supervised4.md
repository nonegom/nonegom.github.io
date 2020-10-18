---
title:  "[머신러닝] 지도학습 - 3.3. 나이브베이즈 분류모형 연습문제"
excerpt: "나이브베이즈 분류 모형을 사용한 데이터 분석 연습문제(load_iris, load_digits, fetch_cotype)"

categories:
  - MachinLearning
tags:
  - SupervisedLearning
  - 10월
  - 예제문제
toc: true
toc_sticky: true
toc_label: 페이지 목차
use_math: true
---
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/11.02%20%EB%82%98%EC%9D%B4%EB%B8%8C%EB%B2%A0%EC%9D%B4%EC%A6%88%20%EB%B6%84%EB%A5%98%EB%AA%A8%ED%98%95.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 1. 붓꽃 분류문제
붓꽃 분류문제를 **가우시안 나이브베이즈 모형**을 사용해 풀기
1. 각각의 종이 선택될 사전확률을 구하라.
2. 각각의 종에 대해 꽃받침의 길이, 꽃받침의 폭, 꽃잎의 길이, 꽃잎의 폭의 평균과 분산을 구하라.
3. 학습용 데이터를 사용하여 분류문제를 풀고 '분류결과표'와 '분류보고서'를 계산하라.

```py
# 0. 데이터 로드
from sklearn.datasets import load_iris
iris = load_iris()
X = load_iris.data
y = load_iris.target

# 1. 모델 생성 -> 사전확률 계산
from sklearn.naive_bayes import GaussianNB
model = GaussianNB().fit(X, y)
model.class_prior_

# 2. 각 독립변수에 대한 평균과 분산값
## 평균값
model.theta_
"""> array(
 index  |x1     |x2    |x3   |x4
class 0 [[5.006, 3.428, 1.462, 0.246],
clsss 1 [5.936, 2.77 , 4.26 , 1.326],
class 2 [6.588, 2.974, 5.552, 2.026]]
)"""
## 분산값
model.theta_

# 3. 분류결과표와 분류보고서
y_pred = model.predict(X)

from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)

from sklearn.metrics import classification_report
print(classification_report(y1, y1_pred))
```

## 2. MNIST 숫자분류 문제
1. 숫자 분류문제에서 sklearn.preprocessing.Binarizer 로 x값을 0, 1로 바꾼다(값이 8 이상이면 1, 8미만이면 0). 즉 흰색과 검은색 픽셀로만 구성된 이미지로 만든다. 그리고 **베르누이 나이브베이즈 모형**적용 후 분류 결과를 분류 보고서 형식으로 나타내라.
2. `BernoulliNB` 클래스의 `binarize` 인수를 사용하여 같은 문제를 풀어본다.
3. 계산된 모형의 모수 벡터 값을 각 클래스별로 8x8 이미지의 형태로 나타내라.

```py
## 0. 데이터 로드
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target

## 1. 데이터 이진화
from sklearn.preprocessing import Binarizer
X_bin = Bainarizer(7).fit_transform(X)

## 1-2. 모형 생성 후 보고서 출력
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB().fit(X, y)

y_pred = model.predict(X)

from sklearn.metrics import classification_report
print(classification_report(y, y_pred))

## 2. binarize 인수를 사용해 fit
model2 = BernoulliNB(binarize = 7).fit(X, y)
y_pred2 = model2.predict(y, y_pred)
print(classification_report(y, y_pred2))
### 앞선 리포트와 결과의 차이가 없다.

## 3. 모수 값 구하기
theta = np.exp(model.feature_log_prob_)
reshape_theta0 = np.reshape(theta[0], (8, 8)) # 0번째 데이터만

## 3-2. 시각화 모형 그리기
plt.imshow(reshape_theta0, cmap=plt.cm.binary)
plt.grid(False)
plt.show()

### 해당 모수 값들은 다른 입력 데이터를 확인해, 숫자를 분류하는 틀이 된다.  

```

## 3. MNIST 숫자 분류문제 2
1. **다항분포 나이브베이즈 모형**을 사용해서 MNIST 숫자 분류문제를 풀고, 이진화(Binarizing)를 하여 **베르누이 나이브베이즈 모형**을 적용했을 경우와 성능을 비교

```py
## 데이터 로드
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target

## 모델 생성
from sklearn.naive_bayes import MultinomialNB
model_mult = MultinomialNB().fit(X, y)
y_pred = model_mult.predict(X

## 분류보고서 출력
from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
### 베르누이분포모형과 다항분포 나이브베이즈 모형 차이 없음.
```

## 4. TF-IDF 인코딩
TF-IDF 인코딩을 하면 단어의 빈도수가 정수가 아닌 실수값이 된다. 실수 값에도 다항분포 모형을 적용할 수 있을까? 
- 상관없이 구동이 된다. 왜냐하면 값이 나온 횟수를 counting할 때 `sum`을 하기 때문이다. 따라서 실수여도 결국은 계산이 된다.
- 베르누이분포로도 계사닝 가능하다.
> 실수든, 0이나 1이든, 정수든 무엇이 나와도 어떤 모형에서든지 보고서가 비슷하게 나온다.

```py
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
y = digits.target

X_float = X / 10

from sklearn.naive_bayes import MultinomialNB
model_mult = MultinomialNB().fit(X_float, y)

y_pred = model_mult.predict(X)

from sklearn.metrics import classification_report
print(classification_report(y, y_pred))
```

## 5. covtype 분류문제
사이킷런에서 제공하는 예제 중 숲의 수종을 예측하는 covtype 분류문제는 **연속확률분포 특징**과 **베르누이확률분포 특징**이 섞여있다. 이 문제를 사이킷런에서 제공하는 나이브베이즈 클래스를 사용하여 풀어라.

### 풀이 방법
- 데이터를 분할해서 각각 모형을 적용시킨다. GaussianNB를 사용하는 부분과 BernoulliNB를 사용하는 부분으로 나눈다. 
- 가능도를 구하고, 이를 다시 곱해 구하면 된다. 

> 사실 그냥 한 모형으로 구해도 accuracy 차이가 많이 안 난다.

```py
### 0. 데이터 생성
from sklearn.datasets import fetch_covtype
covtype = fetch_covtype()
X = covtype.data
y = covtype.target
X1 = X[:10] # 연속확률분포 특징
X2 = X[10:] # 베르누이확률분포 특징

## 1. 모형 생성
from sklearn.naive_bayes
mode1 = GaussianNB().fit(x1, y)
mode2 = GaussianNB().fit(x2, y)

## 2. 모수 계산
# 모수값 계산
prob1 = model1.predict_proba(X1)
prob1 = model2.predict_proba(X1)

# 가능도 계산 "모수 값 / 사전확률"
likeli1 = prob1 / model1.class_prior_
likeli2 = prob2 / model1.calss_prior_ 
"""
model1과 model2의 사전확률은 같다. 그렇지만 model2의 경우
np.exp(model2.class_log_prior_)로 사전확률을 구해야 한다.
"""

## model1과 model2의 가능도를 통해 '모수 값' 구하기
prob = likeli1 * likeli2 * model1.class_prior_
y_pred = np.argmax(prob, axis=1) + 1

# "+1"을 해주는 이유는 수종 데이터는 0부터 시작하는 값을 가지지만, 클래스는 1부터 시작하기 때문이다.

## 3. 분류 보고서 
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y, y_pred)
print(classification_report(y, y_pred)
# >  accuracy : 0.65

## 참고. 베르누이 부분과 가우시안 부분 나누지 않고 구하기
model = BernoulliNB().fit(X, y)
y_pred = model.predict(X)
print(classification_report(y, y_pred))
# > accuracy : 0.63

```

- 앞서서 covtype데이터를 두 부분으로 나눠서 구하는 방법을 사용했는데, 두 부분으로 나누지 않고 그냥 `BernoullliNB`를 사용해 accuracy를 구하게 되면 실질적으로 accuracy에서 '0.02' 정도 차이가 난다.