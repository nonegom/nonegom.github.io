---
title:  "[머신러닝] 지도학습 - 6.3. 특징선택"
excerpt: ""

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
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/14.03%20%ED%8A%B9%EC%A7%95%20%EC%84%A0%ED%83%9D.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  


## 0. 특징선택
실무에서는 대규모의 데이터를 기반으로 분류예측 모형을 만들어야 하는 경우가 많다. 대규모의 데이터라고 하면 **표본의 갯수가 많거나** 아니면 독립변수 즉, **특징데이터의 종류가 많거나** 혹은 이 두가지 모두인 경우가 있다.
여기에서는 특징데이터의 종류가 많은 경우에 가장 **중요하다고 생각되는 특징데이터만 선택**하여 특징 데이터의 종류를 줄이기 위한 방법(**특징선택**)에 대해 알아본다.

### 샘플 데이터 로드
```py
# fetch_rcv1, 로이터 뉴스 말뭉치 사용 (텍스트 데이터 사용)
from sklearn.datasets import fetch_rcv1

rcv_train = fetch_rcv1(subset="train")
rcv_test = fetch_rcv1(subset="test")
X_train = rcv_train.data
y_train = rcv_train.target
X_test = rcv_test.data
y_test = rcv_test.target

# Ont-Hot-Encoding된 라벨을 정수형으로 복원
classes = np.arange(rcv_train.target.shape[1])
y_train = y_train.dot(classes)
y_test = y_test.dot(classes)

print(X_train.shape)
# > (23149, 47236)
```

## 1. 분산에 의한 선택

원래 예측모형에서 **중요한 특징데이터**란 **종속데이터와의 상관관계가 크고 예측에 도움이 되는 데이터**를 말한다. 하지만 상관관계 계산에 앞서 **특징데이터의 값 자체가** 표본에 따라 **그다지 변하지 않는다면** 종속데이터 예측에도 도움이 되지 않을 가능성이 높다. 따라서 표본 변화에 따른 데이터 값의 변화 즉, **분산**이 **기준치보다 낮은 특징 데이터는 사용하지 않는 방법**이 분산에 의한 선택 방법이다. 예를 들어 종속데이터와 특징데이터가 모두 0 또는 1 두가지 값만 가지는데 종속데이터는 0과 1이 균형을 이루는데 반해 대부분(예를 들어 90%)의 특징데이터 값
이 0이라면 이 **특징데이터는 분류에 도움이 되지 않을 가능성이 높다.**
하지만 '분산'에 의한 선택은 반드시 상관관계와 일치한다는 보장이 없기 때문에 신중하게 사용해야 한다.
- 따라서 y값에 대해 **x값이 아예 안움직이거나, 변하지 않는 데이터들**을 뺀다는 것이다. 이 값은 '분산'(standard diviaton)으로 알아낼 수 있기 때문에 결국에는 '분산'값이 너무 작은 애들은 빼겠다는 방법이다.

> 이 방법은 계산량이 얼마 되지 않지만 효율이 좋다.

```py
from sklearn.feature_selection import VarianceThreshold

selector = VarianceThreshold(1e-5)
X_train_sel = selector.fit_transform(X_train)
X_test_sel = selector.transform(X_test)
X_train_sel.shape

# > (23149, 14330)
## 데이터의 종류가 줄었다.
```

### 나이브 베이즈 모형 사용
나이브 베이즈 모형의 경우 '텍스트 데이터 분류'에 사용된다. 텍스트 데이터 중 'stopwords'를 걸러냄으로 performance가 오히려 올라가는 경우가 생긴다.

```py
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

model.fit(X_train, y_train)
print("train accuracy:{:5.3f}".format(accuracy_score(y_train, model.predict(X_train))))
print("test accuracy :{:5.3f}".format(accuracy_score(y_test, model.predict(X_test))))
""" >
train accuracy:0.381
test accuracy :0.324
"""

# X_train_sel 데이터는 위 코드 참고
model = BernoulliNB()
model.fit(X_train_sel, y_train)
print("train accuracy:{:5.3f}".format(accuracy_score(y_train, model.predict(X_train_sel))))
print("test accuracy :{:5.3f}".format(accuracy_score(y_test, model.predict(X_test_sel))))
""" >
특징 데이터만을 선택해서 모댈링을 하면, 성능이 더 증가함을 확인할 수 있다.

train accuracy:0.529
test accuracy :0.441
"""
```

## 2. 단일 변수 선택
각각의 독립변수를 하나만 사용한 예측모형의 성능을 이용해서 가장 **분류성능 혹은 상관관계**가 높은 변수만 선택하는 방법이다. 즉, 개별적으로 성능이 좋은 $X$를 뽑아내서 변수를 선택하는 것이다. 하지만 단일 변수의 성능이 높은 특징만 모았다고 반드시 전체 성능이 향상된다는 보장은 없다.

`feature_selection`서브패키지는 아래와 같은 성능지표를 제공한다.
  - `chi2` : 카이제곱 검정 통계값
  - `f_classif` : 분산분석(ANOVA) F검정 통계값
  - `mutual_info_classif` : 상호정보량(mutual information)

`feature_selection` 서브패키지는 성능이 좋은 변수만 사용하는 전처리기인 `SelectBest`클래스도 제공한다.
```py
from sklearn.feature_selection import chi2, SelectKBest

# 기준은 Chi^2 값 사용 (데이터 가공)
selector = SelectKBest(chi2, k=14330)
X_train1 = selector.fit_transform(X_train, y_train)
X_test1 = selector.transform(X_test)

# 모델 생성
model = BernoulliNB()
model.fit(X_train1, y_train)
print("train accuracy:{:5.3f}".format(accuracy_score(y_train, model.predict(X_train1))))
print("test accuracy :{:5.3f}".format(accuracy_score(y_test, model.predict(X_test1))))

# > train accuracy:0.505
#   test accuracy :0.438
```

## 3. 다른 모형을 이용한 특성 중요도 계산
특성 중요도(feature importance)를 계산할 수 있는 랜덤포레스트 등의 **다른 모형을 사용**하여 일단 특성만 선택하고 최종 분류는 다른 모형을 사용하는 방법이다.

```py
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier

# 별도의 모델을 생성해서 Selector를 정한다.
n_sample = 10000
idx = np.random.choice(range(len(y_train)), n_sample)
model_sel = ExtraTreesClassifier(n_estimators=50).fit(X_train[idx, :], y_train[idx])
selector = SelectFromModel(model_sel, prefit=True, max_features=14330)
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

# 예측 모델은 기존처럼 베르누이 나이브 베이즈 모형을 이용한다.
model = BernoulliNB()
model.fit(X_train_sel, y_train)
print("train accuracy:{:5.3f}".format(accuracy_score(y_train, model.predict(X_train_sel))))
print("test accuracy :{:5.3f}".format(accuracy_score(y_test, model.predict(X_test_sel))))

"""
역시나 퍼포먼스가 좋아짐을 확인할 수 있다.
 > train accuracy:0.604
   test accuracy :0.491
"""
```