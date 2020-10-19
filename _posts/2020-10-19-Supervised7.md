---
title:  "[머신러닝] 지도학습 - 4.2. 모형결합"
excerpt: "Ensemble methos라고도 불리는 '모형결합' 중 취합 방법론 "

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
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/12.02%20%EB%AA%A8%ED%98%95%20%EA%B2%B0%ED%95%A9.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 0. 모형결합
회귀분석 모델은 '주관식 문제'라고 볼 수 있는 대신, **분류 모델**은 '객관식 문제'라고 할 수 있기에 정답이 쏠리는 부분이 나타날 수 있다. (예를들어 '4지선다 문제'의 경우 특정한 답에 쏠릴 수 있다.) 그렇기에 분류모델에서는 여러 모델을 **'동시에'** 활용해 이런 문제를 해결한다. 이를 **모형 결합(model combining)**이라고 한다. 
다시 말해, 모형 결합(model combining)방법은 앙상블 방법론(ensemble methods)라고도 한다. 이는 특정한 하나의 예측 방법을 사용하는 것이 아니라, 복수의 예측 모형을 결합해 더 나은 성능 예측을 하려는 시도이다. 모형 결합을 하면 다음과 같은 이점이 있다.

- 단일 모형을 사용할 때보다 **성능 분산이 감소**하고, **과최적화를 방지**한다.
- '개별 모형'이 성능이 안좋아도 여러개를 모은 **'결합 모형'의 성능은 더 향상**된다.

## 1. 모형 결합 방법론
모형 결합 방법은 크게 '취합 방법론(agrregation)'과 부스팅 방법론(boosting)방법으로 나눌 수 있다.

### 취합 방법론 (agrregation)
사용할 모형의 집합이 이미 결정되어 있어서, 모든 모형이 다 같은 문제를 푼다.
   - 다수결 (Majority Voting)
   - 배깅 (Bagging)
   - 랜덤포레스트 (Random Forests)

### 부스팅 방법론 (Boosting)
 사용할 모형을 점진적으로 늘려간다. 그렇게 함으로 새로 추가된 모형은 이전 모형들이 잘 풀지 못하는 문제를 풀게 된다. 
   - 에이다부스트 (AdaBoost)
   - 그레디언트 부스트 (Gradient Boost)

> 아래에서는 '취합 방법론'에 대해 먼저 알아보겠다.

## 2. 다수결 방법
가장 단순한 모형 결합 방법으로 전혀 다른 모형도 결합할 수 있다. 다수결 방법은 두 가지로 나눠진다.
  - `hard voting`: 단순 투표. 개별 모형의 결과 기준 (각 모델이 어떤 클래스를 뽑았나를 취합)
  - `soft voting`: 가중치 투표. 개별 모형의 조건부 확률의 합 기준 (각 모델이 어떤 클래스의 비중이 높은가를 취합)

`votingClassfier(estimate, voting, weights)`클래스: 가중치는 voting값에 가중치를 곱하게 된다.
- `estimate`: 개별 모형 목록, 리스트나 named parameter 형식으로 입력
- `voting` : 문자열 {hard, soft} hard voting과 soft voting 선택. 디폴트는 hard
- `weights` : 사용자 가중치 리스트

### 예제 코드

우선, 다음과 같은 데이터가 있다고 가정하자.  

```py
X = np.array([[0, -0.5], [-1.5, -1.5], [1, 0.5], [-3.5, -2.5], [0, 1], [1, 1.5], [-2, -0.5]])
y = np.array([1, 1, 1, 2, 2, 2, 2])
x_sample = [0, -1.5]
plt.scatter(X[y == 1, 0], X[y == 1, 1], s=100, marker='o', c='r', label="클래스 1")
plt.scatter(X[y == 2, 0], X[y == 2, 1], s=100, marker='x', c='b', label="클래스 2")
plt.scatter(x_sample[0], x_sample[1], s=100, marker='^', c='g', label="테스트 데이터")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("이진 분류 예제 데이터")
plt.legend()
plt.show(
```
![](/assets/images/Supervised7_1.png)

여기서 **테스트 데이터의 클래스를 예측**하는 모델들을 만들어보고, 각 모델들이 어떤 클래스로 예측하는지 확인해보자.
```py
# 세 가지 다른 방법으로 풀어본다.
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier

model1 = LogisticRegression(random_state=1) # 로지스틱 회귀 모형
model2 = QuadraticDiscriminantAnalysis()    # QDA 모형
model3 = GaussianNB()                       # 가우시안 나이브베이즈 모형
ensemble = VotingClassifier(                # 결합 모형(다수결 방법)
    estimators=[('lr', model1), ('qda', model2), ('gnb', model3)], voting='soft')

probas = [c.fit(X, y).predict_proba([x_sample])
              for c in (model1, model2, model3, ensemble)]
class1_1 = [pr[0, 0] for pr in probas]
class2_1 = [pr[0, 1] for pr in probas]

########## 그래프 그리는 코드는 이하 생략 (링크 참고) ################

```
![](/assets/images/Supervised7_2.png)

그래프를 보면 '로지스틱 회귀 모형'과 '가우시안 나이브베이즈 모형'은 x_sample데이터(0,1.5)의 위치를 클래스 1이라고 예측했고, 'QDA 모형'은 클래스 2라고 예측했다. 그래서 이 모형들을 합한 '소프트 다수결 모형'의 경우 클래스 2에 더 가깝다고 예측했다. 위 예측 결과를 다른 플롯으로 본다면, 아래와 같다.
![](/assets/images/Supervised7_3.jpg)

### 모형결합을 통한 성능향상
- 개별 모형이 정답을 출력할 확률이 $p$인 경우에, 서로 다르고 **독립적인 모형**을 N개 모아서 다수결 모형을 만들면 정답을 출력할 확률은 다음과 같다.
$$\sum^{N}_{k>N/2} {N \choose k} p^k (1-p)^{N-k}$$
- 모형을 모을 때, 정답을 맞춘 모형들($k$)이 전체 모형의 수($N$)보다 많아야 한다.
- $p$는 정답을 맞출 확률, $1-p$는 오답을 맞출 확률이다.

```py
from scipy.special import comb
def total_error(p, N):
    te = 0.0
    for k in range(int(np.ceil(N/2)), N + 1):
        te += comb(N, k) * p**k * (1-p)**(N-k)
    return te

x = np.linspace(0, 1, 100)
plt.plot(x, x, 'g:', lw=3, label="개별 모형")
plt.plot(x, total_error(x, 10), 'b-', label="다수결 모형 (N=10)")
plt.plot(x, total_error(x, 100), 'r-', label="다수결 모형 (N=100)")
plt.xlabel("개별 모형의 성능")
plt.ylabel("다수결 모형의 성능")
plt.legend(loc=0)
plt.show()
```
![](/assets/images/Supervised7_4.png)

해당 커브는 굉장히 중요한 의미를 가지고 있다. 그것은 모형 결합을 함에 있어서 개별 모형의 성능이 어느 정도 높은 애들을 모아야 한다는 것이다. 만약 개별 모형의 성능이 낮다면, 다수결 모형의 성능이 더 안좋아질 수 있다. 또한 모형을 모을 때는 각 모형이 **독립적인 의사결정**을 할 수 있어야 한다. 
> 즉, **성향이 다르면서**, **성능이 좋은** 모형 여러 개를 모아야 한다.

### 연습문제 1 (iris 데이터 이용)
iris 분류 문제를 다수결 방법을 사용하여 푼다. 모형의 종류 및 개수나 다수결 방법은 마음대로 한다. K=5인 교차 검증을 하였을 때 성능의 평균과 표준편차를 구하라.

``` py
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target

## 모형 생성
model1 = LogisticRegression(random_state=1) # 로지스틱 회귀 모형
model2 = QuadraticDiscriminantAnalysis() # QDA 모형
model3 = GaussianNB() # 가우시안 나이브베이즈 모형
model4 = DecisionTreeClassifier(max_depth=3)
ensemble = VotingClassifier(
    estimators=[('lr', model1), ('qda', model2), ('gnb', model3), ('dt', model4)], voting='soft')
model = ensemble.fit(X, y)
y_pred = model.predict(X)

# K-폴드
from sklearn.model_selection import KFold, cross_val_score
cv = KFold(5, shuffle = True, random_state=0) 
cross_val_score(model, X, y, scoring="accuracy", cv=cv).mean() #.mean() 평균

# > 0.96666666
```

## 3. 배깅
모형 결합에서 사용하는 독립적인 모형의 수가 많을 수록 성능 향상이 일어날 가능성이 높다. 하지만 각각 다른 모형을 사용하는 데에는 그 수의 한계가 있다. 따라서 **배깅방법**을 사용해 **같은 확률 모형**을 쓰지만 **서로 다른 결과를 출력하는 다수의 모형***을 만든다.  
 부트스트래핑과 유사하게 트레이닝 데이터를 랜덤하게 선택해서 다수결 모형을 적용한다. 트레이닝 데이터를 선택하는 방법에 따라 다르게 부른다.  
  - Pasting: 같은 데이터 샘플을 중복 사용(replacement)하지 않음
  - Bagging같은 데이터 샘플을 중복 사용(replacement)
  - RandomSubspaces: 데이터가 아니라 다차원 독립 변수 중 일부 차원을 사용
  - Random Patches: 데이터 샘플과 독립 변수 차원 모두 일부만 랜덤하게 사용 
    
    
- **성능 평가** 시 트레이닝용 데이터가 아닌 **다른 데이터**를 사용하는 경우, 이런 데이터를 **OOB(out-of-bag)데이터**라고 한다.

### `BaggingClassifier`클래스 인수
- base_estimator : 기본 모형
- n_estimators : 모형 갯수. 디폴트 10
- bootstrap : 데이터의 중복 사용 여부. 디폴트 True
- max_samples : 데이터 샘플 중 선택할 샘플의 수 혹은 비율. 디폴트 1.0
- bootstrap_features : 특징 차원의 중복 사용 여부. 디폴트 False
- max_features : 다차원 독립 변수 중 선택할 차원의 수 혹은 비율 1.0

### 예제 코드
- model1은 의사결정모델에서 depth를 길게 함. - overfitting이 발생
- model2는 Bagging을 이용해서, 성능이 안 좋은 개별모형을 여러 개(100개) 사용한다. overfitting이 잘 안 일어나고, 성능 분산이 감소한다. 성능 분산이 감소하는 이유는, overfitting을 발생시키는 소수의 데이터를 포함하는 모형이 적어지기 때문에 다수의 모형에 묻힌다. 따라서 Bagging을 하면 Test퍼포먼스의 성능이 더 좋아진다.

```py
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
iris = load_iris()
X, y = iris.data[:, [0, 2]], iris.target


model1 = DecisionTreeClassifier(max_depth=10, random_state=0).fit(X, y)
model2 = BaggingClassifier(DecisionTreeClassifier(
    max_depth=2), n_estimators=100, random_state=0).fit(X, y)

########## 그래프 그리는 코드 생략 ###########

```
![](/assets/images/Supervised7_5.png)

### 연습문제 2 (breast cancer 데이터 이용)

1. breast cancer 분류 문제를 Bagging을 사용하여 풀어라. 모형의 종류 및 개수나 Bagging 방법은 마음대로 한다. K=5인 교차 검증을 하였을 때 성능의 평균과 표준편차를 구하라.
2. bagging 모형의 성능을 개별 모형과 비교하라.

```py
from sklearn.datasets import load_breast_cancer

bc = load_breast_cancer()
X2 = bc.data
y2 = bc.target

model1 = DecisionTreeClassifier(max_depth=10, random_state=0)
model2 = BaggingClassifier(DecisionTreeClassifier(
    max_depth=10), n_estimators=100, random_state=0)

from sklearn.model_selection import cross_val_score
print("origin:", cross_val_score(model1, X2, y2, scoring="accuracy", cv=5).mean(),'\n'
"bagging:", cross_val_score(model2, X2, y2, scoring="accuracy", cv=5).mean())

# > origin: 0.9173730787144851 
#  bagging: 0.9578636857630801
```
> bagging을 사용한 방법이 그냥 모델을 사용한 것보다 성능이 향상되는 것을 확인할 수 있다.

## 4. 랜덤포레스트
의사결정나무(Decision Tree)를 개별 모형으로 사용하는 모형 결합 방법을 말한다. **Tree의 한 노드노드마다 새로운 종류의 데이터**를 받아서 데이터를 분류하는데 사용할 수 있다. 이렇게 하면 **개별 모형들 사이의 상관관계가 줄어들기 때문에** 모형 성능의 변동이 감소하는 효과가 있다. 이럴 경우 **greedy한 선택으로 인한 가장 좋은 선택의 기회가 박탈되지 않을 가능성**을 열어준다. (의사결정나무의 greedy한 문제 해결방법)
- 쉽게 얘기하면 분류할 기준을 '주사위를 던져'결정한다고 생각할 수 있다. 즉, 제일 성능이 좋을 것 같은 기준으로만 분류를 하는 것이 아니라, 여러가지 기준으로 다양하게 분류를 해볼 수 있다는 뜻이다.


### 예제 코드
iris 데이터를 일반적인 의사결정나무(model1)와 RandomForestClassifier(model2)로 비교해서 문제를 푼다. 
```py
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
iris = load_iris()
X, y = iris.data[:, [0, 2]], iris.target
model1 = DecisionTreeClassifier(max_depth=10, random_state=0).fit(X, y)
model2 = RandomForestClassifier(max_depth=2, n_estimators=100, random_state=0).fit(X, y)

############ 그래프 코드 생략 ############

```
![](/assets/images/Supervised7_6.png)

### 독립변수의 중요도 계산
그리고 랜덤포레스트의 장점 중 하나는 각 '분류기준'에 따라 엔트로피와 관련된 **'정보획득량(information gain)'**을 체크해서 이를 평균내어, 각 **'독립변수의 중요도(feature importance)'** 또한 계산할 수 있다.

```py
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
## 10개의 독립변수를 가지는 임의의 데이터 1000개 생성
X, y = make_classification(n_samples=1000, n_features=10, n_informative=3, n_redundant=0, n_repeate
n_classes=2, random_state=0, shuffle=False)
forest = ExtraTreesClassifier(n_estimators=250, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_

############ 그래프 코드 생략 ############
```
![](/assets/images/Supervised7_7.png)
1,2,0 변수가 중요도가 유독 높게 나오는 것을 확인할 수 있다. 

> **랜덤포레스트**를 사용함에 있어 알아둬야 할 점은 각 노드별로 데이터를 받는다는 게 실제적인 데이터의 이동이 아니라, 데이터를 비교하기 위해 데이터의 정보를 선택하여 이용한다는 느낌으로 생각해야 한다.

이러한 방법을 극단적으로 적용한 것이 Extremely Randomized Trees모형이다. 이 경우 각 노드에서 랜덤하게 독립변수를 선택한다. '랜덤 포레스트'와 'Extremely Randomized Trees'모형은 `RandomForestClassifier 클래스`와 `ExtraTreesClassifier 클래스`로 구현되어 있다.

### 연습문제 3 (breast cancer 데이터 이용)
1. breast cancer 분류 문제를 Extreme 랜덤포레스트를 사용하여 풀어라. K=5인 교차 검증을 하였을 때 평균성능을 구하라.
2. 특징 중요도를 구하라. 어떤 특징들이 판별에 중요하게 사용되는가?

```py
# 0. 데이터 생성
from sklearn.datasets import load_breast_cancer
bc = load_breast_cancer()
X = bc.data
y = bc.target

# 1. 분류 모델 생성 및 K-Fold검증
from sklearn.ensemble import ExtraTreesClassifier
forest = ExtraTreesClassifier(n_estimators=500, random_state=0)
forest.fit(X, y)

from sklearn.model_selection import KFold, cross_val_score
cv = KFold(5, shuffle=True, random_state=0)
cross_val_score(forest, X, y, scoring="accuracy", cv=cv).mean()
# > 0.9701754385964911

# 2. 특징 중요도
idx = np.argsort(forest.feature_importances_)
names = bc.feature_names[idx]
values = forest.feature_importances_[idx]

# 그래프 그리는 코드
plt.figure(figsize=(10, 10))
plt.barh(names, values)
plt.xlabel("특성 중요도")
plt.title("Breath Cancer feature importanve")
plt.show()
```
![](/assets/images/Supervised7_8.png)


### 침고) 이미지 데이터의 특징 중요도 
extreme random forest를 통해 특징(픽셀)중요도를 나타낼 수 있다. 올리베티 얼굴 사진에 적용시켜 보자.
```py
from sklearn.ensemble import ExtraTreesClassifier
data = fetch_olivetti_faces()
X = data.data
y = data.target

from sklearn.datasets import fetch_olivetti_faces
forest = ExtraTreesClassifier(n_estimators=1000, random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
importances = importances.reshape(data.images[0].shape)

# Drawing
plt.figure(figsize=(8, 8))
plt.imshow(importances, cmap=plt.cm.bone_r)
plt.grid(False)
plt.title("픽셀 중요도(pixel importance)")
plt.show()
```
![](/assets/images/Supervised7_9.png)
- 검은색 부분의 픽셀이 사람의 모양을 결정하는 데 중요한 역할을 하는 부분이다.
- 그런데 '배경'부분(5시 방향 하단)에서 overfitting이 되는 경우가 발생한다. 이 경우 `Image Augmentation`이라는 방법을 이용한다. 원본 이미지에 다양한 변화(확장, 회전 등)를 준 복제 이미지를 여러 개 만들어서 분석해 배경이 미치는 영향이 적어지게 만든다.
