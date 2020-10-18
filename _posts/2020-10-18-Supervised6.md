---
title:  "[머신러닝] 지도학습 - 4.1. 의사결정나무"
excerpt: "판별적 확률모형에 해당하는 의사결정나무"

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
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/12.01%20%EC%9D%98%EC%82%AC%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 1. 의사결정나무
의사결정나무(decision tree)는 여러 가지 규칙을 순차적으로 적용하면서 독립 변수 공간을 분할하는 분류모형이다.

- 분류(Classification)와 회귀 분석(Regression)에 모두 사용될 수 있기 때문에 **CART(Classification And Regression Tree)**라고도 한다.
 
### 의사결정나무를 이용한 분류학습
1. 하나의 독립변수를 선택하고 **기준값(threshold)**을 정한다.이를 **분류규칙**이라고 한다. 
2. 전체 학습 데이터 집합(부모 노드)에서 기준값보다 작은 데이터 그룹(자식 노드1)과 해당 독립수의 값이 기준값보다 큰 데이터 그룹(자식 노드2)으로 나눈다.
3. 각각의 자식 노드에 대해 1~2단계를 반복해 하위의 자식노드를 만들어 나간다. 단, 자식 노드에 한 가지 클래스의 데이터만 존재하게되면, 그 노드는 더 이상 자식 노드를 나누지 않고 중지한다. 

- **분류학습**을 통해 노드가 계속 증가하는 나무(tree)형태로 표현할 수 있는데, 깊이(depth)가 너무 길어지지 않게 한계를 정해놓을 수 있다.

- **분류학습**을 멈추는 3가지 경우 

  1. 노드 분류가 더 이상 되지 않을 때  
  2. 한계 깊이를 정해놨을 때   
  3. 데이터의 개수가 너무 적을 때  

> `fit` 단계에서 이렇게 나무를 만드는 과정이 존재한다.

### 의사결정나무를 이용한 분류예측
- 테스트 데이터를 집어넣어서, 해당 데이터가 **분류 규칙**을 차례대로 적용하며 어떤 class의 '리프 노드'로 떨어지는지 확인한다. 그리고 **마지막에 도달하는 노드의 조건부 확률 분포를 이용**해 **클래스를 예측**한다.
$$\hat{Y} = arg\ \underset{k}max P(Y=k|X_{test})_{last node}$$

- 해당 모형은 베이즈 정리를 사용하지 않았기 때문에 '생성모형'이 아닌 **'판별적 확률모형'**에 해당한다. 로지스틱 회귀 모형과 비슷하다.


## 2. 분류규칙을 정하는 방법
부모 노드와 자식 노드 간의 **엔트로피를 가장 낮게** 만드는 최상위 독립변수와 기준값을 찾는다.
- 이러한 기준을 정량화한 것이 **'정보 획득량(IG, information gain)'**이다.
  
따라서 **모든 독립변수**와 **모든 가능한 기준값**에 대해 정보획득량을 구해 **가장 정보획득량이 큰 독립변수**와 **기준값**을 선택한다.
 
### 정보획득량(★)
$X$라는 조건에 의해 확률 변수 $Y$의 엔트로피가 얼마나 감소하였는가를 나타내는 값
- 정보획득량은 $Y$의 엔트로피에서 $X$에 대한 $Y$의 조건부 엔트로피를 뺸 값으로 정의
$$IG[Y, X] = H[Y] - H[Y|X]$$

> 엔트로피: 확률이 얼마나 '골고루 분포'되어있는지(엔트로피 증가할 때), '한쪽에 집중'되어있는지 알려주는 지표(엔트로피 감소할 때). 

> 엔트로피 계산 방법은 [출처 링크](https://datascienceschool.net/03%20machine%20learning/12.01%20%EC%9D%98%EC%82%AC%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4.html) 활용

## 3. Scikir-Learn의 의사결정나무 클래스
`DecisionTreeClassifier`클래스로 구현되어 있다. 붓꽃 분류 문제를 예로 들어 의사결정나무를 설명한다. 독립변수 공간을 공간상에 표시하기 위해 두 개의 독립변수(꽃의 길이와 폭)만을 사용한다.
- `criterion(기준)`: default값은 '지니 인퓨리티'로 log를 쓰지 않으므로 계산이 빠르다. 하지만 위에서 설명한 'entropy'로 설정했다.
- `max_depth(깊이)`: 나무 깊이(depth)의 한계치

```py
# 데이터 로드
from sklearn.datasets import load_iris
data = load_iris()
y = data.target
X = data.data[:, 2:]
feature_names = data.feature_names[2:]

from sklearn.tree import DecisionTreeClassifier
tree1 = DecisionTreeClassifier(criterion='entropy', max_depth=1, random_state=0).fit(X, y)
```
### 의사결정나무 시각화 코드
- `draw_decision_tree`함수: 의사결정나무의 의사결정 과정의 세부적 내역을 다이어그램으로 보여준다.
- `plot_decision_regions`함수: 의사 결정에 의해 데이터의 영역이 어떻게 나뉘어졌는지 시각화하여 보여준다. 또한 분류 진행 과정에 대한 설명도 생략하겠다.

> 해당 코드는 너무 길어지기 때문에, 생략한다. 위 [출처 링크](https://datascienceschool.net/03%20machine%20learning/12.01%20%EC%9D%98%EC%82%AC%EA%B2%B0%EC%A0%95%EB%82%98%EB%AC%B4.html)를 통해 세부적인 코드와 그래프를 확인할 수 있다.

```py
# 해당 코드는 pydot을 import와 ipython을 import해야 사용가능하다.

import io
import pydot
from IPython.core.display import Image
from sklearn.tree import export_graphviz

def draw_decision_tree(model):
  # 생략

def plot_decision_regions(X, y, model, title):
  # 생략

```

## 4. 연습문제
1. 붓꽃 분류 문제에서 꽃받침의 길이와 폭(sepal length, sepal width)을 사용하여 `max_depth=3` 인 의사결정나무 모형을 만들고 정확도(accuracy)를 계산하라.
2. K=5 인 교차 검증(K-Fold)을 통해 테스트 성능 평균을 측정하라.
3. `max_depth` 인수를 바꾸어 가면서 테스트 성능 평균을 구하여 `cross validation curve`를 그리고 가장 테스트
성능 평균이 좋은 `max_depth` 인수를 찾아라.

```py
# 1. 모델 생성 및 정확도 구하기
from sklearn.datasets import load_iris
iris = load_iris()
X1 = iris.data
y1 = iris.target
X1 = X1[:, :2]

from sklearn.tree import DecisionTreeClassifier
model1 = DecisionTreeClassifier(max_depth=3).fit(X1, y)
y1_pred = model1.predict(X1)

from sklearn.metrics import accuracy_score
accuracy_score(y1, y1_pred)
# > 0.8133333333333334

# 2. K=5인 K-fold 교차검증과 테스트 성능의 평균
from sklearn.model_selection import KFold, cross_val_score
cv = KFold(5, shuffle = True, random_state=0)
cross_val_score(model1, X1, y1, scoring ="accuracy", cv=cv).mean()
# > 0.7466666666666666

# 3. max_depth에 따른 테스트 정확성
mean_test_accuracy = []
train_accuracy = []
for max_depth in range(3, 10):
    cv = KFold(5, shuffle=True, random_state=0)
    model = DecisionTreeClassifier(
        criterion='entropy', max_depth=max_depth).fit(X1, y1)
    train_accuracy.append(accuracy_score(y1, model.predict(X1)))
    mean_test_accuracy.append(cross_val_score(
        model, X1, y1, scoring="accuracy", cv=cv).mean())

# 3-1. 그래프 그리기
plt.plot(np.arange(3, 10), train_accuracy)
plt.plot(np.arange(3, 10), mean_test_accuracy)
plt.show()

```
![](/assets/images/Supervised6_1.png)
- train 데이터에 정확도는 계속 올라가지만, test데이터의 경우 max_depth가 4일 때 가장 높은 정확도를 보여준다.

## 5. 타이타닉호 생존자 예측
`Decisiontree`를 이용하면 사람이 이를 통해 insight를 얻기 쉽다.
```py
# 0. 데이터 로드
## 여기에서는 pclass, sex, age 3개의 데이터만 이용

df = sns.load_dataset("titanic")
feature_names = ["pclass", "sex", "age"]
dfX = df[feature_names].copy()
dfy = df["survived"].copy()
dfX.tail()

# 1. 데이터 전처리
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder

# Decisiontree 이용을 위해 모든 변수를 더미변수화 시켜줘야 한다. 
## 의사결정나무는 '카테고리값'을 구분하지 못한다.

# sex를 0과 1로 구분 / 
dfX["sex"] = LabelEncoder().fit_transform(dfX["sex"])
dfX.tail()

# age열의 null값에 '평균값'을 넣어줌
dfX["age"].fillna(dfX["age"].mean(), inplace=True)
dfX.tail()

# pclass도 '더미변수화'되어야 한다.
dfX2 = pd.DataFrame(LabelBinarizer().fit_transform(dfX["pclass"]), 
                    columns=['c1', 'c2', 'c3'], index=dfX.index)
dfX = pd.concat([dfX, dfX2], axis=1)
del(dfX["pclass"]) # 열 삭제
dfX.tail()

## 2. 모델 생성 및 의사결정나무 그리기
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
X_train, X_test, y_train, y_test = train_test_split(dfX, dfy, test_size=0.25, random_state=0)
model = DecisionTreeClassifier(
criterion='entropy', max_depth=3, min_samples_leaf=5).fit(X_train, y_train)

#...(의사결정나무 그리기 생략)...

## 3. confusion_matrix 및 분류 결과 보고서 출력
from sklearn.metrics import confusion_matrix, Classification_report
confusion_matrix(y_train, model.predict(X_train)) # 전체 데이터 중 0.75%
confusion_matrix(y_test, model.predict(X_test)) # 전체 데이터 중 0.25%

print(classification_report(y_train, model.predict(X_train)))
print(classification_report(y_test, model.predict(X_test)))
```
- train 데이터의 리포트
![](/assets/images/Supervised6_2.jpg)

- test 데이터의 리포트
![](/assets/images/Supervised6_3.jpg)

## 6. Greedy 의사 결정
**의사결정나무**에서는 특징의 선택이 greedy한 방식으로 이루어지기 때문에, 의사결정나무의 선택이 반드시 최적의 선택인 것은 아니다.

### greedt한 의사결정의 예시
```py
X = [
#x1|x2|x3  
[0, 0, 0],
[1, 0, 0],
[0, 0, 1],
[1, 0, 1],
[0, 1, 0],
[1, 1, 0],
[0, 1, 1],
[1, 1, 1],
]
y = [0,0,1,1,1,1,0,0]
```
위 표를 예를 들어 X는 x1, x2, x3 3개의 독립변수를 갖는다. **첫 노드**에서 분류할 때 x1, x2, x3의 성능은 같다(0과 1이 4개씩 존재). 그런데 만약 첫 노드에서 특징으로 x1을 선택하면 2단계로 완벽한 분류를 하는 것이 불가능한 반면에, **첫 노드에서 특징으로 x3을 선택**하면 두번째 특징으로 x2를 선택함으로써 **2단계만에 완벽한 분류를** 할 수 있다. 
- 하지만 어떤 특징이 최적의 선택일지는 첫 노드에서 특징을 결정할 때 알 수 없다.

## 7. 회귀나무
예측값 $\hat{y}$을 각 특징값 영역마다 고정된값 $y_1, y_2$를 사용해서, **기준값 및 $y_1, y_2$**($x$가 기준값보다 크면 $y_1$, $x$가 기준값보다 크면 $y_2$)를 선택하는 **목적함수로 오차제곱합을 사용**하면 **회귀분석**이 가능하다. 이를 **'회귀 나무'**라고 한다.

- 회귀나무의 경우 `DecisionTreeRegressor`를 이용한다.

```py
# 아래 코드는 참고만

# 샘플 데이터 생성
from sklearn.tree import DecisionTreeRegressor
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# 회귀나무 생성
regtree = DecisionTreeRegressor(max_depth=3)
regtree.fit(X, y)

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_hat = regtree.predict(X_test)

# result plot
plt.figure()
plt.scatter(X, y, s=20, edgecolor="black", c="darkorange", label="데이터")
plt.plot(X_test, y_hat, color="cornflowerblue", linewidth=2, label="예측")
plt.xlabel("x")
plt.ylabel("$y$ & $\hat{y}$")
plt.title("회귀 나무")
plt.legend()
plt.show()

```
![](/assets/images/Supervised6_4.png)