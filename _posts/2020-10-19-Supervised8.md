---
title:  "[머신러닝] 지도학습 - 4.3. 부스트방법"
excerpt: "'모형결합' 중 부스팅 방법론(에이다 부스트, 그레디언트 부스트)"

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
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/12.03%20%EB%B6%80%EC%8A%A4%ED%8C%85%20%EB%B0%A9%EB%B2%95.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 0. 부스트 방법
부스트(boost) 방법은 미리 정해진 갯수의 모형 집합을 사용하는 것이 아니라 하나의 모형에서 시작하여 모형 집합에 포함할 **개별 모형을 하나씩 추가**한다.

- 모형의 집합은 **위원회(commitee) $C$**라고 하고 $m$개의 모형을 포함하는 위원회를 **$C_m$**으로 표시한다., 각 개별 모형을 **'약 분류기(weak classifier)'라고 하며, $k$**로 표기한다. 
- 위원회에 추가할 **개별 모형 $k_m$의 선택 기준**은 그 **전단계의 위원회의 성능을 보완**할 수 있는지이다. 
- 위원회 $C_m$의 최종 결정은 다수결 방법을 사용하지 않고 **각각의 개별 모형의 출력**을 **가중치 $\alpha$로 가중선형조합한 값**을 **판별함수**로 사용. 일종의 softvoting방법과 같다.
- 부스트 방법은 **이진 분류**에만 사용할 수 있으며 **y값은 -1 or 1**값만 가진다.
$$y = -1\ or\ 1$$
$$C_m(x_i) = sign\ (\alpha_1k_1(x_i) + \cdots + a_mk_m(x_i))$$

## 1. 에이다 부스트
에이다부스트라는 이름은 '적용 부스트(adaptive boost)'라는 용어에서 나왔다. 
손실함수 $L$을 사용해서, 이 손실함수를 최소화하는 모형 $k_m$을 선택한다.
$$L_m = \sum^{N}_{i=1}w_{m,i}I(k_m(x_i) \ne y_i) $$

- 위 식에서 $I$는 indicator함수(지시함수)다(괄호 안의 값이 참이면 1 거짓이면 0).
$I$는 벌점을 매기는 방식으로 여기에서는 틀리면 손실함수의 점수가 증가한다.  

그렇다면 '가중치 $w_i$값을 어떻게 정할 것인가'하는 문제와 또 '그 문제의 벌점을 어떻게 바꿀 것인가'하는 두 가지 이슈를 해결해야 한다. 즉, 위원회$C_m$에 포함될 개별 모형 **$k_m$이 선택된 후**에는 **가중치 $\alpha_m$을 결정**해야 한다. 에이다 부스트에서는 다음과 같은 식을 사용한다.
$$\epsilon_m = \frac{\Sigma^N_{i=1} w_{m,i}I(k_m(x_i) \ne y_i)}{\Sigma^N_{i=1}w_{m,i}}$$

이때, $\Sigma^N_{i=1}w_{m,i}$은 문제가 다 틀린 경우를 의미한다. 따라서 **$\epsilon$의 값**은 **성능이 좋으면 0**에 가까운 값이, **성능이 나쁘면 1**에 가까운 값이 나온다.  

따라서 $\alpha_m$의 값은 아래와 같이 나온다.(로지트 함수를 이용) 
$$\alpha_m = \frac{1}{2}log \left( \frac{1-\epsilon_m}{\epsilon_m} \right)$$
그러므로 $\alpha_m$의 값은 $(-\infty \sim +\infty)$ 값이 나온다.

따라서 성적이 좋은 애들에게는 가중치를 높게(투표권을 많이), 나쁜 애들에게는 가중치를 낮게(투표권을 적게)배분할 수 있게 한다. 그런데 데이터에 대한 가중치 $w_{m,i}$는 최초에는 모든 데이터에 대해 같은 값을 가지지만 위원회가 증가하면서 위원회 $C{m-1}$이 맞춘 문제는 작게, 틀린 문제는 크게 확대(Boosting)된다.

### 정리
>  Boosting(확대)라는 이름이 붙은 이유는 틀린 문제들의 가중치를 **확대**시키면서 분류를 수행하기 때문이다.

결론적으로 에이다부스팅은 사실 다음과 같은 손실함수 $L_m$을 최소화하는 $C_m$을 찾아가는 방법이라고 할 수 있다. 
$$L_m = \sum^N_{i=1}exp(-y_iC_m(x_i))$$

개별 멤버 $k_m$과 위원회의 관계는 아래와 같다.
$$C_m(x_i) = \sum_{j=1}^m \alpha_j k_j(x_i) = C_{m-1}(x_i) + \alpha_mk_m(x_i)$$

이 관계를 위의 식에 대입하고, $y_i$와 $k_M(x_i)$가 1또는 -1값만 가질 수 있다는 점 등을 이용하면 특정한 식을 구할 수 있다.   
그래서 결국 $L_m$을 최소화하려면 $\sum_{i=1}^N w_{m, i}I(k_m(x_i) \ne y_i)$을 최소화하는 $k_m$함수를 찾은 다음 $L_m$을 최소화하는 $\alpha_m$을 찾아야 한다.

$$\frac{dL_m}{d_\alpha_m} = 0$$

이 조건으로부터 $\alpha_m$ 공식을 유도할 수 있다.

(구체적인 증명은 [원본 링크](https://datascienceschool.net/03%20machine%20learning/12.03%20%EB%B6%80%EC%8A%A4%ED%8C%85%20%EB%B0%A9%EB%B2%95.html) 참조)

### 예제 코드
scikit-learn의 ensemble 서브패키지가 제공하는 `AdaBoostClassifier` 클래스를 사용해서 분류 예측을 할 수 있다.  
아래 예제코드는 **약분류기로**는 '깊이가 1인 단순한 의사결정나무'를 채택하였다. 또한 모형은 20개 생성해서 진행하는 것으로 설정했다.
- 또한 각 표본 데이터의 가중치 값을 알아보기 위해 기존 `AdaBoostClassifier`클래스를 서브클래싱`MyAdaBoostClassifier`하여 가중치를 속성으로 저장하도록 수정한 모형을 사용한다. (참고사항)

```py
## 데이터 생성
from sklearn.datasets import make_gaussian_quantiles
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
X1, y1 = make_gaussian_quantiles(cov=2.,
n_samples=100, n_features=2,
n_classes=2, random_state=1)
X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
n_samples=200, n_features=2,
n_classes=2, random_state=1)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, - y2 + 1))

## Adaboost 클래스 생성(여기에서는 MyAdaBoostClassifier 사용)
model_ada = MyAdaBoostClassifier(DecisionTreeClassifier(max_depth=1, random_state=0), n_estimators=20)
model_ada.fit(X, y)

### MyAdaBoostClassifier 및 plot_result 함수 정의 생략()

```
![](/assets/images/Supervised8_1.png)
그래프를 보면 일부 데이터에서 과촤적화(overfitting)가 발생하는 부분들을 확인할 수 있다.


### 에이다 부스트 모형 정규화
 과최적화가 되는 경우 학습 속도(learning range)를 조정하여 정규화 할 수 있다. 이렇게 함으로써 **필요한 멤버의 수를 강제로 증가**시켜서 과최적화를 막는 역할을 한다.
$$C_m = C_{m-1} + \mu\alpha_mk_m$$

 `AdaBoostClassifier클래스` `learning_rate`인수를 1보다 적게 주면 새로운 멤버의 가중치를 제대로 낮춘다.

### 연습문제
- 멤버의 수를 1부터 500까지 100씩 증가하면서 성능의 변화를 확인하고, 과최적화를 확인한다. 또한 성능이 가장 좋은 멤버의 수를 찾아라.

```py
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
mean_test_accuracy = []
for n in np.arange(1, 501, 100):
    model1 = AdaBoostClassifier( #  learning_rate=0.5
        DecisionTreeClassifier(max_depth=1), n_estimators=n)
    mean_test_accuracy.append(cross_val_score(model1, X, y, cv=5).mean())

plt.plot(np.arange(1, 500, 100), mean_test_accuracy)
plt.show()
```
![](/assets/images/Supervised8_2.png)

## 2. 그레디언트 부스트
 변분법(calculus of variations)을 사용한 모형이다. 간단하기에 그냥 미분과 같다고 생각해도 된다. 함수 f(x)를 최소화하는 $x$는 gradient descent방법으로 찾을 수 있다.
 $$ \frac{d L_m}{d\alpha_m}$$
 
이를 활용해, **그레디언트 부스트 모형()**에서는 손실범함수를 최소화$L(y, C_{m-1})$하는 개별함수 {k_m}을 찾는다. 이론적으로 가장 최적의 함수는 범함수의 미분이다. 
$$C_m = C_{m-1} - \alpha_m\frac{\delta L(y,C_{m-1})}{\delta C{m-1}} = C_{m-1} + \alpha_mk_m$$

에이다부스트에서는 놓친 부분을 **벌점**을 이용해서 했다면, 여기서는 실제 목푯값 y와 $C_{m-1}과의 차이 즉, **'잔차'**라는 회귀분석 함수($y-C_{m-1}$)를 이용해서 앞 위원회가 놓친 부분을 활용한다. 

- 그레디언트 부스트 모형은 분류/회귀 문제에 상관없이 **개별 멤버 모형**으로 회귀분석 모형을 사용한다. 따라서 `decisionTree`라는 것을 명시하지 않아도 함수를 이용할 수 있다.

- 그레디언트 부스트 모형에서는 다음과 같은 과정을 반복해서 멤버와 가중치를 계산한다.
1. $- \frac{\delta L(y,C_{m})}{\delta C{m}}$를 목표값으로 개별 멤버 모형 $k_m$을 찾는다.
2. (y-(C_{m-1} + a_mk_m))^2를 최소화하는 스텝사이즈 $\alpha_m$을 찾는다.
3. C_m = C_{m-1} + a_mk_m 최종 모형을 갱신한다. 

```py
# 0. 샘플 데이터는 위의 코드 사용

# 1. 모델 생성 (100개의 개별 모형 생성)
from sklearn.ensemble import GradientBoostingClassifier
model_grad = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=0).fit(X, y)

####### 그래프 코드 생략 #########

```
![](/assets/images/Supervised8_3.png)

### XGBoost 라이브러리 (참고)
Gradient Boost를 사용할 때, XGBoost를 활용하면 속도가 더 빠르다. 이전까지는 RandomForest를 사용했지만, XGBoost 라이브러리가 나온 이후 이 XGBoost가 가장 많이 사용됐다.
- 여기에서는 정규화 방법으로 (learning_rate 말고) drop_out이라는 방법을 한 가지 더 이용한다. 데이터가 들어오기 전에 일부 멤버 몇 가지를 강제로 제외하고, 다른 멤버를 추가함으로써 여러 조합의 멤버를 증가시킨게 하는 것이다.`reg_alpha`인수로 설정한다.
    
> 보통 실무에서는 '랜덤포레스트'와 'Gradient부스팅'방식이 가장 많이 쓰인다고 한다.

```py
# xgboost 라이브러를 설치해야 한다.
import xgboost
model_xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=1, random_state=0).fix(X, y)

####### 그래프 코드 생략 #########
```

### LightGBM 라이브러리 (참고)
- Gradiant 부스트방법 중 가장 최근 라이브러리

```py
# xgboost 라이브러를 설치해야 한다.
import xgboost
model_xgb = xgboost.XGBClassifier(n_estimators=100, max_depth=1, random_state=0)

####### 그래프 코드 생략 #########
```

> XGBoost와 LightGBM의 그래프는 위 '그레디언트 부스트'방식으로 그린 그래프와 큰 차이가 없다. 대신 학습성능 면에서 속도가 차이가 나게 된다.