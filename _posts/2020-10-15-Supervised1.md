---
title:  "[머신러닝] 지도학습 - 2.1. 로지스틱 회귀분석"
excerpt: "Logistic 회귀분석에 대한 이론과 간단한 코드 정리"

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
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 좀 더 구체적인 자료를 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/10.01%20%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1%20%ED%9A%8C%EA%B7%80%EB%B6%84%EC%84%9D.html) 참고 부탁드립니다. 특히 아래 코드의 plt 그래프의 모습을 보고 싶으시면 링크 확인부탁드립니다.  

> 코드만 참고하시길 원하시면 [6. Scikit-Learn 패키지의 로지스틱 회귀 코드 예제(핵심)] 부분으로 이동하시면 됩니다.

## 0. 로지스틱 회귀분석
**로지스틱(Logistic) 회귀분석**은 회귀분석이라는 명칭과는 다르게 회귀분석 문제뿐만 아니라 분류문제에도 모두 사용될 수 있다. 로지스틱 회귀분석 모형에서는 종속변수가 이항분포를 따르고 모수 $\mu$가 독립변수 x에 의존한다고 가정한다. 그럴 경우 **로지스틱함수**는 y의 값이 **특정한 구간 내의 값 (0 ~ N)만 가질 수 있기 때문**에 종속변수가 이러한 특성을 가진 경우(Binaomial일 경우) 회귀분석 방법으로 쓸 수 있다. 
- 특히 이항분포의 특별한 경우(N = 1)로 y가 베르누이 확률분포인 경우가 있을 수 있다. 이 경우 **이진 분류 문제**에 사용할 수 있는데, 여기에서는 (N=1)일 때의 베르누이 확률분포를 따르는 로지스틱 회귀분석만 고려하기로 한다.
$$p(y|x) = Bern(y;\mu(x))$$
- $\mu$는 $x$에 따라 달라진다. 

 종속변수 y가 0또는 1인 **분류 예측 문제**를 풀 때는 x값을 이용해서 $\mu(x)$를 예측 한 후 $\hat{y}$값을 출력한다.
$$\hat{y}=\begin{cases}
1 & \mbox{if }\mu(x) \ge 0.5 \\
0 & \mbox{if }\mu(x) < 0.5
\end{cases}$$

- **회귀분석**을 할 때는 $\hat{y}$으로 y = 1이 될 확률값 $\mu(x)$를 직접 사용
$$\hat{y}=\mu(x)$$

## 1. 시그모이드 함수
로지스틱 회귀모형에서는 베르누이 확률 분포의 모수 $\mu$가 $\x$의 함수라고 가정한다. $\mu(x)$는 x에 대한 함수를 0부터 1사이의 값만 나올 수 있도록 **시그모이드 함수(sigmoid function)**라는 함수를 사용하여 변형한 것을 사용한다.
- Sigmoid는 S-like (S자와 닮았다)는 뜻.


- 시그모이드함수의 조건(종속변수의 모든 실수 값에 대해)
    - 유한한 구간(a, b) 사이의 **한정된(bounded) 값**을 가지고
    $$a < f(x) < b $$

    - 항상 **양의 기울기**를 가지는 **단조증가하는 함수의 집합**을 말한다. 
    $$ a>b → f(a) > f(b)$$
  
### 시그모이드 함수에 해당하는 함수들
- **로지스틱 함수** (보통 시그노이드 함수라고 하면 99프로 로지스틱함수를 가르킨다.)
$$logistic(z) = \sigma(z) = {1 \over 1+exp(-z)}$$

- **하이퍼볼릭탄젠트**
**'하이퍼볼릭탄젠트함수'**는 '로지스틱함수'를 위아래 방향으로 2배 늘리고 좌우 방향으로 1/2 축소한 것과 같다. 따라서 '로지스틱 함수'의 기울기의 4배이다.
$$tanh(z) = 2\sigma(2z)-1$$

- **오차(Error)함수**


## 2. 로지스틱함수
- 음의 무한대부터 양의 무한대까지의 **실수값을 0부터 1사이의 실수값으로 1대1 대응**시키는 시그모이드 함수이다.
- 보통 시그모이드 함수는 로지스틱함수를 가르킨다.

### 로지스틱 함수가 나오는 과정

0. $\mu$는 0부터 1사이의 값만 가진다. 
1. 베르누이 시도에서 1이 나올 확률 **$\mu$**와 0이 나올 확률 **$1 - \mu$**의 비율을 **승산비(odds ratio)**라고 한다. $\mu$를 승산비로 변환하면 0부터 양의 무한대까지의 값을 갖는다.
2. 승산비를 로그변환하면 음의 무한대부터 양의 무한대까지의 값을 갖는 **'로지트 함수(Logit function)'**가 된다.
$$z = logit(odds\ ratio) = log \left( {\mu \over 1-\mu} \right)$$


3. **로지트 함수의 역함수**를 구하면 **로지스틱 함수**가 된다. 즉, 로지스틱 함수는 음의 무한대부터 양의 무한대까지의 값을 가지는 **입력변수**를 0부터 1사이의 값을 가지는 **출력변수**로 변환한 것이다.

### 선형 판별함수
- 로지스틱함수 $\sigma(z)$를 사용하는 경우 $z$와 $\mu$값은 관계가 있다.
- $ z= 0$이면 $\mu = 0.5$이고, 
- $ z> 0$이면 $\mu > 0.5 → \hat{y}=1$이고, 
- $ z < 0$이면 $\mu < 0.5 → \hat{y}=0$이다. 


즉, $z$가 양수가 되면 $\mu$값은 커지게 되고, 반대로 음수가 되면 $\mu$값은 작아지게 된다. 따라서  $z$가 **분류 모형의 판별함수(decision function)**의 역할을 한다.  


로지스틱 회귀분석에서는 판별함수 수식으로 선형함수를 사용한다.
$$z=w^Tx$$
- 따라서 **판별 경계면***도 **'선형'**이 된다.

## 3. 로지스틱 회귀분석 모형의 모수 추정

 로지스틱 회귀분서의 목적은 모형의 모수 $w$의 값을 알아내는 것이다. 모수 $w$는 최대가능도 방법(MLE, Maximum Likelihood Estimation)으로 추정할 수 있다.

 > '모수 추정'의 세부적인 정의를 원하시면 위 [링크](https://datascienceschool.net/03%20machine%20learning/10.01%20%EB%A1%9C%EC%A7%80%EC%8A%A4%ED%8B%B1%20%ED%9A%8C%EA%B7%80%EB%B6%84%EC%84%9D.html)를 확인해주시기 바랍니다.

 - '베르누이 확률분포의 확률밀도함수'에 '로지스틱 함수'를 적용해서 조건부 확률을 구하고, 전체 데이터의 로그 가능도 $LL$을 구한 뒤, 로그가능도를 최대화하는 모수 $w$의 값을 구하기 위해 모수로 미분한다. 그렇게 한 차례 과정을 더 거치면 그레디언트 벡터의 수식을 아래와 같이 구할 수 있다.
$$\frac{\partial LL}{\partial \mu(x_i;w)} = \sum_{i=1}^N (y_i - \mu(x_i;w))x_i$$


- **그레디언트 벡터가 영벡터가 되는 모수의 값**이 **로그가능도($LL$)를 최대화**하는 값이다.
- 하지만 선형 모형과 같이 간단하게 그레디언트가 0이 되는 모수 $w$값에 대해 수식을 구할 수 없기 때문에 **수치적 최적화 방법**을 통해 반복적으로 **최적 모수 $w$의 값**을 구한다.

### 수치적 최적화
 **로그가능도 함수 $LL$**을 최대화하는 것은 $J = -LL$와 같은 **목적함수**를 최소화하는 것과 같다.
 최대 경사도(Steepest Gradient Descent) 방법을 사용하면, 그레디언트 벡터는 아래와 같다. 
$$g_k = \frac{d}{dw}(-LL)$$

그리고 이 방향으로 스텝사이즈 $\eta$만큼 이동한다.

$$\begin{split}
\begin{eqnarray}
w_{k+1} 
&=& w_{k} - \eta_k g_k \\
&=& w_{k} + \eta_k \sum_{i=1}^N \big( y_i  - \mu(x_i; w_k) \big) x_i\\
\end{eqnarray}
\end{split}$$
- **최적화** 이후 과최적화 방지를 위해 **'정규화'**를 진행한다.

## 중간 정리
- '로지스틱 회귀분석 모형'의 목적은 **모수 w값을 찾는 것**이다.
- y값은 0이 될 수도 있고, 1이 될 수도 있지만, **예측 값** $\hat{y}$의 값은 0이든 1이든 일정하게 정해져 있어야 한다.
- 모델링할 때의 데이터는 **'베르누이 분포'**이다. $x$가 정해지면 $\mu$가 정해진다. $\mu$가 정해지면 **확률적인 값** $y$가 나오게 된다. $\mu$가 높으면 1이 나올 확률이 높지만 꼭 그 값이 나온다고 할 수 없다. 하지만 **'예측값 $\hat{y}$'**의 경우 $x$가 주어지면 $\hat{y}$의 값은 고정이 되어 1이나 0이 나와야 한다. 이 값은 $\mu$ 값을 이용해서 0.5를 기준으로 높으면 1, 낮으면 0으로 구할 수 있다.

## 4. StatsModels 패키지의 로지스틱 회귀
StatsModels 패키지는 베르누이 분포를 따르는 로지스틱 회귀 모형 `Logit`를 제공한다. 사용방법은 `OLS` 클래스 사용법과 동일하다. 
- 종속변수와 독립변수 데이터를 넣어 모형을 만들고 `fit` 메서드로 학습 시킨다. > 참고: `fit`메서드의 `disp=0`인수는 최적화 과정에서 문자열 메세지를 나타내지 않는 역할을 한다.
- 결과 객체에서 `summary`메서드를 사용해 리포트를 출력할 수 있다.

```py
### 가상의 데이터 생성
from sklearn.datasets import make_classification
X0, y = make_classification(n_features=1, n_redundant=0, n_informative=1,
n_clusters_per_class=1, random_state=4)

### 로지스틱 회귀 모형 생성 및 최적화
X = sm.add_constant(X0)
logit_mod = sm.Logit(y, X)
logit_res = logit_mod.fit(disp=0)

### 리포트 출력
print(logit_res.summary())

### coef_x1(x)값 그래프 그리기
xx = np.linspace(-3, 3, 100)
mu = logit_res.predict(sm.add_constant(xx))
plt.plot(xx, mu, lw=3)

# 이하 생략

```
### 기준값(threshold) 구하기
![](/assets/images/Supervised1_1.jpg)
 리포트 결과를 보면 원래의 판별함수의 수식은 다음과 같다.
$$\mu(x) = \sigma(4.2382x + 0.2515)$$

따라서 $4.2382x +0.2515 = 0.5$이 되는 값 $x$는 $(0.5 -0.2515)/4.2382$로 구해야 한다. 하지만`const`상수항의 값은 유의확률을 감안하면 0과 마찬가지이므로 $\mu(x)$의 값은 다음과 같다고 할 수 있다.
$$\mu(x) = \sigma(4.2382x)$$

그래서 이렇게 생각하면 z값의 부호를 나누는 기준값(threshold) $x$는 $4.2382x = 0.5$로 실질적으로 0.5/4.2382 = 0.118이라고 할 수 있다. 

### 판별함수 (fittedvalues)
- Logit 모형의 결과 객체에는 `fittedvalues`라는 속성으로 판별함수 $z=w^Tx$값이 들어가 있다. 이를 통해 **분류문제**를 풀 수 있다. (ROC커브를 그릴 수 있다)

```py
### 위 코드의 데이터 활용

plt.scatter(X0, y, c=y, s=100, edgecolor="k", lw=2, label="데이터")
plt.plot(X0, logit_res.fittedvalues * 0.1, label="판별함수값")
plt.legend()
plt.show()
```

## 5. 로지스틱 회귀 성능 측정
 로지스틱 회귀 성능은 **맥파든 의사결정계수**(McFadden pseudo R square)값으로 측정한다.
$$R^2_{pseudo} = 1 - {G^2\over G^2_0}$$
항상 1보다 작은 값을 가진다.

- $G^2$: **이탈도**라고 하는 양으로 모형이 정확한 경우 0, 성능이 나빠질 수록 큰 값을 가짐
- $G^2_0$: **귀무모형으로 측정한 이탈도**로 가장 성능이 나쁜 모형을 말한다.

따라서 맥파든 의사결정계수는 가장 성능이 좋을 때는 1이고, 가장 성능이 나쁠 때는 0이 된다. (이탈도를 구해 1을 빼준다는 것을 기억)

### `log_loss`함수
scikit-learn패키지의 metric 서브패키지에는 로그 손실을 계산하는 `log_loss`함수가 있다. `normalize=False`로 놓으면 **이탈도**와 같은 값을 구한다.

```py
## 위 코드의 최적 모형 사용

## 이탈도 log_loss계산
from sklearn.metrics import log_loss
y_hat = logit_res.predict(X)
log_loss(y, y_hat, normalize=False)

## 귀무모형으로 측정한 이탈도
mu_null = np.sum(y) / len(y) # 귀무 모형의 모수값
y_null = np.ones_like(y) * mu_null 
log_loss(y, y_null, normalize=False) # log_loss계산

## 맥파든 의사결정계수 값 계산
mcfadden = 1 - (log_loss(y, y_hat) / log_loss(y, y_null))
```

## 6. Scikit-Learn 패키지의 로지스틱 회귀 코드 예제 (핵심)

> 위에는 다 버리고 코드만 봐도 됩니다.

```py
### 가상의 데이터 생성
from sklearn.datasets import make_classification
X0, y = make_classification(n_features=1, n_redundant=0, n_informative=1,
n_clusters_per_class=1, random_state=4)

## 로지스틱 회귀 모델 생성
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X0, y)

## 보고서 출력
print(model.summary(y, y_pred))

## 기준값 구하기 / params[0]은 상수항의 모수, params[1]은 x의 모수
threshold = (0.5 - model.params[0]) / model.params[1]

## 기준값을 기준으로 0.5이상으로 클래스 분류
y_pred = model.predict(threshold >= 0.5)

## Confusion Matrix출력
from sklearn.metrics import confusion_matrix
confusion_matrix(y, y_pred)

## classification_report출력
from sklearn.metrics import clasification_report
print(classification_report(y , y_pred))
```
> 확장. ROC 커브 및 AUC 구하기

```py
## 판별함수(fittedvalues)를 이용한 roc커브 그리기
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y, model.fittedvalues)
plt.plot(fpr, tpr)
plt.show()

# auc값 구하기
from sklearn.metrics import auc
auc(fpr, tpr)
```

## 7. 로지스틱 회귀를 사용한 회귀분석
로지스틱 회귀는 분류문제뿐만 아니라 종속변수 $y$가 0부터 1까지 막혀있는 회귀분석 문제에도 사용할 수 있다.이때는 다음처럼 $\mu$값을 종속변수 $y$의 예측값으로 사용한다.
$$\hat{y} = \mu(x)$$

- 만약 실제 y의 범위가 0부터 1이 아니면 **스케일링**을 통해 종속변수 값을 바꿔야 한다.

> 1974년도 "여성은 가정을, 남성은 국가 일을 맡아야 한다에 대한 찬성, 반대 입장"에 관한 데이터이다.
- 각 열은 `education(교유기간)`, `sex(성별)`, `agree(인원)`, `disagree(인원)`, `ratio(찬성비율)`로 구성되어 있다.

```py
## 데이터 생성(wrole(women role))

data_wrole = sm.datasets.get_rdataset("womensrole", package="HSAUR")
df_wrole = data_wrole.data
df_wrole["ratio"] = df_wrole.agree / (df_wrole.agree + df_wrole.disagree)
df_wrole.tail()

## 모형 생성 및 분석 결과 (education과 sex)
model_wrole = sm.Logit.from_formula("ratio ~ education + sex", df_wrole)
result_wrole = model_wrole.fit()
print(result_wrole.summary())

## 분석 결과에서 성별은 유의미하지 않다는 사실 도출 (sex 제외한 모형)
model_wrole2 = sm.Logit.from_formula("ratio ~ education", df_wrole)
result_wrole2 = model_wrole2.fit()
print(result_wrole2.summary())

## 그래프를 통한 예측 그래프 확인 

sns.scatterplot(x="education", y="ratio", data=df_wrole) # education열에 해당하는 데이터 (점)
xx = np.linspace(0, 20, 100)
df_wrole_p = pd.DataFrame({"education": xx})
plt.plot(xx, result_wrole2.predict(df_wrole_p),
 "r-", lw=4, label="예측") # education열로 예측한 모형 (곡선)
plt.legend()
plt.show()
```
![](/assets/images/Supervised1_2.jpg){: .align-center}