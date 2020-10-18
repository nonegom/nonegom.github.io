---
title:  "[머신러닝] 지도학습 - 3.1. 선형판별분석법과 이차판별분석법"
excerpt: "생성모형 중 선형판별분석법과 이차판별분석법"

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
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/11.01%20%EC%84%A0%ED%98%95%ED%8C%90%EB%B3%84%EB%B6%84%EC%84%9D%EB%B2%95%EA%B3%BC%20%EC%9D%B4%EC%B0%A8%ED%8C%90%EB%B3%84%EB%B6%84%EC%84%9D%EB%B2%95.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 plt 그래프의 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 0. 가능도

> 선형판별분석법(LDA)과 이차판별분석법(QDA)는 대표적인 **확률론적 생성모형**이다. $y$의 클래스 값에 따른 $x$의 분포에 대한 정보(**가능도**)를 먼저 알아낸 후. **베이즈 정리**를 사용해 주어진 x에 대한 y의 확률분포를 찾아낸다.


감기가 걸린 사람의 체온을 예측하는 모형을 예시로 들어보면, 예측 결과인 y값을 정상인일 때는 "y=0" 환자일 때는 "y=1"로 가정하자. 만약 정상인 사람들의 온도가 대략 36.5도가 나왔고, 환자들의 온도가 38.0도가 나왔다면 우리는 보통 

> 정상(y=0)일 때, 온도가 36.5도 주위겠구나.   
환자(y=1)일 때, 온도가 38.0도 주위겠구나.

를 알아낼 수 있다. 이를 **가능도**라고 한다.

- 그런데 y값이 꼭 이진일 필요는 없다.

## 1. 생성모형
생성모형에서는 **베이즈 정리**를 사용해서 조건부 확률$p(y=k\ |\ x)$을 계산한다.

- 베이즈 정리
$$P(y=k\ |\ x) = {P(y=k\ |\ x) P(y=k)\over P(x)}$$

**분류 문제**를 풀기 위해서는 **각 클래스 $k$**에 대한 확률을 비교해서 **가장 큰 값**을 선택한다.

- 값이 같은 분모 $P(x)$은 굳이 계산하지 않아도 괜찮다.
 결국 **생성적 확률 모형**의 핵심은 **'가능도 부분'**을 구하는 것이다. 이걸 구하는 방법론에 따라 이름이 달라지게 된다.
$$P(y=k\ |\ x) \propto P(y=k\ |\ x) P(y=k)$$

### 사전확률 P(y=k)
특별한 정보가 없는 경우 **"y=k인 데이터의 수 / 모든 데이터의 수"**로 한다. 하지만 다른 정보가 주어진 경우 그 값을 사용하면 된다.

### 가능도 계산법
1. [**정규 가정**] : $P(x \mid y=k)$ 가 **특정한 확률분포 모형**을 따른다고 **가정**한다. 즉, 확률밀도함수의 형태를 가정한다.(x의 분포를 찾는다.)  
2. [**$\mu, \sigma^2$ 계산**] : $k$번째 클래스에 속하는 학습 데이터 $(x_1, \ ...\ , x_N)$을 사용하여 이 모형의 **모수 값**을 구한다.  
3. [**$P(x)$**] : 모수 값을 알고 있으므로 $P(x \mid y = k)$의 확률 밀도 함수를 구한 것이다. 즉, 새로운 독립변수 값 $x$이 어떤 값이 되더라도 $P(x \mid y=k)$의 값을 계산할 수 있다.

- 위를 다 구해 '가능도'를 계산하면, 독립변수 값 $x$로 어떤 것을 주더라도 확률밀도 값을 알아낼 수 있다.  

## 2. 이차판별분석법 (QDA)
이차판별분석법(QuadraticDiscriminantAnaysis, QDA)에서는 **독립변수가 실수**이고 **확률분포가 다변수 정규분포**라고 가정한다. 단, x분포의 위치와 형태는 클래스에 따라 달라질 수 있다. 위 분포들을 알고 있으면 '조건부 확률 분포'는 **베이즈 정리**와 **전체 확률의 법칙**으로 구할 수 있다.

- 다시 말하면, QDA는 $\mu$(뮤)값과 $\sigma^2$(시그마)값이 클래스마다 다르고, 뮤값과 시그마 값을 알기 때문에 이제 **가능도**를 계산할 수 있는 능력이 된 것이다.

### 코드
Scikit-Learn은 `QuadradicDiscriminantAnalysis` 클래스를 제공한다.  
학습용 데이터에서 **가능도를 추정한 후**에는 다음과 같은 **속성**을 가지게 된다.
- `priors_` : 각 클래스 $k$의 사전확률.
- `means_` : 각 클래스 $k$에서 $x$의 기댓값 벡터 $\mu_k$의 추정치 벡터.
- `covariance_` : 각 클래스 $k$에서 $x$의 공분산 행렬 $\Sigma_k$의 추정치 행렬. (생성자 인수 store_covariance 값이
True인 경우에만 제공)

> qda의 속성 `priors_`, `means_`, `covariance_`을 통해서 **확률밀도함수**를 구할 수 있다.

```py
## 샘플 데이터 생성 (확률변수가 다변수 정규분포)
N = 100
rv1 = sp.stats.multivariate_normal([ 0, 0], [[0.7, 0.0], [0.0, 0.7]])
rv2 = sp.stats.multivariate_normal([ 1, 1], [[0.8, 0.2], [0.2, 0.8]])
rv3 = sp.stats.multivariate_normal([-1, 1], [[0.8, 0.2], [0.2, 0.8]])
np.random.seed(0)
X1 = rv1.rvs(N)
X2 = rv2.rvs(N)
X3 = rv3.rvs(N)
y1 = np.zeros(N)
y2 = np.ones(N)
y3 = 2 * np.ones(N)
X = np.vstack([X1, X2, X3])
y = np.hstack([y1, y2, y3])

## QDA 모델 생성
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis(store_covariance=True).fit(X, y)

## QDA의 속성
qda.priors_
qda.means_
qda.covariance[0] # [1], [2]

## 확률밀도함수 구하기
rv1 = sp.stats.multivariate_normal(qda.means_[0], qda.covariance_[0])
rv2 = sp.stats.multivariate_normal(qda.means_[1], qda.covariance_[1])
rv3 = sp.stats.multivariate_normal(qda.means_[2], qda.covariance_[2])
rv1.pdf([2, -1]), rv2.pdf([2,-1]), rv3.pdf([2,-1])


```

## 3. 선형판별분석법 (LDA)
선형판별분석법(Linear DiscriminantAnalysis, LDA )각 $Y$클래스에 대한 독립변수 $X$의 조건부확률분포가 **공통된 공분산 행렬**을 가지는 **다변수 정규분포(multivariate Gaussian normal distribution)**라고 가정한다.
$$\Sigma_k = \Sigma\  for\  all\  k$$

정리를 생략하면 결국 사전확률은 아래와 같이 된다.,
$$log P(y = k | x) = w_k^Tx+C_k^n$$

따라서 모든 클래스 k에 대해 위와 같은 식이 성립하므로 클래스 k1과 클래스  k2의 경계선, 즉 **두 클래스에 대한 확률값이 같아지는 $x$위치**를 찾으면 다음과 같다.
$$w^Tx+C=0$$

결국, 판별함수가 $x$에 대한 '선형방정식'이 되고 경계선의 모양이 **'직선'**이 된다.

- **QDA를 LDA로 바꿔도 퍼포먼스가 크게 떨어지지 않는다.**


### 코드 
Scikit-Learn은 선형판별분석법을 위한 `LinearDiscriminantAnalysis` 클래스를 제공한다.

```py
### 위의 샘플데이터를 이용
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=3, solver="svd", store_covariance=True).fit(X, y)

lda.means_
lda.covariance_ # 공분산행렬값은 각 클래스의 중간값 정도가 나온다.

```

## QDA와 LDA 요약
- QDA: 생성모형을 사용하는데 가장 기본이 된다.
- LDA: QDA에서 $\Sigma$값이 다 똑같다고 가정한다.
- NB(나이브베이즈): LDA에서 추가적으로 **조건부 독립**이라고 가정한다. 훨씬 더 간단한 모형이 된다.