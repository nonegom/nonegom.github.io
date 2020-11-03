---
title:  "[머신러닝] 그래프 모형 - 4. 히든 마코프 모형"
excerpt: "그래프 모형 중 히든 마코프 모형"

categories:
  - GraphModel
tags:
  - UnSupervisedLearning
  - 11월
toc: true
toc_sticky: true
toc_label: 페이지 목차
use_math: true
---
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/20.01%20%ED%9E%88%EB%93%A0%20%EB%A7%88%EC%BD%94%ED%94%84%20%EB%AA%A8%ED%98%95.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  


# 히든 마코프 모형
- 히든 마코프 모델에 대해서만 책 한권이 나오기 때문에 이 모델에 대한 이해만 하면 된다. 히든 마코프 모델 음성인식이나, 텍스트 분석에도 사용되는데, 그럴 경우 변수 값을 카테고리 값으로 주면 된다. 

## 1. 독립혼합모형
독립 혼합 모형(independent Mixture Model)은 **연속 확률 변수**이지만 단일한 확률 분포를 가지지 않고 **복수의 연속 확률 분포 중 하나를 확률적으로 선택**하는 모형을 말한다. 이때 연속 확률 분포의 선택은 **독립적인 이산 확률분포를 사용**한다.
$$
\begin{eqnarray*}
p(x)
&=& \sum_{i=1}^m P(C=i) \cdot P(X=x \mid C=i) \\
&=& \sum_{i=1}^m \delta_i p_i(x)
\end{eqnarray*}
$$

꼭 이진분포가 아니더라도 봉우리의 개수에 따라서 **변수의 개수가 달라지는 카테고리 변수를 사용**하면 된다.

- $p(x)$ : 전체 Independent Mixuture 분포
- $p_i(x)$ : Independent Mixuture의 각 성분(component)이 되는 개별적인 연속 확률분포
- $\delta_i$ : mixing parameter. 특정시간에 대해 모든 성분 중 특정한 $p_i(x)$가 선택될 확률. 이산 확률 분포
- $\sum\delta_i = 1$ : mixing parameter에 대한 확률 제한 조건

### 독립 혼합 모형의 예: 베르누이-정규 혼합 모형
 베르누이-정규 혼합 모형은 베르누이 확률 변수의 값에 따라서 두 개의 서로 다른 연속 정규 분포 중 하나를 선택하는 확률 분포이다. 즉, **이전 확률변수의 확률이 다음 확률변수**에 영향을 준다. 

## 2. 마코프 체인
마코프 체인(Markov Chain)은 다음과 같은 마코프 특성을 가지는 이산시간 확률 프로세스를 말한다. 즉, 어떠한 국면이 지속되는 상황을 말한다고 할 수 있다.
$$
P(C_{t+1} \mid C_t, \cdots, C_1) = P(C_{t+1} \mid C_t)
$$

## 3. 히든 마코프 모형
히든 마코프 모형(Hidden Markov Model)은 **독립 혼합 모형**에서 연속 확률 분포를 선택하는 **이산 확률 과정 $C_t$가 마코프 체인**이고 **연속 확률 분포$X_t$ 가 그 시점의 이산 확률 과정의 값에만 의존**하는 모형을 말한다. 그러나 연속 확률 분포의 값 $X_t$  만 측정 가능하고 이산 확률 과정의 값 $C_t$ 는 측정할 수 없다.
마코프 체인 ($C_t$)
$$
P(C_t \mid |C_{t-1}, \cdots, C_1) = P(C_t \mid C_{t-1})
$$

$C_t$값에 의존하는 연속확률분포 $X_t$
$$
P(X_t \mid X_t, \cdots, X_1, C_t, \cdots, C_1) = P(X_t \mid C_t)
$$


### hmmlearn 패키지
Python에는 다양한 HMM(히든 마코프 모델) 시뮬레이션 및 추정용 패키지가 제공된다. 여기에서는 그 중 하나인 hmmlearn 패키지를 사용해보자.

`GaussianHMM` 클래스를 사용하면 카테고리-다변수 정규분포 혼합 모형을 시뮬레이션 할 수 있다. 다음과 같은 속성을 설정 가능

- `startprob_`: 초기 확률 벡터
- `transmat_`: 전이 확률 행렬 
    - (전이확률: 특정 시간 t동안 특정한 한 상태에서 다른 상태로 전이할 확률 / 확률행렬: 모든 상태 조합에 대한 전이 확률)
- `means_`: 정규 분포의 기댓값 벡터
- `covars_`: 정규 분포의 공분산 행렬

가장 간단한 '베르누이-정규분포'예제

```py
from hmmlearn import hmm
np.random.seed(3)

model = hmm.GaussianHMM(n_components=2, covariance_type="diag")
model.startprob_ = np.array([0.9, 0.1])
model.transmat_ = np.array([[0.95, 0.05], [0.15, 0.85]])
model.means_ = np.array([[1.0], [-3.0]])
model.covars_ = np.array([[15.0], [40.0]])
X, Z = model.sample(500)

# 시각화 코드
plt.subplot(311)
plt.plot(X)
plt.title("random variable")

plt.subplot(312)
plt.plot(Z)
plt.title("discrete state")

plt.subplot(313)
plt.plot((1 + 0.01*X).cumprod())
plt.title("X cumulated")
plt.tight_layout()
plt.show()
```

![](assets/images/Graph5_1.png)

```py
sns.distplot(X)
plt.show()
```

![](assets/images/Graph5_2.png)


## 4. 디코드

관측된 히든 마코프 모형의 **연속 확률 변수 값**으로부터 **내부의 이산 확률 변수 값을 추정**하는 과정을 디코드(decode)라고 한다.  디코드 알고리즘 중에서는 `Viterbi`알고리즘이 가장 많이 사용된다.

`hmmlearn`패키지의 `HMM`클래스들은 모형 추정을 위한 `fit`메서드와 디코딩을 위한 `decode`메서드를 제공한다. 아래는 위 시뮬레이션 결과를 디코딩한 예시이다.

```py
model2 = hmm.GaussianHMM(n_components=2, n_iter=len(X)).fit(X)
model2

Z_hat = model2.decode(X)[1]
X_cum = (1 + 0.01*X).cumprod()
X_cum_hat = X_cum.copy()
X_cum_hat[Z_hat == 0] = np.nan

plt.subplot(211)
plt.plot(X_cum, lw=5)
plt.plot(X_cum_hat, 'r-', lw=5)
plt.title("X cumulated")

plt.subplot(212)
plt.plot(Z, 'bo-')
plt.plot(Z_hat, 'ro-')
plt.title("discrete state")
plt.tight_layout()
plt.show()
```

![](assets/images/Graph5_3.png)

- 컴퓨터가 예측한 값이 '빨간색'이다. 이를 보면 '파란색'(실제)과 비슷한 양상을 보여주고 있음을 확인할 수 있다.