---
title:  "[머신러닝] 비지도학습 - 7. 몬테카를로 베이지안 분석"
excerpt: "가우시안 혼합모형과 EM방법을 통한 확률분포 추정방법"

categories:
  - UnSupervisedLearning
tags:
  - UnSupervisedLearning
  - 11월
toc: true
toc_sticky: true
toc_label: 페이지 목차
use_math: true
---

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/19.01%20%EB%AA%AC%ED%85%8C%EC%B9%B4%EB%A5%BC%EB%A1%9C%20%EB%B2%A0%EC%9D%B4%EC%A7%80%EC%95%88%20%EB%B6%84%EC%84%9D.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 0. 몬테카를로 베이지안 분석
몬테카를로 방법은 사전분포가 켤레사전분포(conjugate prior)가 아닐 때에도 **베타 분포($Beta(\alpha, \beta)$)**를 구할 수 있게 계산가능한 수식들을 만들어서 확률분포를 계산하고자 하는 것이다. 최종적인 목적은 **베이지안 추정**을 하고자 하는 것이다.

- 베이지안(Baysian)이란 본래 사전분포가 어떻게 되든 업데이트를 통해 모수를 찾아가는 방법이다. 하지만 모형에 대해 어느 정도 아는 바가 있다면 적절한 사전분포를 사용함으로써 수학적 계산을 간단히 하고 결과를 이해하기 쉽다.

하지만 **몬테카를로 방법**은 사전분포가 어떻든 샘플을 이용해 확률을 그리는 방법이다. 즉 PDF를 알고 있으면 그 PDF와 유사한 히스토그램을 가지는 랜덤샘플을 만드는 것이다.

- 단, 계산은 알 수 있지만 그래프를 그리드 서치로 그릴 수는 없는데 왜냐하면 변수가 너무 많아지면 그리기 힘들어지기 때문이다. 하지만 일부 샘플을 뽑아서 그리면 일부일지라도 비교적 정확한 모형을 그릴 수 있다.

## 1. 표본생성
확률변수의 분포함수 $p(X)$를 알고 있다고 할 때, 이 분포함수를 따르는 표본을 생성하는 방법을 알아보도록 하자.

- 균일분포
- 역변환
- rejection sampling
- importance sampling

### 균일분포
---
파이썬의 난수 생성기(Radom Number Generator)는 $2^{19937}-1$인 **MT19937 알고리즘을 사용**한다. 이 알고리즘에서 생성되는 값은 사실 정확한 난수가 아니라 $2^{19937}-1$ 주기로 반복되는 결정론적 수열이므로 유사 난수 생성기(Pseudo Radom Number Generator)이다. 이 값을 주기로 나누어 **0부터 1사이의 부동소수점을 출력하는 균일분포(uniform distribution) 표본 생성**에 사용한다.  
(위 난수 생성기를 쓰는 이유는 균일하게 값이 나오는 알고리즘이 필요했기 때문이다.)

### 역변환
---
확률분포함수가 **수식으로 주어지는 기본적인 확률분포들**의 경우 역변환(inverse transform)방법을 사용해 **표본을 생성**할 수 있다.

위에서 **균일분포함수**는 컴퓨터로 생성할 수 있다고 했다. 이 균일분포에서 생성된 **$x$값을 임의의 실수로 변환하는 단조증가함수 $f(x)$**를 생각해보자.
이 함수를 적용한 값을 $y$라고 하고 **$y$를 표본으로 가지는 확률변수를 $Y$**라고 하면 $Y$의 **누적분포함수는 $f$의 역함수인 $f^{-1}$**이 된다.
$$ F_Y(y) = f^{-1}(x) $$

- 사이파이에서 `rvs`메소드는 위와 같은 방법대로 구해진다.

![](/assets/images/UnSupervised5_1.JPG)

반대로 우리가 원하는 **확률분포함수 $p(Y)$**가 있다면 이 함수를 **적분한 함수의 역함수**를 **균일분포 표본에 적용**하면 **우리가 원하는 확률분포**를 가지게 된다.
$$ x = h(y) = \int_{-\infty}^{y} p(u)du $$
$$ y = h^{-1}(x) $$

### Rejection Sampling
---
우리가 원하는 분포함수가 적분 및 역함수를 구하기 쉽다면(수식으로 주어짐) 역변환 방법을 쓸 수 있지만, 그렇지 않을 경우 **Rejection Sampling**방법을 쓸 수 있다.  
Rejection Sampling에서는 목표확률분포 $p(x)$와 유사하지만 표본생성이 쉬운 유사 확률분포 $q(x)$를 사용한다.

- $p(x)$: 샘플링하고자 하는 **목표 확률분포**
- $q(x)$: 샘플링 가능한 **유사 확률분포**

방법은 간단한데, 우선 유사 확률분포 $q(x)$를 생성한 후 $p(z) / k\ q(x)$의 확률로 이 표본을 **채택할지 아니면 버릴지를 결정**한다. 이떄 **$k$**는 $kq(x) \geq p(x)$$가 되도록 하는 **스케일링 상수**이다.   
즉, 데이터를 유사확률에 맞춰 뽑고 기존 확률의 비율에 맞춰 데이터를 버리는 방법이다.

#### - 코드 예시

예를 들어 $a=2, b=6$인 **베타 분포**의 표본을 만들고 싶은데 지금 생성할 수 있는 확률분포는 **정규 분포**밖에 없다고 하자. 그러면 일단 정규분포의 표본을 생성한다. 그리고 모수가 $p(z)/kq(z)$인 베르누이 확률분포의 표본값을 사용하여 표본을 버릴지 채택할 지 결정한다.  
아래 그림에서 위쪽은 **처음 유사분포로 생성한 표본**이고 오른쪽은 **채택된 표본(rejection sampling)**을 보이고 있다.

```py
# 모수값 설정
a, b = 2, 6
rv_p = sp.stats.beta(a, b)
rv_q = sp.stats.norm(loc=0.5, scale=0.5)
k = 5

# 유사표본 생성
np.random.seed(0)
x_q0 = rv_q.rvs(int(1e4))
x_q = x_q0[(x_q0 >= 0) & (x_q0 <= 1)] # 0에서 1사이 데이터

# rejection sampling
crits = rv_p.pdf(x_q) / (rv_q.pdf(x_q) * k)
coins = np.random.rand(len(x_q))
x_p = x_q[coins < crits]

# 시각화 코드
plt.subplot(211)
sns.distplot(x_q, kde=False)
plt.title("유사 분포의 표본")

plt.subplot(212)
sns.distplot(x_p, kde=False)
plt.title("rejection sampling 방법으로 걸러낸 후의 표본")
plt.tight_layout()
plt.show()
```

![](/assets/images/UnSupervised5_2.png)

```py
plt.subplot(211)
xx = np.linspace(0, 1, 100)
plt.plot(xx, rv_p.pdf(xx), 'r-', label="목표 분포: 베타분포")
plt.plot(xx, rv_q.pdf(xx) * k, 'g:', label="유사 분포: 정규분포 N(0.5, 0.25)")
plt.legend()
plt.title("목표 분포와 유사 분포")

plt.subplot(212)
y = np.random.rand(len(x_q)) * rv_q.pdf(x_q)
plt.plot(x_q, y, 'bs', ms=1, label="버려진 표본")
ids = coins < crits 
plt.plot(x_q[ids], y[ids], 'ro', ms=4, label="채택된 표본")
plt.legend()
plt.title("버려진 표본과 채택된 표본")
plt.tight_layout()
plt.show()
```

> 이해를 위해 실제로 유사 분포에 의해 만들어진 표본을 그리고 채택된 것은 크게, 버려진 것은 작게 표시했다. 

- **Rejection Sampling**의 장점은 간단하지만, 힘들게 뽑은 데이터를 버리는 양이 많다는 것이다. 특히 차원이 커질 수록 버리는 데이터의 양이 많아지게 된다. 

#### - 기댓값 추정
확률분포의 표본을 생성하는 이유 중 하나는 표본을 이용해 그 **확률분포의 기댓값을 추정**할 수 있기 때문이다. 우리가 관심을 가지는 확률분포 $p(X)$에 대해 기댓값을 구하고 싶다면 N개의 표본 데이터 ${x_1, \dots, x_N}$이 존재한다면 **몬테카를로 적분**을 이용해 기댓값을 추정할 수 있다.
- 기댓값
$$
\text{E}[f(X)] = \int f(x)p(x)dx
$$  
- 몬테카를로 적분
$$
\text{E}[f(X)] \approx \dfrac{1}{N} \sum_{i=1}^N f(x_i)
$$

앞 예제에서 $a =2, b=6$인 베타 분포의 기댓값은 1/4이다. 하지만 **표본 데이터를 몬테카를로 적분**해도 유사한 값을 얻을 수 있다.
$$
\frac{a}{a+b} = \frac{1}{4}
$$

```py
x_p.mean()
# > 0.25452461685193456
```

### Importance Sampling
---
만약 **기댓값을 계산**하고자 하는 것이 표본을 생성하는 유일한 목적이라면 **표본 생성**과 기댓값 계산을 위한 **몬테카를로 적분을 하나로 합친** importance sampling을 사용할 수 있다. rejection sampling에서와 같이 $kq > p$인 **유사 분포 $q$의 표본을 생성**하고 다음 식을 이용하여 **직접 기댓값을 계산**한다.

$$
\begin{eqnarray}
\text{E}[f(X)] 
&=& \int f(x)p(x)dx  \\
&=& \int f(x)\dfrac{p(x)}{q(x)} q(x) dx  \\
&\approx & \dfrac{1}{N} \sum_{i=1}^N f(x_i)\dfrac{p(x_i)}{q(x_i)}  \\
\end{eqnarray}
$$

- 평균을 구하는 식으로 간단하게 변하게 된다.

이 식에서 $\dfrac{p(x_i)}{q(x_i)}$은 표본에 대한 **가중치 역할**을 하므로 **importance**라고 한다. rejection으로 인해 버리는 표본이 없기 때문에 더 효율적인 방법이라고 할 수 있다.

코드를 보면 **rejection sample**과정에서 버린 표본의 수는 전체의 80%에 해당하지만, **importance sampling** 방법을 사용하면 버리는 표본없이 **바로 기댓값을 구할 수 있다**.

```py
# rejection sample
len(x_q0), len(x_p)
## > (10000, 2038)

# importance sampling (f(x) * p(x) / q(x))
np.mean(x_q0 * rv_p.pdf(x_q0) / rv_q.pdf(x_q0))
```

## 2. 마코프 체인
상태값이 $K$개의 유한한 값만 가질 수 있는 **이산 상태(discrete-state) 시계열 확률과정**에서 시간 $t$의 값 $x_t$의 **확률분포 $p(x_t)$**가 시간 $t-1$의 값의 **확률분포 $p(x_{t-1})$**과 **조건부확률분포 $p(x_t|x_{t-1})$에만 의존**하면 이 시계열 확률과정을 **마코프 체인**(Markon chain)이라고 한다. 선형 체인 모양의 마코프 네트워크(linear chain Markov network)와는 전혀 다른 뜻이라는 점에 주의하라.

**이산 상태 마코프 체인**에 **전체 확률의 법칙을 적용**하면 아래 식이 성립한다.
$$ p(x_{t}) = \sum_{x_{t-1}} p(x_t, x_{t-1}) = \sum_{x_{t-1}} p(x_t | x_{t-1}) p(x_{t-1}) $$

$p(x_t)$가 카테고리 분포이므로 행 벡터(row vector) $p_t$로 표현하면 위 식은 아래와 같은 **행렬식으로 표현**할 수 있다.
$$ p_t = p_{t-1} T  $$

이 식에서 **조건부 확률을 표현**하는 $K\times K$ 행렬 $T$는 **전이행렬(transition matrix)**라고 한다.
$$ T_{ij} = P(x_t =j \,|\, x_{t-1}=i) $$

**전이행렬이 대칭행렬**인 **마코프 체인**을 reversible Markov chain이라고 하거나 detailed balance condition을 만족한다고 한다. 이러한 마코프 체인은 시간($t$)이 흘러감에 따라 초기 조건 ($p_0$)과 관계없이 **항상 같은 분포 $p_{\infty}$로 수렴한다**는 것이 증명되어 있다.
$$ p_{t'} = p_{t'+1} = p_{t'+2} = p_{t'+3} = \cdots = p_{\infty} $$

위 식에서 $t'$는 수렴상태에 도달한 이후의 시간을 뜻한다. 따라서 **수렴상태에 도달한 후**에는 $\{x_{t'}, x_{t'+1}, x_{t'+1}, \ldots, x_{t'+N} \}$ 표본은 모두 같은(identical) 확률분포에서 나온 값이 된다. 이 **표본 집합** $\{x_{t'}, x_{t'+1}, x_{t'+1}, \ldots, x_{t'+N} \}$을 **trace**라고 한다.

### MCMC
MCMC(Markov Chain Monte Carlo) 방법은 rejection sampling이나 importance sampling과 달리 **마코프 체인을 이용하는 표본 생성 방법**이다. 마코프 체인의 수렴분포가 원하는 분포 $p(x)$가 되도록 하는 **마코프 체인을 만들고** 이 마코프 체인을 **$t'$시간 이상 가동**하면 그 다음부터는 원하는 분포의 표본을 얻을 수 이다.

### 메트로폴리스 해이스팅스 방법
메트로폴리스 해이스팅스 표본 생성법은 MCMC방법의 일종이다. rejection sampling과 비슷하지만 계산용 분포 $q$로 무조건부 분포 $q(x)$가 아니라 조건부 분포 $q(X^* \mid x_t)$를 사용한다. $x_t$를 기댓값으로 하는 **가우시안 정규분포를 많이 사용**한다.
$$ q(x^{\ast} \mid x_t) = \mathcal{N}(x^{\ast} \mid x_t, \sigma^2) $$

**표본 생성 방법**은 다음과 같다.

1. $t=0$이면 무작위로 $x_t$ 생성. 
2. 표본 생성이 가능한 $q(x^{\ast} \mid x_t=x_t)$ 분포로부터 표본 $x^{\ast}$을 생성한다.
3. 다음 확률에 따라 $x^{\ast}$를 $x_{t+1}$로 선택한다. 이를 메트로폴리스 해이스팅스 기준이라고 한다.
$$ p = \min \left( 1, \dfrac{p(x^{\ast}) q(x_t \mid x^{\ast})}{p(x_t) q(x^{\ast} \mid x_t)} \right)$$
만약 선택되지 않으면(rejection) 새 값을 버리고 한 단계 전의 과거 값을 다시 사용한다. $x_{t+1} = x_{t}$
4. 충분한 수의 샘플이 모일때까지 위 **2 ~3 과정을 반복**한다.


만약 계산용 분포 $q(x^{\ast} \mid x_t)$가 가우시안 정규분포이면 조건 $x_t$과 결과 $x^{\ast}$가 바뀌어도 확률값은 같다. 
$$ q(x^{\ast} \mid x_t) = q(x_t \mid x^{\ast}) $$

이 경우에는 기준확률은 아래와 같이 된다. 이를 **메트로폴리스 기준**이라고 한다. 
$$ p = \min \left( 1, \dfrac{p(x^{\ast})}{p(x_t)} \right)$$

메트로폴리스 기준을 따르면  $p(x^{\ast})$가 $p(x_t)$보다 커지려고 노력한다.

- MCMC 방법에는 이외에도 `Hamilton Monte Carlo`나 `NUTS(No-U-Turn Sampler)` 등이 있다.

## 3. PyMc3
MCMC 표본 생성과 이를 이용한 베이지안 추정을 하기 위한 파이썬 패키지이다. `Theano`패키지를 기반으로 하며, 심볼릭 미분과 GPU 사용으로 빠른 계산이 가능하다.

```py
# MKL 라이브러리가 설치되어 있어야 한다.
# 우선 실행 코드
import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import pymc3 as pm
```

### 대략적인 사용법

1. `Model` 클래스를 생성한다. 이 클래스 인스턴스가 theano의 function 혹은 텐서플로우의 세션 역할을 한다.
2. 원하는 심볼릭 확률변수를 생성한다.   
3. 샘플러 인스턴스를 생성한다. 다음 샘플러를 포함한 다양한 방법을 지원한다.
   - `Metropolis`, `HamiltonianMC`, `NUTS`
4. `sample` 명령으로 표본 데이터를 생성한다. 데이터는 **Trace 객체형태**로 반환하고 `traceplot` 명령으로 시각화할 수 있다.

### 예제 코드

메트로폴리스 해이스팅스 방법으로 기댓값과 분산이 아래와 같은 **2차원 가우시안 표본**을 생성한다.
$$ 
\mu = \begin{bmatrix} 1 \\ -1 \end{bmatrix}, \;\;
\Sigma = \begin{bmatrix} 1 & 1.5 \\ 1.5 & 4 \end{bmatrix}
$$

```py
# 기댓값과 분산
cov = np.array([[1., 1.5], [1.5, 4]])
mu = np.array([1, -1])

# 모델 클래스 생성
with pm.Model() as model:
    x = pm.MvNormal('x', mu=mu, cov=cov, shape=(1, 2))  
    step = pm.Metropolis()                    # 샘플러 인스턴스 생성
    trace = pm.sample(1000, step)             # 표본 데이터 생성

import warnings
warnings.simplefilter("ignore")

pm.traceplot(trace)
plt.show()
```

![](/assets/images/Graph5_4.jpg)

```py
plt.scatter(trace['x'][:, 0, 0], trace['x'][:, 0, 1])
plt.show()
```

![](/assets/images/Graph5_5.png)

- 1하고 -1 근처에서 값이 비교적 많이 나오는 것을 확인할 수 있다.

### MCMC를 사용한 베이지안 추정
$$ P(\theta \mid x_{1},\ldots,x_{N}) \propto P(x_{1},\ldots,x_{N} \mid \theta)  P(\theta) $$

- $ P(\theta) $: Beta (베타분포)
- $ P(x_{1},\ldots,x_{N} \mid \theta) $: Binomial (이항분포)

```py
## 0. 변수 생성
#앞면(1)이 나올 확룰이 0.7인 동전이 있다고 가정
theta0 = 0.7
np.random.seed(0)
x_data1 = sp.stats.bernoulli(theta0).rvs(10) # 베르누이 분포로 랜덤 데이터 10개 생성
x_data1
# > array([1, 0, 1, 1, 1, 1, 1, 0, 0, 1])

## 1. 모델 생성
with pm.Model() as model:
    theta = pm.Beta('theta', alpha=1, beta=1)
    x = pm.Bernoulli('x', p=theta, observed=x_data1)
    start = pm.find_MAP()
    step = pm.NUTS()
    trace1 = pm.sample(2000, step=step, start=start)

## 2. traceplot 생성
pm.traceplot(3)
plt.show()
```

![](/assets/images/Graph5_5.JPG)

```py
## 3. posterior
pm.plot_posterior(trace1)
plt.xlim(0, 1)
plt.show()
```

![](/assets/images/Graph5_6.JPG)

```py
## 4. summary
pm.summary(trace1)
```

![](/assets/images/Graph5_7.JPG)


> 데이터의 개수가 늘어나면 성능도 더 좋아진다. (데이터의 개수를 늘려서 진행)

```py
## 0. 랜덤 데이터 생성 (10 -> 500)
np.random.seed(0)
x_data2 = sp.stats.bernoulli(theta0).rvs(500)

## 1. 모델 생성
with pm.Model() as model:
    theta = pm.Beta('theta', alpha=1, beta=1)
    x = pm.Bernoulli('x', p=theta, observed=x_data1)
    start = pm.find_MAP()
    step = pm.NUTS()
    trace1 = pm.sample(1000, step=step, start=start)  # 2000 -> 1000

## 2. traceplot
pm.traceplot(trace2)
plt.show()
```

![](/assets/images/Graph5_8.JPG)

```py
## 3. posterior
pm.plot_posterior(trace2)
plt.xlim(0, 1)
plt.show()
```

![](/assets/images/Graph5_9.PNG)

```py
## 4. summary
pm.summary(trace2)
```

![](/assets/images/Graph5_10.JPG)

### MCMC를 사용한 베이지안 선형 회귀
```py
from sklearn.datasets import make_regression
# 샘플데이터 생성
x, y_data, coef = make_regression(
n_samples=100, n_features=1, bias=0, noise=20, coef=True, random_state=1)
x = x.flatten()
coef
# > array(80.71051956)

# 시각화 코드
plt.scatter(x, y_data)
plt.show()
```

![](/assets/images/Graph5_11.png)

```py
# 모델 생성
with pm.Model() as m:
    w = pm.Normal('w', mu=0, sd=50) # w가 0 ~ 50 근처이다. 
    b = pm.Normal('b', mu=0, sd=50) # b값도 0 ~50 정도이다.
    esd = pm.HalfCauchy('esd', 5)   # esd는 disturbance의 크기이다. / HalfCauchy(양수)
    y = pm.Normal('y', mu=w * x + b, sd=esd, observed=y_data) 
    # 위의 뮤값과 분산의 값을 이용해 새로 정규분포(y)를 만든다.

    start = pm.find_MAP()
    step = pm.NUTS()
    trace3 = pm.sample(10000, step=step, start=start)

## traceplot
pm.traceplot(trace3)
plt.show()

```
![](/assets/images/Graph5_12.png)

```py
pm.summary(trace3)
```

![](/assets/images/Graph5_13.jpg){: .align-center}


**참고)**  

누가 봐도 이상한 값이 나왔을 경우 **Hierachical Linear Model**(계층적 선형모델)을 사용한다. 어느 정도의 값이 나온다는 것을 가정하고, 사전확률을 주는 방법이다.  

(주식에서)Pair Trading 같은 것을 하기 위해서는 **기울기가 같아야 한다**(양이거나 음이거나). 그런데 만약 실제 데이터의 기울기가 음수가 되지 않는다는 것을 알고 있다면, 기울기가 음수가 되지 않는다는 사전확률을 준다. Linear Regression할 때 과거에 나왔던 **+**데이터들을 토대로 Regularization(정규화)를 해주면서, pairtrading을 이용할 수 있는 정상적인 기울기 값이 나올 수 있게 한다.