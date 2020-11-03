---
title:  "[머신러닝] 비지도학습 - 6. 가우시안 혼합모형과 EM방법"
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

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/18.01%20%EA%B0%80%EC%9A%B0%EC%8B%9C%EC%95%88%20%ED%98%BC%ED%95%A9%EB%AA%A8%ED%98%95%EA%B3%BC%20EM%20%EB%B0%A9%EB%B2%95.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 1. 가우시안 혼합모형
$K$-클래스 카테고리 확률변수 $Z$가 있다고 할 때, 실수값을 출력하는 확률변수 $X$는 확률변수 $Z$의 표본값 $k$에 따라 기댓값 $\mu_k$, 분산 $\Sigma_k$가 달라진다.  

이 때 $p(x)$의 값은 다음과 같다.
$$ p(x) = \sum_Z p(z)p(x\mid z) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k) $$

실수값을 출력하는 확률변수 $X$가 $K$-클래스 카테고리 확률변수 $Z$의 값에 따라 **다른 기댓값과 분산을 가지는 복수의 가우시안 정규분포들**로 이루어진 모형을 **가우시안 혼합 모형**(Gaussian Mixture)이라고 한다.


### 베르누이-가우시안 혼합모형
카테고리가 두 개인 가우시안 혼합모형은 베르누이-가우시안 혼합모형(Bernouilli Gaussian-Mixuture Model)이라고 한다.

![](/assets/images/Graph4_1.JPG)

- 데이터만 보고 **히든 스테이트의 변수**를 찾아내고, **봉우리 각각의 시그마값과 뮤값**을 찾아내야 한다. ($\theta$값과 $z$(hidden)값을 찾아야 된다. ) 

아래는 2개의 카테고리와 2차원 가우시안 정규분포를 가지는 가우시안 혼함모형 데이터의 예이다.

```py
# 가우시안 혼합모형 데이터 생성

from numpy.random import randn

n_samples = 500

mu1 = np.array([0, 0])
mu2 = np.array([-6, 3])
sigma1 = np.array([[0., -0.1], [1.7, .4]])
sigma2 = np.eye(2)

np.random.seed(0)
X = np.r_[1.0 * np.dot(randn(n_samples, 2), sigma1) + mu1,
          0.7 * np.dot(randn(n_samples, 2), sigma2) + mu2]
plt.scatter(X[:, 0], X[:, 1], s=5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("베르누이 가우시안 혼합 모형의 표본")
plt.show()
```

![](/assets/images/Graph4_2.png)

- 여기서 우리는 첫 번째 파라미터 $\theta$를 알아내야 한다. 즉, 중심위치와 공분산 행렬을 찾아낸다.


### 가우시안 혼합모형의 모수 추정

데이터로부터 **가우시안 혼합모형의 모수를 추정**한다는 것은 관측되지 않는 카테고리 분포의 확률분포와 **각각의 카테고리**에서의 가우시안 정규분포 모수를 모두 추정하는 것을 말한다.  
이 때 어려운 점은 확률분포함수가 선형대수방법으로 쉽게 구할 수 없는 복잡한 형태가 된다.  

N개의 데이터에 대한 X의 확률분포에 로그를 취하면
$$  \log p(x) = \sum_{i=1}^N \log \left( \sum_{k=1}^{K} \pi_k \mathcal{N}(x_i\mid \mu_k, \Sigma_k) \right) $$
- 여기서 두 식 모두 미분값이 0이 되는 모수를 쉽게 구하기 힘들다. 여기서 파이($\pi$), 뮤($\mu$), 시그마($\Sigma$)가 'theta($\theta$)'이다.   


데이터 $x_i$가 어떤 카테고리 $z_i$에 속하는지를 알 경우 같은 카테고리에 속하는 데이터만 모아서 **카테고리 확률분포 $\pi_k$**도 알 수 있고, **가우시안 정규분포의 모수 $\mu_k, \Sigma_k$**도 쉽게 구할 수 있다. 하지만 실제로는 데이터 $x_i$가 가지고 있는 **카테고리 값 $z_i$를 알 수가 없기 때문**에 위 확률분포함수를 최대화하는 $\pi_k$와 $\mu_k, \Sigma_k$를 **비선형 최적화**를 통해 구해야 한다.

네트워크 확률모형 관점에서는 확률변수 $Z_i$가 확률변수 $X_i$에 영향을 미치는 단순한 모형이다. 하지만 $i=1,\dots, N$ 인 모든 경우에 대해 반복적으로 영향을 미치므로 아래와 같이 **판넬모형**으로 표현한다.

![](/assets/images/Graph4_3.JPG)

## 2. EM(Expectation-Maximizatoin)
혼합모형의 모수추정에서 중요한 역할을 하는 것 중의 하나가 바로 각 데이터가 어떤 카테고리에 속하는가를 알려주는 조건부 확률 $p(z|x)$값이다. 이 값을 **responsibility**(($z$))라고 한다. 가우시안 혼합모형의 경우 다음과 같이 정리할 수 있다.
$$ 
\pi_{ik} = \dfrac{\pi_k \mathcal{N}(x_i\mid \mu_k, \Sigma_k)}{\sum_{k=1}^K \pi_k \mathcal{N}(x_i\mid \mu_k, \Sigma_k)} 
$$


만약 우리가 모수 $\theta$를 정확하게 안다면 $\$값을 알 수 있다. 이는 likelihood를 미리 구해, 어느 클래스에 속하는지 알아내는 QDA방법과 비슷하다.
이 식은 모수로부터 reponsibility를 추정한다.
$$(\pi_k, \mu_k, \Sigma_k) \;\; \implies \;\; \pi_{i,k}$$

- $\pi_{ik}$는 $i$번째 데이터 $x_i$가 카테고리 $k$에서 만들어졌을 확률을 나타낸다.  

### 로그-결합확률분포함수 최대화
다음으로 **로그-결합확률분포함수**를 최대화해야 한다. 그러기 위해서 위에서 구한 각 변수들로 미분해서 최대화되는 모수값을 구해야한다.    


우선 $\mu_k$로 미분하여 0이 되도록 하는 방정식을 정리하면 다음과 같다.
$$\mu_k = \frac{1}{N_k}\sum^N_{i=1}\pi_{ik}x_i$$

위 식에서 $N_k$는 $k$ 카테고리에 속하는 데이터의 수와 비슷한 의미를 가진다.
$$N_k = \sum^N_{i=1}\pi_{ik}$$

즉 **$\mu_k$**는 $k$ **카테고리에 속하는 데이터의 샘플 평균**(선택된 애들만의 평균)과 같은 의미이다.


다음으로 로그-결합확률분포함수를 $\Sigma_k$로 미분해서 최대화하는 모수값을 구하면 다음과 같다.
$$\Sigma_k = \frac{1}{N_k} \sum_{i=1}^N\pi_{ik}(x_i-\mu_k)(x_i-\mu_k)^T $$

마지막으로 로그-결합확률분포함수를 $\pi_k$로 미분해서 최대화하는 모수값을 구해야 하는데 이 때 카테고리 값의 모수가 가지는 제한 조건으로 인해 Lagrange multiplier(라그랑주 승수)를 추가해야 한다.
$$log P(x) + \lambda \left(\sum_{k=1}^K\pi)k -1 \right)$$

이를 $\pi_k$로 미분해서 0이 되는 값을 찾으면 아래와 같이 된다.
\pi_k = \frac{N_k}{N}
---
위 세가지 식은 모두 **responsibility**로부터 모수를 구한다.
$$ \pi_{ik} \;\; \implies \;\; (\pi_k, \mu_k, \Sigma_k ) $$

원래는 연립방정식의 해를 구하는 방법으로 **responsibility**를 포함하는 모수값을 추정해야 한다. 그러나 만약 식의 형태가 reponsibility를 알고 있다면 모수를 추정하는 것이 간단하도록 만들어져 있기 때문에 **EM(Expectation-Maximization)**이라고 하는 **iterative방법**을 사용하면 연립방적식의 해를 구하는 것보다 더 **쉽게 모수를 추정**할 수 있다.

### EM방법 정의
**EM 방법**은 **모수**와 **responsiblity**를 **번갈아 추정**하며 정확도를 높여가는 방법이다.

- **E step**(Expectation step)에서는 우리가 현재까지 알고 있는 모수($\theta$)가 정확하다고 가정하고 이를 사용하여 각 데이터가 어느 카테고리에 속하는지 즉, responsiblity($z$)를 추정한다.
$$ (\pi_k, \mu_k, \Sigma_k) \;\; \implies \;\; \pi_{ik} $$

* **M step**(Maximization step)에서는 우리가 현재까지 알고 있는 responsibility($z$)가 정확하다고 가정하고 이를 사용하여 모수값($\theta$)을 추정한다.
$$ \pi_{ik} \;\; \implies \;\; (\pi_k, \mu_k, \Sigma_k)  $$

이를 반복하면 모수와 responsibility를 동시에 **점진적으로 개선**할 수 있다.

## 3. 클러스터링
각각의 데이터에 대해 responsibility알게되면 responsibility가 가장 큰 카테고리를 찾아내어 그 데이터가 어떤 카테고리에 속하는지를 알아낼 수 있다. 즉 **클러스터링을 할 수 있다**는 것이다.
$$ k_i = \arg\max_{k} \pi_{ik} $$

- K-Means 클러스터링은 EM방법의 특수한 경우라고 할 수 있다.

### Scikit-Learn의 GaussianMixture 클래스

```py
from sklearn.mixture import GaussianMixture
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# 가우시안 혼합모형 생성
def plot_gaussianmixture(n):
    model = GaussianMixture(n_components=2, init_params='random', random_state=0, tol=1e-9, max_iter=n)
    with ignore_warnings(category=ConvergenceWarning):
        model.fit(X)
    pi = model.predict_proba(X)
    plt.scatter(X[:, 0], X[:, 1], s=50, linewidth=1, edgecolors="b", cmap=plt.cm.binary, c=pi[:, 0])
    plt.title("iteration: {}".format(n))

# 시각화 코드
plt.figure(figsize=(8, 12))
plt.subplot(411)
plot_gaussianmixture(1)
plt.subplot(412)
plot_gaussianmixture(5)
plt.subplot(413)
plot_gaussianmixture(10)
plt.subplot(414)
plot_gaussianmixture(15)
plt.tight_layout()
plt.show()
```

![](/assets/images/Graph4_4.png)

- 시각화 코드를 보면 iteration이 진행될 수록 어떤 데이터가 하얀 state에 속하고, 까만 state에 속하는지 확인할 수 있게 됐다.

## 4. 일반적인 EM알고리즘 
EM 알고리즘은 잠재변수 $z$에 의존하는 확률변수 $x$가 있고 $z$는 관측 불가능하며 $x$만 관측할 수 있는 경우 확률분포 $p(x)$를 추정하는 방법이다. 
다만 네트워크 모형에 의해 조건부확률분포 $p(x\mid z, \theta)$는 모수 $\theta$에 의해 결정되며 그 수식은 알고 있다고 가정한다.

혼합모형의 경우에는 $z$가 이산확률변수이므로 아래와 같은 식이 성립한다.
$$ p(x \mid  \theta) = \sum_z p(x, z \mid  \theta) = \sum_z q(z) p(x\mid z, \theta) $$

- $\theta$를 구하면 x가 나오는 분포를 수식으로 정확하게 설명할 수 있다. 즉, $x$가 어떤 모양을 그리는지 알 수 있는 파라미터이다.
- $p(x \mid \theta)$는 **쌍봉분포**이다.
- $\sum_z q(z) p(x\mid z, \theta)$는 카테고리 분포로 **단봉분포**이다.
- 각각의 단봉분포를 찾아내고, 그 봉우리가 어디에 속하는지를 확인한다.


**EM 알고리즘의 목표**는 주어진 데이터 $x$에 대해  가능도 $p(x \mid  \theta)$를 가장 크게 하는 잠재변수에 대한 확률분포 $q(z)$와 $\theta$를 구하는 것이다.  

우선 다음과 같은 $log p(x)$가 있다고 하자.
$$
\log p(x) = 
\sum_z q(z) \log \left(\dfrac{p(x, z \mid  \theta)}{q(z)}\right) -
\sum_z q(z) \log \left(\dfrac{p(z\mid x,  \theta)}{q(z)}\right)
$$
(증명 생략)

이 식에서 첫 항은 $L(q,\theta)$로 두번째 항은 KL(q \mid p)라고 쓰도록 한다.
따라서 위 식은 아래와 같이 변하게 된다.
$$log p(x) = L(q,\theta) + KL(q \mid p)$$

- $log p(x)$는 우리가 **최대화 하려는 p**이다.
- $L(q, \theta)$는 분포함수 $q(z)$를 입력하면 **수치가 출력되는 범함수**(functional)이다. 
- $KL(q \mid  p)$은 분포함수 $q(z)$와 $p(z\mid x, \theta)$의 차이를 나타내는 **쿨백-라이블러 발산**이다. 쿨백-라이블러 발산은 항상 0과 같거나 크기 때문에 $L(q, \theta)$는 $\log p(x)$의 **하한**(lower bound)이 된다. 그래서 $L$을 ELBO(evidence lower bound)라고도 한다. 또 반대로 이야기하면 $\log p(x)$가 $L(q, \theta)$의 **상한(upper bound)**이라고 할 수도 있다.
$$ \log p(x) \geq L(q, \theta) $$


따라서 $L(q, \theta)$를 최대화하려면 $log p(x)$도 최대화된다. 따라서 EM알고리즘은 $L(q, \theta)$를 최대화하기 위해 **$q$와 $\theta$의 최적값**을 교대로 찾아낸다.

(1) **E 단계**에서는 **$\theta$를** 현재의 값 $\theta_{\text{old}}$으로 **고정**시키고 $L(q_\text{old}, \theta_\text{old})$를 **최대화하는 $q_{\text{new}}$를 찾는다**.  
  
맞게 찾았다면 $L(q_{\text{new}}, \theta_\text{old})$는 상한인 $\log p(x)$와 같아진다. 즉 **쿨백-라이블러 발산은 0이 된다**.
$$  L(q_{\text{new}}, \theta_\text{old}) = \log p(x) $$
$$ KL(q_{\text{new}} \mid  p) = 0 $$
$$ q_{\text{new}} = p(z\mid x, \theta_{\text{old}}) $$


(2) **M 단계**에서는 **$q$**를 현재의 함수 $q_{\text{new}}$로 **고정**시키고 $L(q_{\text{new}}, \theta)$를 **최대화하는 $\theta_{\text{new}}$값을 찾는다**. 최대화를 하였으므로 당연히 $L(q_{\text{new}}, \theta_{\text{new}})$는 **옛날 값보다 커진다.** 
$$ L(q_{\text{new}}, \theta_{\text{new}}) > L(q_{\text{new}}, \theta_{\text{old}}) $$


그리고 동시에 $p(Z\mid X, \theta_{\text{new}})$이 과거의 값 $p(Z\mid X, \theta_{\text{old}})$과 달라졌으므로 $q_{\text{new}}$는 $p(Z\mid X, \theta_{\text{new}})$와 달라진다.
그러면 **쿨백 라이블러 발산의 값도 0보다 커지게 된다**.
$$ q_{\text{new}} \neq p(Z\mid X, \theta_{\text{new}}) $$
$$ KL(q_{\text{new}} \mid  p) > 0 $$

(3) E단계와 M단계를 반복하면서 모수와 responsibility를 동시에 **점진적으로 개선**한다.