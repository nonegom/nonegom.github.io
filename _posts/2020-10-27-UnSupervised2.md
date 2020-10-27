---
title:  "[머신러닝] 비지도학습 - 2. K-평균 군집화"
excerpt: "군집화 중 K-평균 군집화 및 K-평균++, 미니배치 k-평균 군집화"

categories:
  - ML_Unsrpervised
tags:
  - UnSupervisedLearning
  - 10월
toc: true
toc_sticky: true
toc_label: 페이지 목차
use_math: true
---
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/16.02%20K-%ED%8F%89%EA%B7%A0%20%EA%B5%B0%EC%A7%91%ED%99%94.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 1. K-평균 군집화
가장 단순하고 빠른 군집화 방법의 하나이다. 목적함수 값이 최소화될 때까지 군집의 중심위치와 각 데이터가 소속될 군집을 반복해서 찾는다. 이를 **관성(inertia)**이라고 한다.
$$J = \sum_{k=1}^K\sum_{i \in C_k} d(x_i, \mu_k)$$


> **중심위치**는 무리의 대장이라고 생각할 수 있고, J는 거리를 의미한다. 중심위치를 통해 데이터들은 각자 자신의 위치를 알 수 있고, 이를 통해 거리를 계산한다. 그 거리들의 합이 $J$이다.


이 식에서 $K$는 군집의 개수이고 $C_k$는 k번째 군집에 속하는 데이터의 집합, $\mu_k$는 $k$번째 군집의 **중심위치(centroid)**, $d$는 $x_i, mu_k$ **두 데이터 사이의 거리** 혹은 비유사도(dissimilarity)로 정의한다. 만약 유클리드 거리를 사용하면 다음과 같이 표현할 수도 있다.
$$d(x_i, \mu_k) = \|x_i - \mu_k \|^2$$
$$J = \sum_{i=1}^N min_{\mu_j \in C} (\|x_i - \mu_k \|^2)$$


### 세부 알고리즘
1. 임의의 중심위치 $\mu_k(k=1, \dots, K)$를 고른다. 보통 데이터 표본 중에서 $K$개를 선택한다.
2. 모든 데이터 $x_i(i=1, \dots, N)$에서 각각의 중심위치 $\mu_k$까지의 거리를 계산한다.
3. 각 데이터에서 가장 가까운 중심위치를 선택하여 각 데이터가 속하는 군집을 정한다.
4. 각 군집에 대해 중심위치 $\mu_k$를 다시 계산한다.
5. 2 ~ 4를 반복한다.

**K-평균 군집화**(K-means cluster)란 명칭은 각 군집의 중심위치를 구할 때 해당 군집에 속하는 데이터의 평균(mean)값을 사용하는데서 유래했다. 만약 평균대신 중앙값(median)을 사용하면 K-중앙값(k-Median)군집화라고 한다.


Scikit-Learn의 `cluster`서브 패키지는 `KMeans`클래스를 제공한다. 다음과 같은 인수를 받을 수 있다.
- `n_clusters`: 군집의 갯수
- `init`: 초기화 방법. "random"이면 무작위, "k-means++"이면 **K-평균++ 방법**. 또는 각 데이터의 군집 라벨.
- `n_init`: 초기 중심위치 시도 횟수. 디폴트는 10이고 10개의 무작위 중심위치 목록 중 가장 좋은 값을 선택한다.
- `max_iter`: 최대 반복 횟수.
- `random_state`: 시드값.

### 예제 코드
`make_blobs`명령으로 만든 데이터를 2개로 K-평균 군집화하는 과정이다. 각각의 그림은 위 세부알고리즘 중 3단계(군집을 정하는 단계)에서 멈춘 것이다. **마커**의 모양은 소속된 군집(▲, ▼)을 나타내고 크기가 큰 마커가 해당 군집의 **중심위치**이다. 각 단계에서 중심위치는 전단계 군집의 평균으로 다시 계산된다.

```py
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 데이터 생성, 종속변수 할당 X
X, _ = make_blobs(n_samples=20, random_state=4)

def plot_KMeans(n):
    # K-평균 군집화 생성 (max_iter로 반복횟수 설정)
    model = KMeans(n_clusters=2, init="random", n_init=1, max_iter=n, random_state=6).fit(X)
    c0, c1 = model.cluster_centers_
    plt.scatter(X[model.labels_ == 0, 0], X[model.labels_ == 0, 1], marker='v', facecolor='r', edgecolors='k')
    plt.scatter(X[model.labels_ == 1, 0], X[model.labels_ == 1, 1], marker='^', facecolor='y', edgecolors='k')
    plt.scatter(c0[0], c0[1], marker='v', c="r", s=200)
    plt.scatter(c1[0], c1[1], marker='^', c="y", s=200)
    plt.grid(False)
    plt.title("반복횟수={}, 관성={:5.2f}".format(n, -model.score(X)))


# 시각화 코드
plt.figure(figsize=(8, 8))
plt.subplot(321)
plot_KMeans(1)
plt.subplot(322)
plot_KMeans(2)
plt.subplot(323)
plot_KMeans(3)
plt.subplot(324)
plot_KMeans(4)
plt.tight_layout()
plt.show()
```

![](/assets/images/UnSupervised2_0.PNG)

반복횟수가 증가할 수록 관성이 줄어듬을 알 수 있다. K-평균 군집화는 항상 수렴하지만 최종 군집화 결과가 **전역 최적점**이라는 보장은 없다. 군집화 결과는 **초기 중심 위치**에 따라 달라질 수 있다.

## 2. K-평균++ 알고리즘
K-평균++ 알고리즘은 초기 중심위치를 설정하기 위한 알고리즘이다. 아래와 같은 방법을 통해 **되도록 멀리 떨어진 중심위치 집합**을 찾아낸다.

1. 중심위치를 저장할 집합을 준비($M$)한다.
2. 일단 하나의 중심위치($\mu_0$)를 랜덤하게 선택하여 집합($M$)에 넣는다.
3. ($M$)에 속하지 않는 모든 표본 $x_i$에 대해 거리 $d(M,x_i)$를 계산한다. $d(M,x_i)$는 $M$안의 모든 샘플 $\mu_k$에 대해 $d(\mu_k,x_i)$를 계산하여 가장 작은 값을 선택한다.
4. $d(M,x_i)$에 비례한 확률로 그 다음 중심위치 $\mu$를 선택한다.
5. $K$개의 중심위치를 선택할 때까지 위를 반복한다.
6. **K-평균 방법** 사용한다.


### 예제 코드
K-평균++ 방법을 사용하여 MNIST Digit이미지 데이터(0 ~ 9 / 10개)를 군집화한 결과이다. 각 군집에서 10개씩의 데이터만 표시한다. 

```py
# 데이터 로드
from sklearn.datasets import load_digits
digits = load_digits()

# K-평균++ 모델 생성(cluster의 개수 10개로 설정)
from sklearn.cluster import KMeans
model = KMeans(init="k-means++", n_clusters=10, random_state=0)
model.fit(digits.data)
y_pred = model.labels_

# 시각화를 위한 함수
def show_digits(images, labels):
    f = plt.figure(figsize=(8, 2))
    i = 0
    while (i < 10 and i < images.shape[0]):
        ax = f.add_subplot(1, 10, i + 1)
        ax.imshow(images[i], cmap=plt.cm.bone)
        ax.grid(False)
        ax.set_title(labels[i])
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        plt.tight_layout()
        i += 1
        
def show_cluster(images, y_pred, cluster_number):
    images = images[y_pred == cluster_number]
    y_pred = y_pred[y_pred == cluster_number]
    show_digits(images, y_pred)
    

for i in range(5): # 10개까지 있지만 5개만 출력
    show_cluster(digits.images, y_pred, i)
```

![](/assets/images/UnSupervised2_1.jpg)

> 이미지의 제목에 있는 숫자는 **군집번호**이기에 **원래의 숫자 번호**와 일치하지 않는다. 그러나 이를 예측문제라고 가정하고 분류결과 행렬을 만들어 군집화 결과의 성능을 측정해볼 수 있다. (ARI, AMI, 실루엣 계수)

```py
# 1. 행렬 생성

from sklearn.metrics import confusion_matrix
confusion_matrix(digits.target, y_pred)
""" > 출력값
array([[  1,   0,   0,   0,   0, 177,   0,   0,   0,   0],
       [  0,   1,  55,  24,   0,   0,   0,   2,   1,  99],
       [  0,  13,   2, 148,   2,   1,   3,   0,   0,   8],
       [  0, 155,   0,   1,  11,   0,   7,   0,   2,   7],
       [163,   0,   7,   0,   0,   0,   7,   0,   0,   4],
       [  2,   1,   0,   0,  42,   0,   0,   1, 136,   0],
       [  0,   0,   1,   0,   0,   1,   0, 177,   0,   2],
       [  0,   0,   0,   0,   0,   0, 177,   0,   0,   2],
       [  0,   4,   6,   3,  48,   0,   5,   2,   4, 102],
       [  0,   6,  20,   0, 139,   0,   7,   0,   6,   2]])
"""

# 2. 성능 측정
from sklearn.metrics.cluster import adjusted_mutual_info_score, 
adjusted_rand_score, silhouette_score

print("ARI:", adjusted_rand_score(digits.target, y_pred))
print("AMI:", adjusted_mutual_info_score(digits.target, y_pred))
print("Silhouette Score:", silhouette_score(digits.data, y_pred))
"""
ARI: 0.6703800183468681   // 67% 이상이므로 좋은 값
AMI: 0.7417664506416767
Silhouette Score: 0.18249069204151275  // 안 좋은 값 
"""
```


**군집화 결과**를 **주성분 분석**을 통해 2차원에 투영할 수 있다. (겹쳐져 있는 부분의 경우  2차원보다 높은 고차원 상에서는 떨어져 있을 수 있다.)


```py
from sklearn.decomposition import PCA

pca = PCA(n_components = 2)   # PCA를 통해 차원축소해서 대강 볼 수 있다.
X = pca.fit_transform(digits.data)

plt.scatter(X[:, 0], X[:, 1], c=y_[pred], cmap=plt.cm.Set1)
plt.show()
```

![](/assets/images/UnSupervised2_2.png)

- K-평균 군집화는 너무 차원이 높을 때는 (유클리드 커리를 사용해도) 군집화 성능이 떨어질 수 있다. 이때는 차원 축소를 한 후 군집화 하는 것이 도움이 될 수 있다.

- K-means의 장점은 알고리즘이 단순해 비교적 빠르고, prediction이 가능하다는 것이다. 하지만 단점은 클러스터가 원 모양이고, 크기가 같다고 가정하기 때문에 그 이외의 값에 적용하기 어렵다. 

## 3. 미니배치 K-평균 군집화

**미니배치 K-평균(mini-batch)군집화 방법**을 사용하면 계산량을 줄일 수 있다. 미니배치 K-평균 군집화는 데이터를 미니배치 크기만큼 무작위로 분리해서 **K-평균 군집화**를 한다. 모든 데이터를 한꺼번에 썼을 때와 **결과가 다를 수 있지만 큰 차이가 없다**.


scikit-learn의 `cluster`서브 패키지는 `MiniBatchMeans`클래스를 제공한다. 미니배치 크기 `batch_size`인수를 추가로 받는다. 

### 예제 코드

아래의 코드는 150,000개의 데이터를 사용해 실행시간을 비교해보는 코드이다. 군집화의 속도는 MiniBatch의 경우가 빠름을 확인할 수 있다.

``` py
# 0. 데이터 생성 (150,000개)
from sklearn.datasets import make_blobs
X, y = make_blobs(n_samples=150000, cluster_std=[1.0, 2.5, 0.5], random_state=170)

# 1. KMenas 방법
from sklearn.cluster import KMeans
%%time
model1 = KMeans(n_clusters=3).fit(X)
"""
CPU times: user 1.48 s, sys: 3.32 s, total: 4.81 s
Wall time: 10 s  <--
"""

# 2. MiniBatchKmeans 방법
from sklearn.cluster import MiniBatchKMeans
%%time
model2 = MiniBatchKMeans(n_clusters=3, batch_size=1000, compute_labels=True).fit(X)
"""
CPU times: user 340 ms, sys: 1 s, total: 1.34 s
Wall time: 2.9 s <--
"""
```

위 확인 결과처럼 **군집화의 속도**가는 'MiniBatchKmeans'가 더 빠름을 확인할 수 있었지만, 실제적으로 **군집화 결과**는 그다지 차이가 없다.

```py
idx = np.random.randint(150000, size=300)
plt.subplot(121)
plt.scatter(X[idx, 0], X[idx, 1], c=model1.labels_[idx])
plt.title("K-평균 군집화")

plt.subplot(122)
plt.scatter(X[idx, 0], X[idx, 1], c=model2.labels_[idx])
plt.title("미니배치 K-평균 군집화")

plt.tight_layout()
plt.show()
```

![](/assets/images/UnSupervised2_3.png)