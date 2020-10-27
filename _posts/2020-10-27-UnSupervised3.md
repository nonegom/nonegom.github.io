---
title:  "[머신러닝] 비지도학습 - 3. 디비스캔(DBSCAN) 군집화"
excerpt: "군집화 중 디비스캔(DBSCAN) 군집화의 장점과 성능측정"

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
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/16.03%20%EB%94%94%EB%B9%84%EC%8A%A4%EC%BA%94%20%EA%B5%B0%EC%A7%91%ED%99%94.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 1. 디비스캔 군집화 
K-평균 군집화 방법은 단순하고 강력한 방법이지만 군집의 모양이 원형이 아닌 경우에는 잘 동작하지 않으며 군집의 갯수를 사용자가 지정해주어야 한다는 단점이 있다.
디비스캔(DBSCAN / Density-Based Spatial Clustering of Applications with Noise) 군집화 방법은 데이터가 밀집한 정도 즉 **밀도**를 이용한다. 군집의 형태에 구애받지 않으며 군집의 개수를 사용자가 지정할 필요가 없다.  
초기 데이터로부터 근접한 데이터를 찾아나가는 방법으로 군집을 확장한다. 이 때 사용자 인수를 사용한다.
- 최소 거리 $epsilon$ (eps): 이웃(neighborhood)을 정의하기 위한 거리
- 최소 데이터 개수: **밀접 지역**을 정의하기 위해 필요한 **이웃의 개수**

만약 $\epsilon$ 최소 거리 안 이웃 영역 안에 최소 데이터 개수 이상의 데이터가 있으면 그 데이터를 **핵심(core) 데이터**라고 한다. **핵심 데이터의 이웃 영역 안에 있는 데이터**도 마찬가지로 연결된 핵심 데이터가 된다.  
만약 고밀도 데이터에 더 이상 이웃이 없으면 이 데이터는 **경계(border) 데이터**라고 한다. 핵심 데이터도 아니고 경계 데이터도 아닌 데이터를 **outlier(Noise Point)**라고 한다.

![](/assets/images/UnSupervised3_1.png)

### - 장점
- 복잡한 데이터 clustering 가능
- 클러스터의 개수를 정하지 않아도 됨
- Outlier도 발견 가능


scikit-learn의 `cluster` 서브패키지는 디비스캔 군집화를 위한 `DBSCAN`클래스를 제공한다. 또한 다음과 같은 인수를 받는다. 

- `eps`: 이웃을 정의하기 위한 거리
- `min_samples`: 핵심 데이터를 정의하기 위해 필요한 **이웃 영역 안의 데이터 개수**

군집화가 끝나면 객체는 다음 속성을 가진다.
- `labels_`: 군집번호. 아웃라이어는 -1값을 가진다.
- `core_sample_indices_`: 핵심 데이터의 인덱스. 여기에 포함되지 않고 아웃라이어도 아닌 데이터는 **경계 데이터**이다.

### - 예제 코드

아래 코드는 `make_circles` 명령과 `make_moons` 명령으로 만든 동심원, 초승달 데이터를 디비스캔으로 군집화한 결과를 나타낸 것이다. 마커(marker)의 모양은 군집을 나타낸다.

```py
from sklearn.datasets import make_circles, make_moons
from sklearn.cluster import DBSCAN

n_samples = 1000
np.random.seed(2)
X1, y1 = make_circles(n_samples=n_samples, factor=.5, noise=.09)
X2, y2 = make_moons(n_samples=n_samples, noise=.1)

def plot_DBSCAN(title, X, eps, xlim, ylim):
    model = DBSCAN(eps=eps)
    y_pred = model.fit_predict(X)
    idx_outlier = model.labels_ == -1
    plt.scatter(X[idx_outlier, 0], X[idx_outlier, 1], marker='x', lw=1, s=20)
    plt.scatter(X[model.labels_ == 0, 0], X[model.labels_ == 0, 1], marker='o', facecolor='g', s=5)
    plt.scatter(X[model.labels_ == 1, 0], X[model.labels_ == 1, 1], marker='s', facecolor='y', s=5)
    X_core = X[model.core_sample_indices_, :]
    idx_core_0 = np.array(list(set(np.where(model.labels_ == 0)[0]).intersection(model.core_sample_indices_)))
    idx_core_1 = np.array(list(set(np.where(model.labels_ == 1)[0]).intersection(model.core_sample_indices_)))
    plt.scatter(X[idx_core_0, 0], X[idx_core_0, 1], marker='o', facecolor='g', s=80, alpha=0.3)
    plt.scatter(X[idx_core_1, 0], X[idx_core_1, 1], marker='s', facecolor='y', s=80, alpha=0.3)
    plt.grid(False)
    plt.xlim(*xlim)
    plt.ylim(*ylim)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    return y_pred

plt.figure(figsize=(10, 5))
plt.subplot(121)
y_pred1 = plot_DBSCAN("동심원 군집", X1, 0.1, (-1.2, 1.2), (-1.2, 1.2))
plt.subplot(122)
y_pred2 = plot_DBSCAN("초승달 군집", X2, 0.1, (-1.5, 2.5), (-0.8, 1.2))
plt.tight_layout()
plt.show()
```

![](/assets/images/UnSupervised3_2.png)

- 자세히 봤을 때, 이중으로 표현되어있는 **핵심 데이터**
- 한 번 표시된 것이 **경계 데이터**
- x로 표시된 것이 **outlier**


위 군집화 결과의 ARI(조정 랜드지수)와 AMI(조정 상호정보량)값은 다음과 같다.

```py
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score
print("동심원 군집 ARI:", adjusted_rand_score(y1, y_pred1))
print("동심원 군집 AMI:", adjusted_mutual_info_score(y1, y_pred1))
# > 동심원 군집 ARI: 0.9414262371038592
#   동심원 군집 AMI: 0.8967648464619999

print("초승달 군집 ARI:", adjusted_rand_score(y2, y_pred2))
print("초승달 군집 AMI:", adjusted_mutual_info_score(y2, y_pred2))
# > 초승달 군집 ARI: 0.9544844153926417
#   초승달 군집 AMI: 0.9151495815452475
```

### - 참고 사항

DBSCAN 군집화의 경우 2개의 인수 `minimum_point`와 `eps`(epsilon)을 결정해야 한다. 하지만 eps를 구하기 위해 `sklearn.cluster.OPTICS`가 존재한다. `OPTICS`는 maximum EPS를 정해놓고 자기가 스스로 try를 해봄으로써 EPS를 구한다. 

- 따라서 우리는 `OPTICS`를 사용해 `eps`를 구하고, `minimum_point`만 구하면 되기 때문에 스스로 정해야 하는 인수를 하나로 줄일 수 있다.
