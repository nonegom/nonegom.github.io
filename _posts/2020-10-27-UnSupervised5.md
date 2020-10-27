---
title:  "[머신러닝] 비지도학습 - 5. Affinity Propagation"
excerpt: "군집화의 방법 중 Affinity Propagation 방법"

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
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/16.05%20Affinity%20Propagation.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## Affinity Propagation
모든 데이터가 특정한 기준에 따라 자신을 대표할 데이터를 선택한다. 만약 스스로가 자기 자신을 대표하게 되었을 때, 그 데이터가 **군집화(Clsuter)**의 중심이 된다.  
K-Median(K-중앙값)군집화와 유사하지만 차이점은 K값을 사용자가 정하는 게 아니라, 다른 하이퍼 파라미터 값만 정해준다.

- responsibility $r(i,k)$
    - k번째 데이터가 i번째 데이터의 **대표가 되어야 한다는 근거**

- availability $a(i,k)$
    - i번째 데이터가 k번째 데이터를 **대표로 선택해야 한다는 근거**


- responsibility와 availability를 모두 0으로 놓고 다음 수식을 수렴할 때까지 반복
$$r(i,k) \leftarrow s(i,k)−max_{k′ \ne k}(a(i,k′) + s(i,k′))$$
$$a(i,k) \leftarrow min(0,r(k,k)+\sum_{i′\ne i,k} r(i′,k))$$

여기에서 $s(i,k)$는 **음의 거리로 정의되는 유사도**이다.
$$s(i,k) = −\|x_i−x_k \|^2$$


특히 $s(k,k)$는 특정한 음수 값으로 사용자가 정해 주게 되는데 이 값에 따라서 클러스터의 개수가 달라지는 **하이퍼 모수**가 된다. $s(k,k)$가 크면 자기 자신에 대한 유사도가 커져서 **클러스터의 수가 증가**한다.


위 알고리즘으로 계산하는 $r, a$가 더 이상 변화하지 않고 수렴하면 계산이 종료되고 종료 시점에서 $r(k,k) + a(k,k) > 0$이 데이터가 클러스터의 중심이 된다.

```py
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import *

# 샘플 데이터 생성
centers = [[1, 1], [-1, -1], [1, -1]]
X, labels_true = make_blobs(n_samples=300, centers=centers, cluster_std=0.5, random_state=0)

# 모델 생성(하이퍼 파라미터 '-50')
model = AffinityPropagation(preference=-50).fit(X)


# 모델 속성 출력
cluster_centers_indices = model.cluster_centers_indices_
labels = model.labels_
n_clusters_ = len(cluster_centers_indices)

print('군집 개수 추정치: %d' % n_clusters_)
print("조정 랜드지수: %0.3f" % adjusted_rand_score(labels_true, labels))
print("조정 상호정보량: %0.3f" % adjusted_mutual_info_score(labels_true, labels))
print("실루엣 계수: %0.3f" % silhouette_score(X, labels, metric='sqeuclidean'))
```

```py
### 그래프 출력

from itertools import cycle

colors = cycle('rgb')
for k, col in zip(range(n_clusters_), colors):
    class_members = labels == k
    cluster_center = X[cluster_centers_indices[k]]
    plt.plot(X[class_members, 0], X[class_members, 1], col + '.')
    for x in X[class_members]:
        plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col, alpha=0.25)
    plt.plot(cluster_center[0], cluster_center[1], 'o', mec='k', mew=3, markersize=7)

plt.show()
```
![](/assets/images/UnSupervised5_1.png)

- 3개의 군집이 각각 대표를 기준으로 군집화되어 있음을 확인할 수 있다.