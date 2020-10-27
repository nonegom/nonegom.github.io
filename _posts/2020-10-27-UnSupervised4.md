---
title:  "[머신러닝] 비지도학습 - 4. 계층적 군집화"
excerpt: "군집화 중 계층적 군집화에서의 비계층적 거리 측정법과 계층적 거리 측정법"

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
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/16.04%20%EA%B3%84%EC%B8%B5%EC%A0%81%20%EA%B5%B0%EC%A7%91%ED%99%94.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 0. 계층적 군집화
**계층적 군집화(hierachical clustering)**는 여러 개의 군집 중에서 **가장 유사도가 높거나**, **거리가 가까운 군집** 두 개를 선택하여 **하나로 합치면서 군집 개수를 줄여 가는 방법**을 말한다. **합체 군집화(agglomerative clustering)**라고도 한다.   
가장 처음에는 모든 군집이 하나의 데이터만을 가진다. 따라서 최초에는 데이터 개수만큼 군집이 존재하지만 군집을 합치면서 최종적으로 하나의 군집만 남게 된다.

### 군집간의 거리 측정
- 사용자는 이 부분에 대해 생각해야 한다.

계측정 군집화를 하기 위해서는 **모든 군집 간 거리를 측정**해야 한다. 군집 간의 거리를 측정하는 방법에는 계층적 방법에 의존하지 않는 **비계층적 방법**과 이미 이전 단계에서 계층적 방법으로 군집이 합쳐진 적인 있다는 가정을 하는 **계층적 방법** 두 가지가 있다.

## 1. 비계층적 거리 측정법
비계층적 거리 측정법은 계층적 군집화가 아니더라도 **모든 경우에 사용**할 수 있는 거리 측정 방법이다. **중심거리, 단일거리, 완전거리, 평균거리** 등이 있다. **계층적 거리측정법**에 비해 계산량이 많다는 단점이 있다.

### - 중심(centroid)거리
  두 군집의 중심점(centroid)를 정의한 다음 **두 중심점의 거리를 군집간의 거리로 정의**한다.
$$d(u,v)= \| c_u−c_v \|_2$$

여기에서 $d(u,v)$는 군집 $u$와 군집 $v$사이의 거리를 뜻한다. 또한 $c_u$와 $c_v$는 각각 두 군집 $u$와 $v$의 중심점이다. **군집의 중심점**은 그 군집에 포함된 **모든 데이터의 평균을 사용**한다.
$$c_u= \frac{1}{|u|}\sum_i{u_i}$$
이 식에서 '|$u$|' 기호는 군집의 원소 개수를 말한다.
- 만약 군집 u의 데이터 5개, 군집 v의 데이터 5개일 경우 총 25번의 거리를 구해서 최소값을 선택해야 한다.

### - 단일(single)거리
  군집 $u$의 모든 데이터 $u_i$와 군집 $v$의 모든 데이터 $v_j$의 **모든 조합**에 대해 데이터 사이의 거리 $d(u_i,v_j)$를 측정해 **가장 작은 값**을 구한다. **최소 거리(Nearest Point) 방법**이라고도 한다.
$$d(u,v)=min(d(u_i,v_j))$$

### - 완전(complete)거리
  군집 $u$의 모든 데이터 $u_i$와 군집 $v$의 모든 데이터 $v_j$의 **모든 조합**에 대해 데이터 사이의 거리 $d(u_i,v_j)$를 측정해서 **가장 큰 값**을 구한다. **최장 거리(Farthest Point) 방법**이라고도 한다.
$$d(u,v)=max(d(u_i,v_j))$$

### - 평균(average)거리
  군집 $u$의 모든 데이터 $u_i$와 군집 $v$의 모든 데이터 $v_j$의 **모든 조합**에 대해 데이터 사이의 거리 $d(u_i,v_j)$를 측정해서 평균을 구한다. |$u$|와 |$v$|는 각각 두 군집의 원소의 개수를 뜻한다.
$$d(u,v)= \sum_{i,j} \frac{d(u_i, v_j)}{|u| \ |v|}$$

## 2. 계층적 거리 측정법¶
계층적 거리 측정법은 **계층적 군집화에서만 사용**할 수 있는 방법이다. 즉, 이전 단계에서 이미 어떤 두 개의 군집이 하나로 합쳐진 적이 있다고 가정하여 이 정보를 사용하는 측정법이다. 비계층적 거리 측정법에 비해 **계산량이 적어 효율적**이다.

### - 중앙값(median)거리¶
**중심거리 방법의 변형**인 방법으로, 중심거리 방법처럼 군집의 **중심점 거리를 군집간의 거리**라고 한다. 하지만 군집의 중심점을 계산하는 방법이 다르다. 만약 군집 $s$와 군집 $t$가 결합하여 군집 $u$가 생겨 $u ← s + t$가 된다면, 군집 $u$의 중심점은 새로 계산하지 않고 원래 군집에서의 **두 군집의 중심점의 평균을 사용**한다. 
$$c_u = \frac {1}{2}(c_s + s_t)$$

따라서 해당 군집의 모든 데이터를 평균하여 중심점을 구하는 것 보다 계산이 빠르게 된다.

### - 가중(weighted)거리
가중거리를 사용하려면 원래 어떤 두 개의 군집이 합쳐져서 하나의 군집이 만들어졌는지 알아야 한다. 만약 군집 $u$가 군집 $s$와 군집 $t$가 결합하여 $u ← s + t$ 생겼다면, 이 군집 $u$와 또 다른 군집 $v$ 사이의 거리는 군집 $u$를 구성하는 원래 군집 $s, t$와 $v$ 사이에서 두 거리의 평균을 사용한다.

![](/assets/images/UnSupervised4_0.JPG)

### - 와드(Ward)거리
가중거리의 변형인 방법으로 위와 같이, 군집 $u$가 군집 $s$와 군집 $t$가 결합하여 $u ← s + t$ 생겼다면, 이 군집 $u$와 또 다른 군집 $v$ 사이의 거리는 군집 $u$를 구성하는 원래 군집 $s, t$와 $v$ 사이의 거리를 사용하는 것은 **가중거리 방법**과 같다. 하지만 원래의 두 군집 $s, t$의 거리가 너무 가까우면 다른 군집 $v$와 거리가 더 먼 것으로 인식하게 만든다.

![](/assets/images/UnSupervised4_1.JPG)

- 따라서 만약 **가중거리 값**이 같다면, 군집을 이루고 있는 다른 데이터들 간의 거리를 계산해서, 다른 데이터들간의 거리가 너무 멀면 그 쪽으로 데이터가 합쳐진다. 

- 위 그림에서 보면 $v$는 $u_2$와 합쳐지게 된다. 


> 계층적 군집화(Hierachy) 방법에서는 **와드거리 측정방식을 사용**한다.

## 3. Scipy의 계층적 군집화
파이썬으로 계층적 군집화를 하려면 **사이파이 패키지**의 `linkage`명령을 사용하거나 **사이킷런 패키지**의 `AgglomerativeClustering`클래스를 사용한다. 각각 장단점이 있지만 사이파이 패키지는 군집화 결과를 트리 형태로 시각화해주는 `dendrogram` 명령도 지원한다.

### 예제 코드
MNIST digit 이미지 중 **20개의 이미지**를 무작위로 골라 **계층적 군집화**를 적용해보는 코드이다.

```py
# 0. 데이터 로드
from sklearn.datasets import load_digits
digits = load_digits()
n_image = 20
np.random.seed(0)
idx = np.random.choice(range(len(digits.images)), n_image)
X = digits.data[idx]
images = digits.images[idx]

# 1. 데이터 출력
plt.figure(figsize=(12, 1))
for i in range(n_image):
    plt.subplot(1, n_image, i + 1)
    plt.imshow(images[i], cmap=plt.cm.bone)
    plt.grid(False)
    plt.xticks(())
    plt.yticks(())
    plt.title(i)

# 2. 계층적 군집화(와드 거리 이용)
from scipy.cluster.hierarchy import linkage, dendrogram
Z = linkage(X, 'ward')

""" >  군집 s       군집 t            거리      데이터의 개수
-------------------------------------------------------------
array([[ 3.        , 18.        , 23.51595203,  2.        ],  - 붙은 순서
       [13.        , 19.        , 25.27844932,  2.        ],
       [ 1.        , 14.        , 28.67054237,  2.        ],
       [17.        , 21.        , 31.04298096,  3.        ],
       ...
       [35.        , 37.        , 93.57946712, 20.        ]])
"""
```

![](/assets/images/UnSupervised4_2.png)

- 랜덤하게 20개의 이미지 데이터 생성


```py
# 3. 

from matplotlib.offsetbox import OffsetImage, AnnotationBbox

plt.figure(figsize=(10, 4))
ax = plt.subplot()

ddata = dendrogram(Z)

dcoord = np.array(ddata["dcoord"])
icoord = np.array(ddata["icoord"])
leaves = np.array(ddata["leaves"])
idx = np.argsort(dcoord[:, 2])
dcoord = dcoord[idx, :]
icoord = icoord[idx, :]
idx = np.argsort(Z[:, :2].ravel())
label_pos = icoord[:, 1:3].ravel()[idx][:20]

for i in range(20):
    imagebox = OffsetImage(images[i], cmap=plt.cm.bone_r, interpolation="bilinear", zoom=3)
    ab = AnnotationBbox(imagebox, (label_pos[i], 0),  box_alignment=(0.5, -0.1), 
                        bboxprops={"edgecolor" : "none"})
    ax.add_artist(ab)

plt.show()
```

![](/assets/images/UnSupervised4_3.png)

- 위를 보면 3번째와 18번째가 가장 먼저 붙은 것으로 확인되는데, 위 트리에서도 그와 같은 부분을 확인할 수 있다.
- 이처럼 **계층적 군집화**는 군집화의 과정을 시각화해 확인할 수 있다는 장점이 있다.