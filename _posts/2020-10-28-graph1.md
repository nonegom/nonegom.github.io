---
title:  "[머신러닝] 그래프 모형 - 1. 그래프 기초 용어"
excerpt: "확률적 그래프 모형에 필요한 기초 용어 및 함수"

categories:
  - GraphModel
tags:
  - UnSupervisedLearning
  - 10월
toc: true
toc_sticky: true
toc_label: 페이지 목차
use_math: true
---
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/17.01%20%EA%B7%B8%EB%9E%98%ED%94%84%20%EC%9D%B4%EB%A1%A0%20%EA%B8%B0%EC%B4%88.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

# 그래프 기초
그래프(Graph)는 **노드**(Node, Vertex)와 그 사이를 잇는 **간선**(Edge)으로 이루어진 구조를 말한다.  
수학적으로 그래프$G$는 **노드 집합 $V$**와 **간선 집합 $E$**로 구성된다. 
$$G = (V, E)$$

![](/assets/images/Graph1_1.JPG)

- 간선은 두 개의 노드로 이루어진 순서가 있는 쌍이다.
- 그래프는 4개의 노드집합{0, 1, 2, 3}과 6개의 간선집합{(0,1), (0,2), (0,3), (1,1), (1,2), (1,3)}을 가진다.

## - 방향성 그래프 / 비방향성 그래프
만약 두 노드 a와 b를 잇는 간선 (a, b)와 (b, a)가 있을 때, 이 두 간선을 다른 것으로 본다면 간선의 방향이 있는 **방향성 그래프(directed graph)**로 보고 두 간선을 같은 것으로 본다면 간선의 방향이 없는 **비방향성 그래프(undirected graph)**로 본다. 방향성 그래프의 경우 화살표로 방향을 표시한다.

### NetworkX패키지

`NetworkX`는 그래프를 다루기 위한 **파이썬 패키지**이다. 그래프를 만드는 클래스 `Graph` (비방향성 그래프), `DiGraph`(방향성 그래프) 를 제공한다.

**함수** (1.11버전 기준)


- `add_node`: 노드를 추가 (노드의 이름은 숫자나 문자)
- `add_edge`: 간선을 추가 (간선을 이을 두 노드를 인수로 입력)
- `nodes`: 그래프에 포함된 노드 확인 
- `edges`: 그래프에 포함된 간선 확인



`graphviz` 프로그램과 `pydot`패키지가 설치되어 있다면 이를 이용해 시각화가 가능하다.

```py
# 노드 추가
gp.add_node("a")
gp.add_node(1)
gp.add_node(2)
gp.nodes()
# > ['a', 1, 2]

# 간선 추가
gp.add_edge(1, "a")
gp.add_edge(1, 2)
gp.edges()
# > [(1, 'a'), (1, 2)]

# 그래프 시각화
from IPython.core.display import Image
from networkx.drawing.nx_pydot import to_pydot

d1 = to_pydot(gp)
d1.set_dpi(300)
d1.set_rankdir("LR")
d1.set_margin(1)
Image(d1.create_png(), width=300)
```

![](/assets/images/Graph1_2.jpg){: width="350" height="200"}{: .align-center}

## - 크기, 이웃, 루프

노드 집합 $V$와 간선 집합 $V$를 가지는 그래프 $G$에 포함된 **노드의 갯수**를 **그래프의 크기(cardinality)**라고 하며 
|G|또는 
|V|로 나타내고 **간선의 갯수**는 
|E|로 나타낸다.


NetworkX 패키지에서는 각각 `len`명령, `number_of_nodes`, `number_of_edges`메서드로 계산 가능하다.

```py
len(gp), gp.number_of_nodes(), gp.number_of_edges()
# > (3, 3, 2)
```

만약 두 노드 $a, b$, 를 포함하는 간선$(a, b)$가 안에 존재하면 두 노드는 **인접하다(adjacent)**고 하며 인접한 두 노드는 서로 **이웃(neighbor)**이라고 한다.
$$(a, b) \in E$$

NetworkX 패키지 `Graph` 클래스의 `neighbors` 메서드는 인수로 받은 노드에 인접한 노드를 생성하므로 **인접성을 확인하는데 사용**할 수 있다.

```py

for n in gp.neighbors(1):
print(n)
# > a
#   2

## - 위에서 계속 예제로 만든 `gp`라는 객체 사용 1의 노드와 연결된 a, 2노드 출력

2 in gp1.neighbors(1), "a" in gp1.neighbors(1), 1 in gp1.neighbors(2), "a" in gp1.neighbors(2)
# (True, True, False, False)
```

어떤 노드에서 출발해 자기 자신으로 돌아오는 간선이 있다면 **셀프 루프(self loop)**라고 한다. 셀프 루프의 경우에는 `graphviz`로만 시각화할 수 있다.

```py
# 그래프 생성
gp2 = nx.Graph()
gp2.add_node(1)
gp2.add_node(2)
gp2.add_node(3)
gp2.add_edge(1, 2)
gp2.add_edge(2, 2) # self loop
gp2.add_edge(2, 3)
np.random.seed(0)

# 시각화
d2 = to_pydot(gp2)
d2.set_dpi(600)
d2.set_rankdir("LR")
image(d2.create_png(), width = 600)
```

![](/assets/images/Graph1_3.JPG){: width="400" height="100"}{: .align-center}


## - 워크, 패스, 사이클, 트레일
- **워크(walk)**:
어떤 노드를 출발해서 다른 노드로 도달하기 위한 인접한 노드의 순서열


- **패스(path)**:
워크 중에서 시작과 끝을 제외한 다른 노드에 대해서 동일한 노드를 두 번 이상 지나지 않는 워크


-  **사이클(cycle)**:
패스 중에서 시작점과 끝점이 동일한 패스. 사이클이 없는 그래프를 어사이클릭 그래프(acyclic graph)라고 한다. 
    - (동일한 노드를 두 번 지나더라도 두 번의 노드가 시작점과 끝점이라면 '사이클'이다.)



- **트레일(trail)**:
어떠한 노드든 동일한 노드를 두 번 이상 지나지 않는 워크 
    - (사이클보다 더 빡빡하고, 패스보다는 더 좁은 개념)

```py
# 그래프 생성
gp3 = nx.Graph()
gp3.add_node("a")
gp3.add_node("b")
gp3.add_node("c")
gp3.add_node("d")
gp3.add_node("e")
gp3.add_node("f")

# 간선 생성
gp3.add_edge("a", "b")
gp3.add_edge("a", "c")
gp3.add_edge("b", "c")
gp3.add_edge("c", "d")
gp3.add_edge("d", "e")
gp3.add_edge("c", "e")

# 그래프 시각화
d3 = to_pydot(gp3)
d3.set_dpi(600)
d3.set_rankdir("LR")
Image(d3.create_png(), width=800)
```

![](/assets/images/Graph1_5.png)

위 그래프에서 **워크, 트레일, 패스, 사이클**을 찾아보도록 하자.

- $a−c−d−c−e$는 $a$에서 $c$로 가는 **워크**이다. 하지만 트레일이나 패스는 아니다.

- $a−b−c−d−e$는 **트레일**이다.

- $a−b−c−d−e−c$는 **패스**지만 트레일은 아니다.

- $a−b−c−a$는 **사이클**이다.


`has_path`명령으로 두 노드간에 패스가 존재하는지 알 수 있다. 만약 패스가 존재하면 `shortest_path`명령으로 가장 짧은 패스를 구할 수 있다.

```py
# 패스 존재 여부
nx.has_path(gp3, "a", "b"), nx.has_path(gp3, "a", "e"), nx.has_path(gp3, "a", "f")
# > (True, True, False)

# 최단 거리 패스
nx.shortest_path(gp3, "a", "e")
# > ['a', 'c', 'e']
```

## - 클리크
**무방향성 그래프의 노드 집합** 중에서 모든 노드끼리 간선이 존재하면 그 **노드 집합**을 **클리크(clique)**라고 한다. 만약 클리크에 포함된 노드에 **인접한 다른 노드를 추가해 클리크가 아니게** 된다면 **최대클리크(maximal clique)**라고 한다. 다음 그래프 `gp4`에서 클리크를 찾아보자.

```py
# 그래프 객체 생성
gp4 = nx.Graph()

# 노드, 간선 추가 및 시각화 코드 생략

```
![](/assets/images/Graph1_4.png)

- ${a,b}$는 **클리크**이다. 하지만 최대클리크는 아니다.

- ${a,b,c}$는 클리크이며 **최대클리크**이다.


`enumerate_all_cliques`명령은 **모든 클리크**를, `find_cliques`는 모든 **최대클리크**를 찾는다.

```py
## - 모든 클리크 찾기
[c for c in nx.enumerate_all_cliques(gp4)]

"""
[['a'],
 ['b'],
 ['c'],
 ['d'],
 ['e'],
 ['f'],
 ['a', 'b'],
 ['a', 'c'],
 ['b', 'c'],
 ['b', 'd'],
 ['c', 'd'],
 ['d', 'e'],
 ['d', 'f'],
 ['e', 'f'],
 ['a', 'b', 'c'],
 ['b', 'c', 'd'],
 ['d', 'e', 'f']]
"""

## - 모든 최대클리크 찾기
[c for c in nx.find_cliques(g4)]
"""
[['d', 'f', 'e'], ['d', 'c', 'b'], ['a', 'c', 'b']]
"""
```