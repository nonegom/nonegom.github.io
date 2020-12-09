---
title:  "선형대수 - 벡터공간 / 일차종속 및 일차독립 / 기저 및 차원"
excerpt: "벡터공간 / 일차종속 및 일차독립 / 기저 및 차원"
categories:
  - LinearAlgebra
tags:
  - 12월
og_image: "/assets/images/green.jpg"
toc: true
toc_sticky: true
toc_label: 페이지 목차
use_math: true
---

> 선형대수 포스팅은 [유튜브 수업](https://www.youtube.com/playlist?list=PLxMkK1K0XECOj2sZG-gCk-CjvZhJ_75I4)을 기반으로 작성되었습니다.

# 벡터공간

## 정의 1
집합 $V$의 임의의 원소 $u, v$와 임의의 스칼라 $k$에 대하여 아래의 조건을 만족할 때, 집합(set)$V$를 공간(spcae)$V$라고 한다.
1. $u+v \in V$
2. $xw \in V$

- 집합 = 공간인데, 위 두 가지 조건을 만족하는 집합을 **공간**이라고 한다.

## 정의 2
공간 $V$의 임의의 원소 $u, v, w$와 임의의 스칼라 $k, l$에 대해 다음이 모두 만족될 때, 공간 $V$를 **벡터공간**(Vector Space) $V$라고 한다.

1. $u+v = v + u$
2. $(u+v) + w = u (v + w)$
3. $u + 0 = u$
4. $u + (-u) = 0$
5. $k(u+v) = ku + kv$
6. $(k + l)u = ku + lu$
7. $(kl)u = k(lu) = l(ku)$
8. $lu = k$

- 공간 중 위 8가지 성질을 모두 만족하면 **벡터공간**이라고 한다.
- 하지만 문제에서는 모든 조건을 다 확인할 수 없기 때문에 아래 3가지 조건만 확인한다. 이 조건들만으로 문제가 풀린다.

1. $u+v \in V$
2. $ku \in V$
3. ZeroVector 존재하는지 확인 (위의 3번 조건)
  - ZeroVector라는 것은 다양한 게 나올 수 있다.
  - 벡터 공간에서는 **영벡터**, 행렬 공간에서는 **영행렬**, 다항식 집합에서는 **0**을 의미한다.


## 정의 3
벡터공간 $V$에 포함된 부분집합 $W$가 벡터공간의 정의를 만족할 때, **부분공간(subspace)** $W$라 한다. 

- 벡터 공간의 일부분을 뜯어냈을 때 벡터공간의 성질을 가지고 있으면 그것을 **부분공간**이라고 한다.

- 부분공간과 벡터 공간의 풀이는 똑같다.
  - 벡터공간이든 부분공간이든 처음에 ZeroVector를 확인해야 한다.

- $V$의 부분공간에는 반드시 {0}과 $V$가 존재한다.

### 예제
- $R^3$에서 영벡터가 아닌 임의의 벡터 a, b, c가 주어질 때 항상 $R^3$의 **부분공간**이 되는 것은, **a,b,c에 모두 직교하는 벡터들의 잡합**이다.

- $R^2$의 부분공간은 {0}, $R^2$ 그리고 $R^2$의 원점을 지나는 모든 직선뿐이다.

- $R^2$의 부분공간은 {0}, $R^3$ 그리고 $R^2$의 원점을 지나는 모든 직선과 **원점을 지나가는 평면**도 된다.

# 일차종속 및 독립

## 일차결합과 생성
벡터공간 $V$의 원소 $V_1, V_2, \dots, V_n$에 대해 $a_1, a_2, \dots, a_n$이 임의의 실수일 때, $a_1V_1 + a_2V_2 + \dots + a_nV_n$을 $V_1, V_2, \dots, V_n$의 **일차결합**이라고 한다.
벡터공간 $V$의 부분집합 $V_1, V_2, \dots, V_n$이 생성하는 벡터공간이라 함은 $a_1V_1 + a_2V_2 + \dots + a_nV_n$을 말한다.
$$
\begin{matrix}
V &=& \begin{Bmatrix} 
\begin{pmatrix} 1\\1 \end{pmatrix}, \begin{pmatrix} 1\\3 \end{pmatrix}, \begin{pmatrix} 2\\5 \end{pmatrix}, \cdots
\end{Bmatrix} \\
&=& 1\begin{pmatrix} 1\\1 \end{pmatrix}+ 2\begin{pmatrix} 1\\3 \end{pmatrix} -3\begin{pmatrix} 2\\5 \end{pmatrix} & \text{일차 결합}\\
&=& \begin{pmatrix} 1\\1 \end{pmatrix}+ \begin{pmatrix} 2\\6 \end{pmatrix} + \begin{pmatrix} -6\\-15 \end{pmatrix} = \begin{pmatrix} -3\\-8 \end{pmatrix} & \text{생성}
\end{matrix}
$$

- 여기서 원소에 상수가 와서 만들어진 모양을 **일차결합**이라 하고, 그것들이 연산되어 만들어진 것을 **생성**이라고 한다.


## 독립과 종속
벡터공간 $V$의 부분집합$V_1, V_2, \dots, V_n$과 임의의 실수 $a_1, a_2, \dots, a_n$에 대해 $a_1V_1 + a_2V_2 + \dots + a_nV_n$이라 할 때,

1. $a_1 = a_2 = \cdots = a_n = 0$ 이면 {$V_1, V_2, \dots, V_n$}은 **일차(선형)독립**이라고 한다.

2. $a_1, a_2, \cdots, a_n$ 중에 적어도 하나 0이 아닌 것이 존재할 때(불필요한 것), {$V_1, V_2, \dots, V_n$}은 **일차(선형)종속**이라 한다.

- 정의의 의미는 '= 0'을 만드는 조건에서 실수($a$)를 모두 0을 곱해야만 모든 원소들($v$)이 0이 되는지를 확인하는 것이다. 만약 a가 모두 0이어야만 '= 0'이 된다면 모든 원소의 값이 다 다르다는 것(실수배를 해서 다른 원소가 될 수 없음)을 의미한다. 
$$
\begin{matrix}
V = \begin{Bmatrix} \begin{pmatrix} 1 \\ 1\end{pmatrix}, \begin{pmatrix} 1 \\ 3 \end{pmatrix} \end{Bmatrix} \text{- 선형독립} \\
V = \begin{Bmatrix} \begin{pmatrix} 1 \\ 1\end{pmatrix}, \begin{pmatrix} 2 \\ 2 \end{pmatrix} \end{Bmatrix} \text{- 선형종속}
\end{matrix}
$$

- 다른 벡터를 실수배해서 또 다른 벡터를 만들어낼 수 있으면 **종속**, 없으면 **독립**

## 일차독립 최대 개수
위에는 종속일 때, 아래는 독립일 때이다. 
$$
\begin{matrix}
\begin{pmatrix} 1 \\ 1\end{pmatrix}, \begin{pmatrix} 2 \\ 2 \end{pmatrix} &\Rightarrow&
\begin{vmatrix} 1 & 1 \\ 2 & 2 \end{vmatrix}  = 0 & \iff &
\begin{vmatrix} 1 & 1 \\ 0 & 0 \end{vmatrix}
\\
\begin{pmatrix} 1 \\ 1\end{pmatrix}, \begin{pmatrix} 1 \\ 3 \end{pmatrix} &\Rightarrow&
\begin{vmatrix} 1 & 1 \\ 1 & 3 \end{vmatrix} \neq 0 & \iff &
\begin{vmatrix} 1 & 1 \\ 0 & 2 \end{vmatrix}
\end{matrix}
$$


- 이를 확인해서 벡터들이 나왔을 때, 벡터를 눕혀서 행렬식을 구하고, **행렬식**이 0이 나오면, 일차종속, 0이 나오지 않으면 일차독립이다. 
$$|A| \neq 0 \iff 일차독립$$
$$|A| = 0 \iff 일차종속$$

- **랭크(rank)**를 구해 맨 마지막에 '0 행'이 나온다면 **종속**, '0 행'이 나오지 않는다면 **독립**이다.

- $n \times n$ 정방행렬이면 행렬식으로 풀고, 아닐 때에는 rank로 푼다.

## 기저
벡터공간 $V$에 대하여 S = $V_1, V_2, \dots, V_n$는 $V$의 부분집합이라 하자. 집합 S가 다음 두 가지 조건을 만족할 때, S를 V의 **기저(basis)**라고 한다.
1. {$V_1, V_2, \dots, V_n$}이 $V$를 **생성**한다.
2. {$V_1, V_2, \dots, V_n$}이 **선형독립**이다.

- 기저는 공간(어떤 것)을 **생성**함에 있어 엑기스(**선형독립**)인 것들만 모인 것을 의미한다.
  - 엑기스라는 의미는 얘네만 있으면 $V$(공간)을 생성해낼 수 있는 것들을 말한다.(중복 X = 종속 X)

# 기저 및 차원
## 표준기저
$R^2$ 공간(2차원)일 때, **표준기저**는 {(1,0), (0,1)}로 모든 좌표를 표시할 수 있다. 
$R^3$ 공간(3차원)일 때, **표준기저**는 {(1,0,0), (0,1,0), (0,0,1)}로 모든 좌표를 표시할 수 있다.

- 기저: 일차 독립이면서 공간에 모든 원소를 생성해낼 수 있는 것

- 그 공간을 생성해내고, 독립 관계성을 가지고 있는 것을 기저라고 하는데, 그 **기저는 정해져 있지 않다**. 다만, 위와 같은 특정한 기저를 **표준기저**라고 한다.

### 행렬의 표준 기저
- $M_{n\times n}$ 표준기저 중 $M_{2\times 2}$
$$
\begin{Bmatrix}
\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix},
\begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix},
\begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix},
\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix},
\end{Bmatrix}
$$

### 다항식의 표준 기저
- $P_n$의 표준기저: {$1, x, x^2, \cdots, x^n$}
  - $P_2$: {$1, x, x^2$}
  - $P_3$: {$1, x, x^2, x^3$}

## 기저
- 똑같이 공간을 생성해낼 수 있는 엑기스이지만, 어느 것을 사용하냐에 따라서 **횟수나 하는 방법**이 달라질 수 있다.
- 표준기저{(1,0), (0,1)}나 기저{(1,1), (0,1)}나 저마다의 방법으로 공간을 생성해낼 수 있다.
$$
\begin{matrix}
(3,4) = 3(1, 0) + 4(0, 1) \rightarrow 좌표 \begin{bmatrix} 3 \\ 4 \end{bmatrix}
\\
(3,4) = 3(1, 1) + 1(0, 1) \rightarrow 좌표 \begin{bmatrix} 3 \\ 1 \end{bmatrix}
\end{matrix}
$$

- **좌표**: 기저로 특정 값을 만들어 내기 위해 곱한 실수의 집합

- 기저는 변할 수 있지만 기저의 개수는 변하지 않는다.
  - 2차원에서 될 수 없는 기저 : {(1,0)} -> 기저가 1개는 안된다.

- $S= {v_1,v_2, ..., v_n}$이 벡터 공간 $V$의 **기저**이면 $V$에 속하는 모든 벡처 $v$는 적당한 실수 $a_1, a_2, \dots, a_n$에 대해 $a_1v_1 + a_2v_2 + \dots + a_nv_n$로 나타낼 수 있다. 이때, ($a_1, a_2, \dots, a_n$)을 기저 $S$에 대한 $v$의 **좌표벡터**(상대좌표, 좌표행렬)라고 한다. 

## 벡터의 차원
벡터공간 $V$가 $n$개의 벡터로 이루어진 기저를 갖는다면, $V$의 차원을 $n$이라 한다. 또한 $V$의 차원을 $dim\ V$로 표시한다.

- 벡터공간 $V$의 **기저의 원소 개수**

- = 벡터공간 $V$의 **선형독립이 되는 최대 개수**

- = 벡터공간 $V$의 **차원** = $dim\ V$

즉, 차원을 구하라고 한다면, 기저의 원소개수나 선형독립이 되는 최대 개수를 구한다.

### 예제
- $V = $ {$a,b,c,d,e \| a+b+c+d+e = 0 $}벡터공간에 대한 차원을 구하면?

a가 1일때, b가 1일때, c가 1일때, d가 1일 때만을 구하면 된다. (이때, e는 -1이 되면 조건을 만족한다)
$$
\begin{Bmatrix}
(1,0,0,0,-1) & (0,1,0,0,-1) \\
(0,0,1,0,-1) & (0,0,0,1,-1) \\
\end{Bmatrix}
$$

- 더한 식으로 나오면 앞의 것과 끝의 것만 남기고 나머지를 0으로 두면서 구하면 기저를 구할 수 있다.

## 대칭/교대 행렬의 차원
- 대칭행렬$(A^T = A)$의 기저는 상삼각행렬의 개수만 구하면 된다. 
   - 대칭행렬 $(M_{n \times n})$의 차원 = $\frac{n(n+1)}{2}$

- 교대행렬$(A^T = -A)$의 기저는 상삼각행렬 중 대각 성분을 미포함한 개수를 구하면 된다.
  - 교대행렬 $(M_{n \times n})$의 차원 = $\frac{n(n-1)}{2}$