---
title:  "선형대수 - 고유치 및 고유벡터 / 닮은 행렬 / 선형 사상"
excerpt: "고유치 및 고유벡터 / 고유치 및 고유벡터의 성질 / 닮은 행렬  및 행렬의 대각화 / 선형 사상"
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

# 고유치 및 고유벡터
$$Av = \lambda v$$
- 고유치: ($\lambda$)
- 고유벡터($v$): 0이 아닌 벡터

- 고유치를 계산하고 싶다면 고유 벡터를 계산하면 된다.

$$
\begin{matrix}
Av - \lambda v &=& 0 \\
(Av - \lambda I) v &=& 0 \\
v &=& (A - \lambda)^{-1}\ \cdot 0 \\
  &=& 0
\end{matrix}
$$
- ($A - \lambda I$)가 역행렬이 존재하면 위처럼 0이 되어버리기 때문에
- ($A - \lambda I$)는 역행렬이 존재하면 안된다. 즉, $\| A - \lambda I \| = 0$이 되어야 하는데, 이때 이를 **고유방정식(특성방정식)**이라고 한다.

## 예제
$\begin{pmatrix} 3 & 2 \\\\ 2 & 0
\end{pmatrix}$ 의 고유치, 고유벡터를 구하시오.

- 고유치 구하기 ($\lambda$)
$$ A = \begin{pmatrix} 3 & 2 \\ 2 & 0 \end{pmatrix}$$
$$ \begin{matrix} |A -\lambda I| = \begin{vmatrix} 3 - \lambda & 2 \\ 2 & -\lambda   \end{vmatrix}
&=& (3-\lambda)(-\lambda) -4 \\
&=& \lambda^2 - 3\lambda -4 \\ 
&=& (\lambda -4) (\lambda +1) \\
&=& \text{고유치}(\lambda) = 4\ or -1
\end{matrix} 
$$

- 고유벡터 구하기 ($v$)
  1. $\lambda = 4$ 일 때,
$$
\begin{matrix} 
Av = 4v  &  Av - 4v = 0  & A-4(v) = 0 \\
\\
\begin{pmatrix} -1 & 2 \\ 2 & -4 \end{pmatrix} 
\begin{pmatrix} x  \\ y \end{pmatrix} =  
\begin{matrix} 0 \\ 0 \end{matrix} &
\begin{cases} -x + 2y = 0 \\ 2x-4y = 0 \end{cases} \\
& \therefore \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 2 \\ 1 \end{pmatrix} \ \cdots
\end{matrix} 
$$

  2. $\lambda = -1$ 일 때,
$$
\begin{matrix} 
Av = -1v  &  Av + v = 0  & (A+1)(v) = 0 \\
\\
\begin{pmatrix} 4 & 2 \\ 2 & 1 \end{pmatrix} 
\begin{pmatrix} x  \\ y \end{pmatrix} =  
\begin{matrix} 0 \\ 0 \end{matrix} &
\begin{cases} 4x + 2y = 0 \\ 2x+y = 0 \end{cases} \\
& \therefore \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} 1 \\ -2 \end{pmatrix} \ \cdots
\end{matrix} 
$$

- 고유벡터의 경우,위 방정식을 만족시키는 무슨 벡터든 가능하다.
- 고유치를 구할 때는 **대각 성분에 $-\lambda$**해야 하는 것 기억하기!

## 3x3행렬의 고유벡터
3x3행렬의 경우 고유벡터는 구하기가 어렵다. 따라서 객관식 문제에 있는 고유벡터들이 주어지면, 이를 이용해 푼다. (고유벡터의 정의 이용)

- 예제)
행렬(, 고유치) 그리고 선택지에 고유벡터가 주어졌을 때, 해당 고유벡터가 행렬의 고유벡터인지 확인하는 방법은 문제를 풀어보면 된다.
$$
\begin{matrix} \begin{pmatrix} 9 & 1 & 1 \\ 1 & 9 &  1 \\ 1 & 1 & 9 \end{pmatrix} &
\begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix} & = \begin{pmatrix} 11 \\ 11 \\ 11 \end{pmatrix}
&= 11 & \begin{pmatrix} 1 \\ 1 \\ 1 \end{pmatrix} \\
B & v &  & \ \lambda & v
\end{matrix}
$$
- 문제의 경우 $Bv$가 고유치 $11v$로 가능하므로 &v&는 고유벡터라고 할 수 있다.
- 따라서 $B$에 $v$를 곱한 행렬의 값을 $\lambda$(고유치)에 $v$를 곱해서 만들 수 있다면, $v$는 행렬 B에 고유벡터라고 할 수 있다. 

## 고유치 및 고유벡터의 성질 (★)

1. $A$행렬의 **고유치 합**은 **대각원소들의 합(trA)**이다.
2. $A$행렬의 **고유치 곱**은 **행렬식($\|A\|$)**의 값과 같다. (문제 많이 출제된다.)

> 위 성질은 절대 까먹으면 안된다.

## 고유치 및 고유벡터의 성질 2
1. $A^T$와 $A$의 고유치는 같다. (하지만 고유벡터는 같지 않을 수 있다.)
2. $A$의 고유치가 $\lambda$이면 
  - $A^n$의 고유치는 $\lambda^n$이다. $\Rightarrow A^{-1}$의 고유치는 $\lambda^{-1}$이다.
  - $aA$의 고유치는 $a\lambda$이다.
  - $A + aI$의 고유치는 $\lambda + a$이다.
  - 2.의 경우 고유치뿐만 아니라 **고유벡터** 또한 같다.

  - A행렬의 고유치에 0이 있으면, **행렬식 값은 0**이 된다.
    - 이는 비가역 행렬이고, 역행렬이 존재하지 않는다는 뜻이다.
    - 역행렬이 존재하지 않으면 행렬식의 값은 0이고, 행렬식의 값이 0이면 역행렬이 존재하지 않는다.

## 고유치 및 고유벡터의 성질 3
1. **대칭행렬**의 고유치는 항상 실수이며, 서로 다른 고유치에 해당되는 각각의 고유벡터는 수직 관계를 갖는다.
2. **교대행렬**의 고유치는 0또는 순허수($i, 2i, 3i$)이다. (복소수 $a+i$의 꼴)
3. **직교행렬**의 고유치는 1 또는 -1 및 켤레복소수이며, 그 절대값은 항상 '1'이다.
  - 이 말은 즉, 켤레복소수의 값이 $a \pm bi$일때 $\sqrt{a^2 + b^2} = 1 $이라는 얘기이다.

# 닮은 행렬
- 아래의 식을 만족하는 경우, A와 B를 **닮은 행렬**이라고 한다.
$$
\begin{matrix}
Q^{-1}AQ = B
\end{matrix}
$$

## 닮은 행렬의 성질
- 행렬 A와 B가 닮은 행렬일 경우
  1. 행렬식이 같다. ($ \|A\| = \|B\|$)
  2. trace가 같다. ($tr(A) = tr(B)$)
  3. 고유치가 같다. (고유벡터는 같지 않다)
  4. 계수(Rank)가 같다. 

## 행렬의 대각화
- 닮은 행렬의 대표적인 예라고 할 수 있다.
- 아무 $Q$나 잡지 않고, 특정한 $P$를 잡는다. 
$$
\begin{matrix}
Q^{-1}AQ = B \\
P^{-1}AP = D 
\end{matrix}
$$

- P: 대각화시키는 행렬 (A의 고유벡터로 이루어진 행렬)
- D: 대각화 행렬 (A의 고유치로 이루어진 행렬)

- 대각화 행렬의 경우 고유 벡터 또한 같다고 할 수 있다. 
- 대각화 행렬 또한 또한  **닮은행렬의 성질(행렬식, trace, 고유치, 계수가 같음)**을 갖는다.

### 행렬의 대각화 이용
- 행렬의 대각화의 경우 제곱에도 이용할 수 있다.
$$P^{-1}AP = D \Rightarrow A = PDP^{-1} \Rightarrow A^n = PD^nP^{-1}$$

### 대각화 가능
 고유벡터들의 행렬식 값이 0이 안 나오려면, **일차독립관계성**을 갖고 있어야 한다. 
 즉, 벡터들이 **독립의 관계**를 갖고 있어야 한다. $n \times n$ 행렬에서 일차 독립인 고유 벡터가 n개 존재해야 한다.

- 일차 독립인 고유 벡터의 개수가 n개 존재하면 대각화 가능하다. 
- 서로 다른 고유치가 n개 존재하면 대각화 가능하다.


- 하지만 고유치가 **중근**일 경우, 대각화가 가능 or 불가능할 수 있다. 이 경우 서로 다른 고유 벡터를 확인해서 문제를 풀어야 한다.
  - 그러나, $A$가 대각화 가능하다면 $A^n$도 대각화 가능하다. 왜냐하면, $A$의 고유벡터나 $A^n$의 고유벡터가 같기 때문이다.

> 상삼각행렬이나, 하삼각행렬의 경우 **고유치의 값은 대각성분의 값과 같다**.

# 선형 사상
- "선형사상 = 일차 변환 = 선형변환" 이라고 할 수 있다.

- 아래 두 조건을 만족하는 함수를 **선형 사상**이라고 한다.
  1. $T(v+w) = T(v) + T(w)$
  2. $T(av) = aT(v) \quad T(0) = 0$
  $$v {\overset{T} \longrightarrow} T$$

- 여기서는 **벡터 공간에서 벡터 공간**으로 보내는 함수로 생각한다.
- 선형 사상에서 원점은 원점에 대응한다 ($T(0) = 0$)
  - 선형사상은 위 조건에 의해서 $T(0 + 0) = T(0) + T(0) = 2T(0)$이 된다. 
  - 하지만 여기서 $T(0)$이 0이 아니라면 조건을 만족시키지 않는다.

## 핵과 치역
$$v\ {\overset{T} \longrightarrow}\ w$$

- 정의역(v): 집어넣을 것
- **핵공간**: 정의역 중 0으로 가는 애들
- 공역(w): 집어넣은 것에 대해 나올 것이라 예상되는 것
- **치역**: 예상된 것 중 실제로 나온 것


- **핵공간(핵집합, 열공간) = Kernel = null space**
  - $ker\ T = $ { $V \| T(v) = 0$} : 0으로 가는 v값

- **치역(상) = Image = range**
  - $Im\ T = ${$T(v) \| v \in V$}

### 핵과 치역의 차원
- 핵의 차원 = $ dim(ker \ T) = nullity\ T = n- rank\ T$
  - $n$은 열의 개수 $= dim(v)$ (정의역의 차원(rank)) 
- 치역의 차원 = $ dim(Im \ T) = rank\ T$

- 핵은 왼쪽에서 구하고, 치역은 오른쪽에서 구한다.
- 핵은 0으로 가는 정의역을 찾는 것이고, 치역은 오른쪽에 맺힌 것을 찾는 것이다.

> v(정의역)와 w(공역)의 차원이 다를 수가 있다.

## 전사함수와 단사함수
- 전사함수: 치역과 공역의 차원이 같을 경우를 의미한다.
  - 전사함수 $\iff$ 치역 = 공역
$$
\begin{matrix}
치역차원 & = 공역차원 \\
rank & = dim(w)
\end{matrix}
$$

- 단사함수: 하나의 정의역이 하나의 치역에만 대응되는 경우를 의미한다.
  - 단사함수 $\iff ker\ T = \left\\{ 0 \right\\} = nullity\ T = 0$
  - 정의역에서 0으로 갈 수 있는 집합이 있었다면, 단사함수에서는 0만이 0으로 갈 수 있다.

## 선형 사상 예제
예제1) $T(x, y) = (x+y, 2x + 2y) = (0, 0)$ 에서 핵과 치역을 구해라.
$$
\begin{matrix}
ker\ T &=& \left\{ (x,y) | y = -x \right\} & \therefore y = -x \\
Im\ T &=& \left\{ (a, b) | b = 2a \right\} & \\
&=& \left\{ (x, y) | y = 2x \right\} &
\end{matrix}
$$

- $ker T$는 0으로 가는 $x, y$를 찾는다. y = -x이다.
- $Im T$는 $a = x + y, b = 2x + 2y$이므로 b = 2a이다.

예제2) $A \in M_{2x2}, T(A) = A + A^T$처럼 행렬로 주어졌을 때, 핵과 치역을 구해라. 
(이 부분은 이해가 잘 안됐음)
$$
\begin{matrix}
\text{핵}&=& A + A^T = 0 \\
&=& A = -A^T \\
&&\therefore A: 교대행렬 \\
\\
\text{치역}&=& A + A^T \\
&&\therefore A+A^T: 대칭행렬
\end{matrix}
$$

- 치역을 구하는 데에는 아래 공식이 사용된다.
  - 공식) $A = \frac{1}{2} (A + A^T) + \frac{1}{2} (A-A^T)$

  
> 선형 사상 파트의 경우 문제로 이해하는 게 중요하다고 생각한다. 따라서 위 강의의 실습으로 연습을 해야 한다.
