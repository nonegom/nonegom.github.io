---
title:  "선형대수 - 랭크 / 벡터 / 벡터의 내적"
excerpt: "랭크 / 랭크의 활용 / 벡터 / 벡터의 내적"
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

# 랭크
랭크의 성질은 연립방정식의 풀이 방법과 완전히 동일하다. 
연립방정식에서는 다음과 같은 변경에 제약이 없다.
$$
\begin{cases}
2x - y =-1\\
x + y = 4
\end{cases}
\iff
\begin{cases}
x + y = 4\\
2x - y =-1
\end{cases} \iff
\begin{cases}
2x + 2y = 8 \\
2x - y = -1
\end{cases} 
\iff
\begin{cases}
2x + 2y = 8 \\
0 - 3y = -9
\end{cases} \iff
\begin{cases}
x + y = 4 \\
y = 3
\end{cases}
$$
- 행과 행의 순서를 바꿔도 되고
- 실수배를 해도 되고
- 대입법이나 가감법의 사용도 된다.

## 랭크의 연산법
랭크도 연립방정식 처럼 사용할 수 있다.
$$\begin{pmatrix} 2 & -1 \\ 1 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix} = \begin{pmatrix} -1 \\ 4 \end{pmatrix}$$

이 식은 아래처럼 표현이 가능하다.
여기서 `|(bar)`를 기준으로 왼쪽은 **계수행렬**, 오른쪽을 포함한 것을 **확대행렬**이라고 한다.
$$\begin{pmatrix} 2 & -1 & | & -1\\ 1 & 1 & | & 4 \end{pmatrix} \iff
\begin{pmatrix}  1 & 1 & | & 4 \\ 2 & -1 & | & -1 \end{pmatrix} \iff
\begin{pmatrix}  2 & 2 & | & 8 \\ 2 & -1 & | & -1 \end{pmatrix} \iff
\begin{pmatrix}  2 & 2 & | & 8 \\ 0 & -3 & | & -9 \end{pmatrix} \iff
\begin{pmatrix}  1 & 1 & | & 4 \\ 0 & 1 & | & 3 \end{pmatrix}
$$

> 해당 행렬은 위의 연립방정식과 해가 동일하다. 이떄 위의 $Rank = 2$이다.

- 행렬식에서는 행과 행을 바꾸면 `-`가 붙지만, 랭크에서는 `-`가 붙지 않고 제약없이 바꿀 수있다.
- 행렬식에서 어떤 행에 실수배 하는게 안되지만, 랭크는 가능하다.
- 하지만 랭크에서는 행작업만 가능하다.(열작업 X)

정리)
1. 행과 행을 바꿀 수 있다.
2. 0을 제외한 실수배가 가능하다.

### 가우스 소거법

- 랭크는 **가우스 소거법**을 이용해서 나온 **기약행사다리꼴**로 구한다. 
- 첫 번째 행의 선두 밑은 0으로, 두 번째 행의 선두 밑은 0으로 ... 
- 0이 나오는 행이 나올때까지 구하거나, 마지막 행까지 연산을 한다.
- 이때 0이 아닌 줄의 개수가 곧 **랭크**가 된다.

-$n \times n$ 의 행렬의 경우 최대 랭크는 $n$이다. 

> 가우스 소거법과 기약행 사다리꼴에 대한 정보는 해당 [행사다리꼴-나무위키](https://namu.wiki/w/%ED%96%89%EC%82%AC%EB%8B%A4%EB%A6%AC%EA%BC%B4)를 참조하자.

# 랭크의 활용

- 미지수가 n개인 연립일차방정식의 근(해)과 계수(Rnank)와의 관계
$$
\begin{cases}
x_1 - x_2 = x_3 = 1\\
2x_1 + x_2 -5x_3 =0
\end{cases}
\iff
\begin{pmatrix} 1 & -1 & -1 \\ 2 & 1 & 5 \end{pmatrix}
\begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} = 
\begin{pmatrix} 1 \\ 0 \end{pmatrix}
=
\begin{pmatrix} 1 & -1 & -1 & | & 1 \\ 2 & 1 & 5 & | & 0 \end{pmatrix}
$$

- 여기서 `|(bar)`를 기준으로 왼쪽만 있는 행렬을 **계수행렬(A)**라고 하고, 오른쪽까지 포함한 것을 **확대행렬($A\|B$)**이라고 한다.


## 랭크의 활용공식
- 일차연립방정식의 계수행렬을 $A$, 확대행렬을 $A\|B$라고 하자.

1. $rankA < rank(A\|B)$이면 근(해)이 존재하지 않는다.
2. $rankA = rank(A\|B) = n$ 이면 근(해)이 오직 하나만 존재한다.
  - = 유일한 해를 가진다.
3. $rankA = rank(A\|B) < n$ 이면 무수히 많은 근(해)을 갖는다. 
  - = ~이외의 해를 가진다.

- 만약 연립방정식의 모든 값이 `= 0`으로 끝면 **선형연립방정식**이다. 이 경우, $rankA$와 $rankA\|B$의 $rank$는 같다.
- $A: n \times n$행렬(암기)
  - 1) $\|A\| \neq 0 = rankA \iff n$
  - 2) $\|A\| = 0 rankA \iff n$


# 벡터
움직임을 표현하기 위해 만든 연산
- 어떤 **방향**을 제시해줘야 한다.
- 어떤 **힘**을 제시해줘야 한다.
$$\text{벡터} \Longleftarrow \text{방향} + \text{힘(크기)} \text{이 제시된 수학적인 연산}$$

1. 표현
벡터: $\vec{a}$ 
벡터의 크기: $\| \vec{a} \|$

2. 벡터의 상등
- 벡터는 크기와 방향만 동일하다면, 위치는 크게 상관이 없다. 즉, 이동이 가능하다.
- 왜냐하면 벡터는 오로지 **크기와 방향**만 갖기 때문이다.


3. 벡터의 실수배
$$k\vec{a} \quad -k\vec{a}$$
- 방향은 그대로 갖고 있지만 **힘의 크기**만 늘거나 줄어든다.
- 음수이면 **방향**이 반대로 바뀐다.


4. 단위 벡터
  - 크기가 1인 벡터를 의미한다.
  - 벡터 a의 단위벡터는 벡터a의 크기 $\times$ 벡터 a이다.

$$\vec{a} \text{의 단위벡터} = \frac{1}{|\vec{a}|} \vec{a}$$


5. 벡터의 덧셈
$$\vec{a} + \vec{b}$$
- 최단거리로 첫 점부터 끝점까지 간 거리로 볼 수 있다.

6. 벡터의 뺄셈
- 어디애서 어디로 가는 지에 따라 식이 달라진다.
- $\vec{a} - \vec{b}$와 \vec{b} - \vec{a}의 방향이 달라진다.

> 자세한 부분은  [유튜브 수업](https://www.youtube.com/playlist?list=PLxMkK1K0XECOj2sZG-gCk-CjvZhJ_75I4)이 **벡터**강의를 참조 바람


## 벡터의 좌표계
- 선형대수학에서는 점인지 벡터인지를 민감하게 받아들여야 한다.

### 벡터의 연산
- 아래에서 원점(0, 0)을 'O'라고 하고, 시작점이 원점인 것을 **위치벡터**라고 한다.
$$
\begin{matrix}
\vec{a} = \vec{OA} = (a_1, b_1) \\
\vec{b} = \vec{OB} = (a_2, b_2)
\end{matrix} \text{일 때, } \quad
\begin{matrix}
\vec{a} + \vec{b} = (a_1 + a_2, b_1 + b_2) \\
\vec{a} - \vec{b} = (a_1 - a_2, b_1 - b_2)
\end{matrix} 
$$

### 실수배 연산
$$
\begin {matrix}
\vec{a} = \vec{OA} = (a_1, b_1) \\
k\vec{a} = (ka_1, kb_1)
\end{matrix}
$$

### 점에서 점으로의 이동
$$
\begin {matrix}
\vec{a} = \vec{OA} = (a_1, b_1) \\
\vec{b} = \vec{OB} = (a_2, b_2) \\
\vec{AB} = -\vec{a} + \vec{b} = \vec{b} - \vec{a} = (a_2 - a_1, b_2 - b_1) \\
\vec{BA} = -\vec{b} + \vec{a} = \vec{a} - \vec{b} = (a_1 - a_2, b_1 - b_2)
\end{matrix}
$$

- 점 A에서 점 B로 가는 벡터 ($\vec{AB}$)는 점 B의 좌표에서 점 A의 좌표를 뺀다.
- 즉, **점에서 점을 만드는 벡터는 뒷점에서 앞점을 뺀다**.
- 이는 공간상(3차원)으로 늘어도 연산이 같다.

### 기본단위 벡터
- 벡터를 다음과 같이 표시할 수도 있다.
- 2차원
$$
\begin{matrix}
\vec{a} & = & (a_1, a_2) \\
        & = & a_1i + a_2j \\
        & = & a_i(1, 0) + a_2(0, 1)
\end{matrix}
$$

- 3차원
$$
\begin{matrix}
\vec{a} & = & (a_1, a_2, a_3)  \\
& = & a_1i + a_2j + a_3k
\end{matrix}
$$

### 벡터의 크기
- 2차원
$$
\begin{matrix}
\vec{OA} = (a,b) \\
|\vec{OA}| =  \sqrt{a^2 + b^2}
\end{matrix}
$$

- 3차원
$$
\begin{matrix}
\vec{OA} = (a,b,c) \\
|\vec{OA}| =  \sqrt{a^2 + b^2 + c^2}
\end{matrix}
$$

# 벡터의 내적
## 내적의 정의
두 벡터 $\vec{a} = (a_1, a_2, a_3), \vec{b} = (b_1, b_2, b_3)$의 사잇각 $\theta (0 \leq \theta \pi)$일 때, 
$$\vec{a} \circ \vec{b} = |\vec{a}| |\vec{b}| cos\theta = a_1 b_1+a_2 b_2+a_3 b_3$$
$$cos\theta \frac{\vec{a} \circ \vec{b}}{|\vec{a}||\vec{b}|}$$

## 내적의 성질
1. $\vec{a} \circ \vec{b}  = \vec{b} \circ \vec{a}$
2. $\vec{a}(\vec{b}+ \vec{c}) = \vec{a} \circ \vec{b} + \vec{a} \circ \vec{c}$
3. $m(\vec{a} \circ \vec{b}) = (m\vec{a}) \circ \vec{b} = \vec{a}(m \vec{b})$
  - 실수m은 a나 b 중 하나에만 곱해야 한다.
4. $\vec{a} \perp \vec{b} \iff \vec{a} \circ \vec{b} = 0$
  - 백터 a와 벡터 b의 내적은 0이면 직교한다.
  - 직교하면 백터 a와 벡터 b의 내적은 0이다.
5. $\vec{a} \circ \vec{a} = \|\vec{a}^2\|$

## 정사영 벡터
- $\vec{a}$를 $\vec{b}$에 투영(Proj)면 다음과 같다.
$$Proj_{\vec{b}}\vec{a} = \frac{\vec{a} \circ \vec{b}}{\vec{b} \circ \vec{b}}\vec{b}$$

- 누구의 방향으로 투영(Projection)되냐에 따라서 공식에서 값의 비중이 높아지는 지가 정해진다고 생각하면 된다.