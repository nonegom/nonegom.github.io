---
title:  "[머신러닝] 데이터 전처리 (이미지)  - 4. 이미지 변환"
excerpt: "이미지 변환을 위한 배경지식과 변환의 종류 - 어파인 변환, 강체변환, 유사변환, 3점 어파인 변환, 원근 변환"

categories:
  - ML_PreProcessing
tags:
  - PreProcessing
  - 11월
toc: true
toc_sticky: true
toc_label: 페이지 목차
use_math: true
---

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.02.04%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EB%B3%80%ED%99%98.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 0. 동치좌표와 어파인 변환
동치 좌표는 2차원 벡터의 이미지를 3차원 벡터로 표현하는 방법이다. 아래와 같이 마지막 원소로 1을 추가해 만든다. 이미지 변환에 관한 수식을 이해하기 위해서는 **동치좌표**의 개념을 알고 있어야 한다.
$$
x = \begin{bmatrix}
x \\ y \\1
\end{bmatrix}
$$

마지막 원소가 1이 아닌 경우에는 마지막 원소를 1로 스케일링 시킨 것과 동일한 위치를 가리키게 된다. 
$$
\begin{bmatrix}
x \\ y \\w
\end{bmatrix} = \begin{bmatrix}
x/w \\ y/w \\1
\end{bmatrix}
$$

**변환행렬**은 크기를 늘이거나 줄일 수 있고, 회전을 할 수 있지만, 원점을 움직일 수는 없다. 따라서 원점을 움직이기 위해서는 $t$값을 더해야 한다. 
따라서 변환행렬을 곱해 **선형변환**을 하고, 원점을 움직이기 위해 **평행하게 이동**할 수 있도록 $t = (t_x, t_y)$만큼 이동하는 과정을 거친다.
$$\begin{split} \begin{bmatrix}x' \\ y'\end{bmatrix} = 
{A}\begin{bmatrix}x \\ y\end{bmatrix} + \begin{bmatrix}t_x \\ t_y\end{bmatrix}= {A}\begin{bmatrix}x \\ y\end{bmatrix} + t \end{split}$$

하지만 동치좌표를 사용하면 위의 과정을 한 번의 행렬 곱으로 나타낼 수 있기 때문에 수식과 연산이 간단해진다. 이 수식을 이용해 이미지 변환하는 것을 **어파인 변환**이라고 한다.
$$\begin{split} \begin{bmatrix}x' \\ y' \\ 1\end{bmatrix}= \begin{bmatrix} {A} & {t} \\ {0} &  1\end{bmatrix}\begin{bmatrix}x \\ y \\ 1\end{bmatrix}\end{split}$$

이렇게 하나의 3×3 행렬로 3차원 좌표의 변환을 할 때, 이 **3×3 행렬을 사영행렬**(homography matrix)이라고 한다. (3x3행렬인 이유는, 변환행렬 A가 2x2행렬이고, $t$는 $t_x$와 $t_y$ 2개를 가지는 벡터이기 때문이다.)
$$x' = Hx$$

## 1. 강체변환 (유클리드 변환)
강체라는 뜻은 '강한 물체'를 의미한다. 강한 물체는 크기가 변하지 않기 마련이다. 따라서 강체변환(rigid transform)의 경우 크기를 변화시키지 않고, **회전($\theta$)과 이동($t$)** 두 가지 요소만 사용해서 이미지를 변환한다. 사영행렬(homography matrixm)는 아래와 같다.
$$H = 
\begin{bmatrix} 
\text{cos}\theta & -\text{sin}\theta &t_x\\
\text{sin}\theta & \text{cos}\theta &t_y \\
0 & 0 & 1
\end{bmatrix}
$$

## 2. 유사변환
유사변환(similarity transform) **확대/축소($s$), 회전($\theta$), 이동($t$)** 세가지 요소를 사용해 이미지를 변환한다. 사영행렬은 아래와 같다.
$$H = 
\begin{bmatrix} 
{s}\ \text{cos}\theta & -{s}\ \text{sin}\theta &t_x\\
{s}\ \text{sin}\theta & {s}\ \text{cos}\theta &t_y \\
0 & 0 & 1
\end{bmatrix}
$$

### 변환행렬 만들기
변환행렬을 만들 때는 `cv2.getRotationMatrix2D` 함수를 사용한다.
- `getRotationMatrix2D(center, angle, scale)`
    - `center`: 이미지의 중심 좌표  / 평행이동할 위치
    - `angle`: 회전시키는 각도 $\theta$ (시계 반대방향) / 몇 도를 회전할 것안가 
    - `scale`: 변환하려는 크기 비율 $s$ / 몇 배를 확대/축소할 것인가

- 사영행렬이 이론적으론 3X3행렬이지만 OpenCV에서는 마지막 행렬을 생략해서 3X2 변환행렬을 사용할 수도 있다.

변환행렬을 실제로 이미지에 적용하여 **어파인 변환**을 할 때는 `warpAffine` 함수를 사용한다.

```py
import cv2
import skimage.data

img_astro = skimage.data.astronaut()
img = cv2.cvtColor(img_astro, cv2.COLOR_BGR2GRAY)
rows, cols = img.shape[:2]

# 이미지의 중심점을 기준으로 30도 회전, 크기는 70%
H = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 0.7)

# 100만큼 평행이동
H2 = H.copy()
H2[:, 2] += 100
H2
'''
array([[  0.60621778,   0.35      ,  61.20824764],
       [ -0.35      ,   0.60621778, 240.40824764]])
'''
```

```py
dst = cv2.warpAffine(img, H, (cols, rows))
dst2 = cv2.warpAffine(img, H2, (cols, rows))

fig, [ax1, ax2, ax3] = plt.subplots(1, 3, figsize=(13, 13)) # what is?
ax1.set_title("Original")
ax1.axis("off")
ax1.imshow(img, cmap=plt.cm.gray)

ax2.set_title("Rigid transformed")
ax2.axis("off")
ax2.imshow(dst, cmap=plt.cm.gray)

ax3.set_title("similarity transformed")
ax3.axis("off")
ax3.imshow(dst2, cmap=plt.cm.gray)
plt.show()
```

![](/assets/images/Preprocessing9_1.png)

## 3. 3점 어파인 변환
**어파인 변환**에 사용되는 3점이 어떻게 변환되는지만 알면 사용할 수 있는 방법이다. OpenCV에는 주어진 두 쌍의 3점으로부터 어파인 변환을 게산하는 `getAffineMatrix` 함수를 제공한다.

```py
# 이미지 로드
img = sp.misc.face()
rows, cols, ch = img.shape

# 3점 생성 - 
pts1 = np.float32([[200, 200], [200, 600], [800, 200]])
pts2 = np.float32([[100, 100], [200, 500], [600, 100]])

pts_x1, pts_y1 = zip(*pts1)
pts_x2, pts_y2 = zip(*pts2)

# 3점 어파인 변환 - 위 점(pst1)이 아래의 점(pst2)으로 바뀌는 H행렬을 구한다.
H_affine = cv2.getAffineTransform(pts1, pts2)
```

```py
# 위에서 구한 affine행렬을 적용
img2 = cv2.warpAffine(img, H_affine, (cols, rows))

fig, [ax1, ax2] = plt.subplots(1, 2)

# 시각화 코드(점(scatter) 및 선(plot) 그리기)
ax1.set_title("Original")
ax1.imshow(img)
ax1.scatter(pts_x1, pts_y1, c='w', s=100, marker="s")
ax1.scatter(pts_x2, pts_y2, c='w', s=100)
ax1.plot(list(zip(*np.stack((pts_x1, pts_x2), axis=-1))),
         list(zip(*np.stack((pts_y1, pts_y2), axis=-1))), "--", c="w")
ax1.axis("off")

ax2.set_title("Affine transformed")
ax2.imshow(img2)
ax2.scatter(pts_x2, pts_y2, c='w', s=100)
ax2.axis("off")

plt.tight_layout()
plt.show()
```

![](/assets/images/Preprocessing9_2.png)

> 사각형의 점(■)을 동그란 점(●)의 방향으로 민다. 그러면서 위쪽으로 쏠리는 모양이 된다.


## 4. 원근변환
원근변환(perspective transform)은 직선의 성질만 유지가 되고, 선의 평행성은 유지가 되지 않는 변환으로 2차원 이미지를 변환하는 방법이다. 원근변환을 지정하는데는 4점이 필요하다. 주어진 두 쌍의 4점으로부터 원근변환을 위한 사영행렬을 계산하는 `getPerspectiveTransform` 함수를 제공한다. 실제 변환에는 `warpPerspective`함수를 사용한다.

```py
# 4점 생성
pts1 = np.float32([[200, 200], [200, 600], [800, 200], [800, 600]])
pts2 = np.float32([[300, 300], [300, 500], [600, 100], [700, 500]])

# 사영행렬 계산
H_perspective = cv2.getPerspectiveTransform(pts1, pts2)

# 이미지 변환
img2 = cv2.warpPerspective(img, H_perspective, (cols, rows))

fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(15, 15))

pts_x, pts_y = zip(*pts1)
pts_x_, pts_y_ = zip(*pts2)

# 시각화 코드
ax1.set_title("Original")
ax1.imshow(img, cmap=plt.cm.bone)
ax1.scatter(pts_x, pts_y, c='w', s=100, marker="s")
ax1.scatter(pts_x_, pts_y_, c='w', s=100)
ax1.plot(list(zip(*np.stack((pts_x, pts_x_), axis=-1))),
         list(zip(*np.stack((pts_y, pts_y_), axis=-1))), "--", c="w")
ax1.axis("off")

ax2.set_title("Perspective transformed")
ax2.imshow(img2, cmap=plt.cm.bone)
ax2.scatter(pts_x_, pts_y_, c='w', s=100)
ax2.axis("off")
plt.show()
```

![](/assets/images/Preprocessing9_3.png)

> 사각형의 점(■)을 동그란 점(●)의 방향으로 민다. 

## 5. 연습문제

원근변환을 사용해서 배경없이 영수증 부분만으로 채운 이미지를 만들어라. 

> 이전 게시글인 [[머신러닝] 데이터 전처리 (이미지) - 2. 이미지 필터링]({% post_url 2020-11-12-Preprocessing08%})에서 사용했던 연습문제 코드를 이용

```py
import cv2

# 0. 이미지 로드
img = cv2.imread("./receipt.png")
img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img_raw.copy().astype('uint8')

# 1. 이미지 가공
# 임계처리
maxval = 255
thresh = 200
_, img = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)

# 중앙값 처리
img = cv2.medianBlur(img, 3)

#closing
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(
                               cv2.MORPH_ELLIPSE, (40, 40)))                                                      
```
```py
# 2. 가공한 이미지에서 모서리 값 알아낸 후 표시
contours, hierachy = cv2.findContours(
    img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
c0 = contours[0]

leftmost = tuple(c0[c0[:, :, 0].argmin()][0])
rightmost = tuple(c0[c0[:, :, 0].argmax()][0])
topmost = tuple(c0[c0[:, :, 1].argmin()][0])
bottommost = tuple(c0[c0[:, :, 1].argmax()][0])
```

> 위는 이전 게시물에서 나왔던 코드 생략 가능 여기부터 **원근 반환 적용**

```py
#3. 원근뱐환
rows, cols = img.shape[:2]
pts1 = np.float32([list(leftmost), list(rightmost), list(topmost), list(bottommost)])
pts2 = np.float32([[0,0], [w, h], [w, 0], [0, h]])
# 이미지에서 왼쪽 상단(11시 방향)이 (0,0)이므로 leftmost는 (0, 0) 이다. 

H_perspective = cv2.getPerspectiveTransform(pts1, pts2)
```
```py
# 원근반환이 적용된 이미지 생성
img2 = cv2.warpPerspective(img_raw, H_perspective, (cols, rows))

# 시각화 코드 (imshow 코드만 실행해도 됨)
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(15, 15))
pts_x, pts_y = zip(*pts1)
pts_x_, pts_y_ = zip(*pts2)

ax1.set_title("Original")
ax1.imshow(img_raw, cmap=plt.cm.bone)
ax1.scatter(pts_x, pts_y, c='w', s=100, marker="s")
ax1.scatter(pts_x_, pts_y_, c='w', s=100)
ax1.plot(list(zip(*np.stack((pts_x, pts_x_), axis=-1))),
list(zip(*np.stack((pts_y, pts_y_), axis=-1))), "--", c="w")
ax1.axis("off")

ax2.set_title("Perspective transformed")
ax2.imshow(img2, cmap=plt.cm.bone)
ax2.scatter(pts_x_, pts_y_, c='w', s=100)
ax2.axis("off")
plt.show()
```

![](/assets/images/Preprocessing9_4.png)