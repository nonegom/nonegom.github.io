---
title:  "[머신러닝] 데이터 전처리 (이미지)  - 5. 이미지 특징 추출"
excerpt: "이미지에서 edge, coner, decriptor 등을 추출하는 알고리즘 - SIFT, SURF, FAST, ORB"

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

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.02.05%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%ED%8A%B9%EC%A7%95%20%EC%B6%94%EC%B6%9C.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 0. 이미지 특징 추출
이미지에서 라인, 코너, 특징점 등과 같은 특징 추출(feature extraction) 방법을 공부한다. Bag of Visual Words: VisualWords에 해당하는 게 '이미지 특징'이다.

## 1. 이미지 미분
가로/세로 위치 변화에 따른 **픽셀 값의 변화율**을 **이미지의 도함수**(image derivatives)라고 한다. $x$위치의 픽셀 명도를 $f(x$)라고 했을 때$g_x, g_y$의 값은 아래와 같다.
$$g_x = \frac{\partial f}{\partial x}, g_y = \frac{\partial f}{\partial y}$$

x, y 방향의 도함수의 크기(Intensity)를 구하는 것이 **라플라스 연산**이다. 등고선의 길이는 각도에 비례하는데, 이 각도를 Intensity라고 한다.
$$g = \sqrt{g_x^2 + g_y^2}$$

실제 이미지 프로세싱에서는 중앙차분법(central difference)을 사용한다.
$$G(x) = f(x+1) - f(x-1) \approx [-1 \quad 0 \quad 1]$$

이 연산은 다음 **이미지 커널를 사용하여 필터링**한 것과 같다.
$$ k=[1,\ 0,\ −1]$$

## 2. 엣지 추출
경계선을 인지하는 것을 엣지 추출(edge detection)이라고 한다. 엣지는 이미지 안에서 픽셀의 값이 갑자기 변하는 곳이다. 

엣지 추출 알고리즘은 이미지를 미분한 **그레디언트 벡터의 크기로 판단**한다. 대표적인 알고리즘으로 'Sobel edge Detection'rhk 'Canny edge Detection'이 있다. 

OpenCV에서는 `Sobel`,`Laplacian` 명령을 사용해서 엣지 추출을 할 수 있다.

```py
from skimage.data import text
import cv2

img = text()

sobelx = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3)
sobely = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3)
laplacian = cv2.Laplacian(img, cv2.CV_8U)

############# 시각화 코드 생략 ###############
```

![](/assets/images/Preprocessing10_1.png)

### 캐니 엣지 추출

캐니 엣지 추출법은 그레디언트의 크기 뿐 아니라 **방향도 사용**한다. 국부 최대값 근처의 그레디언트값을 제
거하는 Nonmaximal suppression을 이용하여 **가짜 엣지를 제거**한다. 마지막으로 **두 개의 기준값(threshold)을 사용**해서 엣지를 구분한다. 조금 더 좋은 성능으로 윤곽을 구할 수 있다.

아래 코드는 `canny` 함수를 사용하여 이미지의 엣지를 추출한다. 기준값으로는 50과 100을 사용하였다.

```py
img = text()
edges = cv2.Canny(img, 50, 100)

############# 시각화 코드 생략 ###############
```

![](/assets/images/Preprocessing10_2.png)

## 3. 코너 추철

코너(Corner)는 엣지가 교차되는 점을 얘기한다.

### 해리스 코너 추출

해리스 코너 추출 함수 `cornerHarris`는 이미지 위에 **커널을 이동**시키면서 그레디언트의 변화량을 구하고 이 값을 적절한 기준값으로 구별하여 **코너점을 찾는다**. 

이 부분이 나중에 Visual Words의 Words 부분이 된다. 다시 말하자면 '점'이 그림의 단어가 되는 것이다.

```py
dst = cv2.cornerHarris(img, 2, 3, 0.24)

# thresholding
ret, dst = cv2.threshold(dst, 0.001 * dst.max(), 1, 0)

x, y = np.nonzero(dst)

############# 시각화 코드 생략 ###############
```

![](/assets/images/Preprocessing10_3.png)

### Shi & Tomasi 코너 추출
Shi & Tomasi 코너 추출법은 더 작은 변화를 보이는 방향의 변화량이 설정한 기준값보다 크면 코너라고 판단한다. OpenCV에서 `goodFeatureToTrack`이라는 메서드로 구현되어 있다. 역시 위의 코너 추출 방법보다 더 좋은 성능을 보여준다.

- `goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)`

```py
corners = cv2.goodFeaturesToTrack(img, 100, 0.001, 40)
corners = np.int0(corners)

############# 시각화 코드 생략 ###############
```

![](/assets/images/Preprocessing10_4.png)

## 4. 허프 라인 변환
허프 라인 변환(Hough line transformation)을 사용하면, 검출된 **엣지로부터 직선을** 이루는 부분을 찾아낼 수 있다. 

이미지에서 한 좌표 ($x_1,y_1$) 가 주어졌을 때, 그 좌표를 지나는, 기울기 $m$ 과, $y$ 절편 $c$ 를 가지는 직선의 방정식 $y_1 = mx_1+c$를 정의할 수 있다. 이 때, 이 직선이 많은 엣지 좌표들을 지난다면, 그것이 이미지에 있는 직선일 확률이 높다. 이 직선을 구하기 위해 다양한 방법이 있을 수 있는데,  
**Hough Line 변환**에서는 주어진 좌표를 지나는 직선을 1도 단위로 180도까지 회전시켜 180개의 2차원 배열을 만든다. 그리고 모든 점에 대해 이 연산을 수행해, 기준 값 이상의 점들을 지나는 직선의 방정식을 구한다. 그 후 원점과 직선 사이의 거리 $r$과 각도 $\theta$값을 구한다면 이미지에서의 직선일 확률이 높은 직선의 방정식을 찾을 수 있다.  

이 방법론은 타원, 사각형, 원을 추출할 때도 사용될 수 있다. 
  
OpenCV에서는 이 알고리즘을 `HoughLine`이라는 함수로 구현해 놓았다.

- 필요한 파라미터 값 
    - 검출한 **엣지**
    - r값을 정의하는 **픽셀 수**, 보통 1픽셀을 사용한다.
    - θ값을 정의하는 **단위**, 보통 1도를 사용한다. r을 1로 설정하면 해당 파라미터는 π/180 으로 설정한다.
    - **기준값(threshold)**, 직선이 교차하는 점의 최소 갯수를 의미한다.

이렇게 입력하면 **엣지로 부터 직선을 추출**할 수 있다.

HoughLine 방법은 **연산량이 너무 많기 때문**에 보통은 확률적 Hough Transfomation 방법을 사용한다. 확률적 `Hough Transfomation` 방법은 모든 점이 아닌 임의의 점에 대해 ($r,θ$)를 계산하는 것이다. `HoughLineP`라는 이름으로 구현되어 있다. 이때는 파라미터로 `minLineLenght`(직선의 최소 길이), `maxLineGap`(같은 선 내에 있는 점들의 최소 간격)이 추가된다.

아래 코드는 **Canny Detection** 방법으로 엣지를 추출하고, 추출한 엣지로 부터 `HoughLineP` 함수를 사용해 **직선을 추출**한다.

```py
from skimage.data import checkerboard

img = checkerboard()
rows, cols = img.shape

pts1 = np.float32([[25, 25], [175, 25], [25, 175], [175, 175]])
pts2 = np.float32([[25, 50], [150, 25], [75, 175], [175, 125]])

H = cv2.getPerspectiveTransform(pts1, pts2)
img = cv2.warpPerspective(img, H, (cols, rows))

edges = cv2.Canny(img, 100, 150, apertureSize=3)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 50, 8)

############# 시각화 코드 생략 ###############
```

![](/assets/images/Preprocessing10_5.png)

## 5. 이미지 히스토그램
이미지 히스토그램은 이미지에서 **특정 밝기 구간에 들어가는 픽셀의 수**를 나타낸 것이다. 히스토그램은 데이터 집합에서 특정 구간의 값을 가지는 데이터의 개수를 세어 나타낸 것이다. 

여기서 특정 구간의 값은 **명도**를 얘기한다.

```py
from skimage.data import camera
img = camera()

hist = cv2.calcHist([img], [0], None, [256], [0, 256])

plt.subplot(121)
plt.imshow(img, "gray")
plt.axis("off")
plt.subplot(122)
plt.plot(hist, color='r')
plt.xlim([0, 256])
plt.tight_layout()
plt.show(
```

![](/assets/images/Preprocessing10_6.png)

- y축은 개수, x축은 밝기

## 6. 그레디언트 히스토그램 설명자

**특징점의 주변 특성**을 이용해 해당 특징점을 표현하는 벡터를 만들어 이미지에서 같은 특징점을 매칭하거나 추출할 때 사용한다. 이를 **설명자(descriptor)**라고 한다. 방법은 다음과 같다.

1. 특징점을 중심으로 16x16 영역을 4x4 크기의 16개 윈도우로 나눈다.

2. 윈도우의 16개 포인트에서 그레디언트 **벡터의 크기와 방향을 계산**한다.

3. 그레디언트 벡터의 방향을 8개의 각도로 **라운딩(rouding)**한다.

4. 8개의 각도에 대해 그레디언트 벡터의 크기를 더하여 일종의 그레디언트 히스토그램을 만든다.

5. 윈도우 16개의 히스토그램을 모두 모으면, 특징점 주변에 대한 정보가 128(8 x 16)차원의 벡터로 표현된다. 
    - 128개의 숫자로 표현되는 어떤 값이 나타나는데, 그 값이 하나의 단어가 된다.


## 7. SIFT
SIFT(Scale-Invariant Feature Transform)은 **특징점의 크기와 각도까지 같이 계산**하여 이미지의 크기가 변하거나 회전해도 **동일한 특징점을 찾을 수 있도록 하는 방법**이다. 또한 특징점 근처의 이미지 특성(히스토그램)도 같이 계산해서 특징점 이미지의 모양도 구별할 수 있도록 한다.

아래 코드는 이미지에 대해 `SIFT`특징을 찾고, 변환된 이미지에서 같은 특징점끼리 매칭하는 작업을 수행한다.

```py
## 0. 이미지 로드
from skimage.data import camera
img = camera()

## 이미지 변환
rows, cols = img.shape
H = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 0.7)
img_rotated = cv2.warpAffine(img, H, (cols, rows)) # 이미지 회전

## sift특징 생성
sift = cv2.xfeatures2d.SIFT_create()
kps, des = sift.detectAndCompute(img, None)
kps_r, des_r = sift.detectAndCompute(img_rotated, None)
kp0 = kps[0]
print("pt=({},{}), size={}, angle={}".format(kp0.pt[0], kp0.pt[1], kp0.size, kp0.angle))

# > pt=(2.347715139389038,486.962890625), size=2.006916046142578, angle=234.960693359375
```

```py
bf = cv2.BFMatcher_create()
matches = bf.knnMatch(des, des_r, k=2)

good = []
for m, n in matches:
    if m.distance < 0.4 * n.distance:
        good.append([m])

np.random.shuffle(good)
image_match = cv2.drawMatchesKnn(
    img, kps, img_rotated, kps_r, good[:10], flags=2, outImg=img)

pts_x = [kp.pt[0] for kp in kps]
pts_y = [kp.pt[1] for kp in kps]
pts_s = [kp.size for kp in kps]

## 그래프 시각화 코드
plt.imshow(img, cmap="gray")
plt.title("SIFT 특징점")
plt.axis("off")
plt.scatter(pts_x, pts_y, s=pts_s, c="w")
plt.show()
```

![](/assets/images/Preprocessing10_7.png){: width="500" height="500"}{: .align-center}

```py
plt.bar(np.arange(len(des[0])), des[0])
plt.xticks(range(0, len(des[0]), 8))
plt.yticks([des[0].min(), des[0].mean(), des[0].max()])
plt.title("첫번째 특징점의 설명자")
plt.show()
```

![](/assets/images/Preprocessing10_8.png){: width="600" height="300"}{: .align-center}

- x축은 벡터값을 의미한다.

```py
plt.imshow(image_match)
plt.title("SIFT 특징점 매칭")
plt.axis("off")
plt.show()
```
![](/assets/images/Preprocessing10_9.png)

## 8. SURF
SURF(Speeded-Up Robust Features)는 인텐서티 계산 방법을 간략화 하는 등의 방법으로 `SIFT` 방법의 속도와 안정성을 개선한 것이다. 아래 코드는 위에서 `SIFT`방법으로 수행한 동일한 작업을 `SURF`방법으로 수행한다.

```py
surf = cv2.xfeatures2d.SURF_create(400)
kps, des = surf.detectAndCompute(img, None)
kps_r, des_r = surf.detectAndCompute(img_rotated, None)

kp0 = kps[0]
print("pt=({},{}), size={}, angle={}".format(kp0.pt[0], kp0.pt[0], kp0.size, kp0.angle))
# > pt=(222.9388885498047,222.9388885498047), size=17.0, angle=343.84210205078125
```

```py
bf = cv2.BFMatcher_create()
matches = bf.knnMatch(des, des_r, k=2)

good = []
for m, n in matches:
    if m.distance < 0.4 * n.distance:
        good.append([m])

np.random.shuffle(good)
image_match = cv2.drawMatchesKnn(
    img, kps, img_rotated, kps_r, good[:10], flags=2, outImg=img)

pts_x = [kp.pt[0] for kp in kps]
pts_y = [kp.pt[1] for kp in kps]
pts_s = [kp.size / 10 for kp in kps]

plt.imshow(img, cmap='gray')
plt.title("SURF 특징점")
plt.axis("off")
plt.scatter(pts_x, pts_y, s=pts_s, c="w")
plt.show()
```

![](/assets/images/Preprocessing10_10.png){: width="500" height="500"}{: .align-center}

```py
plt.bar(np.arange(len(des[0])), des[0])
plt.xticks(range(0, len(des[0]), 8))
plt.yticks([des[0].min(), des[0].mean(), des[0].max()])
plt.title("첫번째 특징점의 설명자")
plt.show()
```

![](/assets/images/Preprocessing10_11.png){: width="600" height="300"}{: .align-center}

```py
plt.imshow(image_match)
plt.title("SURF 특징점 매칭")
plt.axis("off")
plt.show()
```

![](/assets/images/Preprocessing10_12.png)

> SIFT방법과 특징점에서 차이가 존재함을 알 수 있다.

## 9. FAST

FAST(Features from Accelerated Segment Test)도 **코너를 찾는 알고리즘**이다. 이름처럼 빠른 연산으로 유명하다. 

코너성을 대변하는 수치를 계산하고 이를 기준으로 인접 픽셀에 대해 가장 높은 코너성 수치를 가지는 픽셀만 코너로 선택하는 것이다.

![](/assets/images/Preprocessing10_13.png){: width="500" height="500"}{: .align-center}

## 10. ORB
ORB(Oriented FAST and Rotated BRIEF)는  FAST 와 BRIEF를 기반으로 만들어진 알고리즘이다.  

SIFT에서 하나의 특징점에 대한 정보(설명자)는 128차원의 실수 벡터를 가지기 떄문에 꽤 많은 메모리를 사용한다. 이러한 잠재적인 리소스 낭비 방지를 위해  BRIEF는 설명자 벡터를 특징점의 픽셀값을 기준으로 0, 1, 이진 값으로 나타낸다. 이때 BRIEF는 설명자 표현법이다.

이미지에 대해 `ORB` 특징을 찾고, 변환된 이미지에서 같은 특징점 끼리 매칭하는 작업이 수행된 이미지이다.

```py
orb = cv2.ORB_create()
kps, des = orb.detectAndCompute(img, None)
kps_r, des_r = orb.detectAndCompute(img_rotated, None)

bf = cv2.BFMatcher_create()
matches = bf.knnMatch(des, des_r, k=2)

good = []
for m, n in matches:
    if m.distance < 0.4 * n.distance:
        good.append([m])

np.random.shuffle(good)
image_match = cv2.drawMatchesKnn(
    img, kps, img_rotated, kps_r, good[:10], flags=2, outImg=img)

pts_x = [kp.pt[0] for kp in kps]
pts_y = [kp.pt[1] for kp in kps]
pts_s = [kp.size / 10 for kp in kps]


plt.imshow(img, cmap='gray')
plt.title("ORB 특징점")
plt.axis("off")
plt.scatter(pts_x, pts_y, s=pts_s, c="w")
plt.show()
```
![](/assets/images/Preprocessing10_14.png){: width="500" height="500"}{: .align-center}

```py
plt.imshow(image_match)
plt.title("ORB 특징점 매칭")
plt.axis("off")
plt.show()
```

![](/assets/images/Preprocessing10_15.png)