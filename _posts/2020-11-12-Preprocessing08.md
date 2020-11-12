---
title:  "[머신러닝] 데이터 전처리 (이미지)  - 3. 이미지 컨투어"
excerpt: "이미지의 경계선 정보를 나타내는 컨투어(contour) - findContours, drawContours, 컨투어 추정"

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

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.02.03%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EC%BB%A8%ED%88%AC%EC%96%B4.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  


## 1.이미지 컨투어

컨투어(contour)란 동일한 색 또는 동일한 픽셀값(강도)을 가지고 있는 **영역의 경계선 정보**다. 물체의 **윤곽선, 외형을 파악**하는데 사용된다.

`OpenCV`의 `findContours`함수로 **이진화된 이미지와 이미지의 컨투어 정보, 컨투어 상하구조(hierachy) 정보**를 출력한다. 흑백 이미지 또는 이진화된 이미지만 적용할 수 있다.  

```py
images, contours, hierachy = cv2.findContours(image, mode, method)
```
- `image`: 흑백이미지 또는 이진화된 이미지

- `mode` : 컨투어를 찾는 방법 (디폴트 모드 사용)

    - cv2.RETR_EXTERNAL: 컨투어 라인 중 **가장 바깥쪽의 라인**만 찾음
    - cv2.RETR_LIST: 모든 컨투어 라인을 찾지만, **상하구조(hierachy)관계를 구성하지 않음**
    - cv2.RETR_CCOMP: 모든 컨투어 라인을 찾고, **상하구조는 2 단계**로 구성함
    - **cv2.RETR_TREE**: 모든 컨투어 라인을 찾고, **모든 상하구조를 구성**함  **# 제일 많이 사용하는 모드.**

- `method` : 컨투어를 찾을 때 사용하는 **근사화 방법**
    - cv2.CHAIN_APPROX_NONE: 모든 컨투어 포인트를 반환
    - cv2.CHAIN_APPROX_SIMPLE: 컨투어 라인을 그릴 수 있는 포인트만 반환
    - cv2.CHAIN_APPROX_TC89_L1: Teh_Chin 연결 근사 알고리즘 L1 버전을 적용하여 컨투어 포인트를 줄임
    - cv2.CHAIN_APPROX_TC89_KCOS: Teh_Chin 연결 근사 알고리즘 KCOS 버전을 적용하여 컨투어 포인트를 줄임

```py
import cv2
from skimage.data import horse

img_raw = horse().astype('uint8')
img_raw = np.ones(img_raw.shape) - img_raw

img = img_raw.copy().astype('uint8')

images, contours, hierachy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
```

### 컨투어 정보

컨투어(contour) 정보는 **컨투어를 구성하는 점들로 이루어진 배열의 리스트**이다. 리스트의 원소의 개수는 컨투어의 개수와 같다.

```py
# 2 개의 컨투어 라인이 잡혔다.
len(contours)
## > 2

# 컨투어의 모양
contours[0].shape
## > (312, 1, 2)

# 컨투어 라인이 이어져있는 좌표정보 (중 일부)
np.squeeze(contours[0])[:5]
'''
array([[350,   9],
       [346,  13],
       [345,  13],
       [339,  19],
       [330,  20]], dtype=int32)
'''

# x좌표와 y좌표를 리스트로 받는다.
x0, y0 = zip(*np.squeeze(contours[0])) # zip(*squeeze)를 하면 열로 잘린다.
plt.plot(x0, y0, c="b")
plt.show()
```

![](/assets/images/Preprocessing8_1.png)

### 컨투어 그리기
**상하구조(hierarchy)**는 1, 0, -1 값으로 이루어진 **(컨투어 수 x 4) 크기의 행렬**이다. 컨투어 라인은 등고선으로 생각하면 편하다. 

- 1번 원소: **같은 수준의 다음 컨투어**의 인덱스. 같은 수준의 다음 컨투어가 없으면 -1
- 2번 원소: **같은 수준의 이전 컨투어**의 인덱스. 같은 수준의 이전 컨투어가 없으면 -1
- 3번 원소: **하위 자식 컨투어의 인덱스**. 가장 하위의 컨투어면 -1
- 4번 원소: **부모 컨투어의 인덱스**. 가장 상위의 컨투어면 -1

위 값에서 **첫번재 컨투어 라인**이 가장 상위 컨투어라는 것을 알 수 있다.

```py
hierachy
'''
        |컨투어1|컨투어2|
array([[[-1, -1,  1, -1],
        [-1, -1, -1,  0]]], dtype=int32)
'''
```

`drawContorus`함수를 사용하면 컨투어 정보에서 비트맵 이미지를 만들 수 있다.

`drawContorus(image, contours, contourIdx, color)`
- `image`: 원본 이미지
- `contours`: 컨투어 라인 정보
- `contourIdx`: 컨투어 라인 번호
- `color`: 색상 (숫자로 표현)

```py
### 컨투어 정보에서 비트맵 이미지를 만들 수 있다.
image = cv2.drawContours(img, contours, 0, 4)

plt.subplot(1, 2, 1)
plt.imshow(img_raw, cmap='bone')
plt.title("원본 이미지")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(image, cmap='bone')
plt.title("컨투어 이미지")
plt.axis('off')
plt.tight_layout()
plt.show()
```

## 2. 컨투어 특징

### 이미지 모멘트
이미지 모멘트는 **컨투어에 관한 특징값**을 뜻한다. OpenCV에서는 `moments` 함수로 이미지 모멘트를 구한다.

컨투어 포인트 배열을 입력하면 해당 컨투어의 모멘트를 딕셔너리 타입으로 반환한다. 반환하는 모멘트는 총 24개로 10개의 **위치 모멘트**, 7개의 **중심 모멘트**, 7개의 **정규화된 중심 모멘트**로 이루어져 있다.
- Spatial Moments : M00, M01, M02, M03, M10, M11, M12, M20, M21, M30
- Central Moments : Mu02, Mu03, Mu11, Mu12, Mu20, Mu21, Mu30
- Central Normalized Moments : Nu02, Nu03, Nu11, Nu12, Nu20, Nu21, Nu30

```py
c0 = contours[0]
moment = cv2.moments(c0)
moment
'''
{'m00': 42355.0,
 'm10': 7943000.166666666,
 'm01': 6115675.833333333,
 'm20': 1914995009.1666665,
 ...
 'mu20': 425412866.6175771,
 'mu11': -103767899.87557864,
 'mu02': 158769774.61250484,
 ...
 'nu20': 0.2371380524771235,
 'nu11': -0.0578433790256196,
 'nu02': 0.08850309451896964,
 ...
 '''
```

- **컨투어의 면적**은 모멘트의 `m00` 값이고, `cv2.contourArea()` 함수로도 구할 수 있다.
- **컨투어의 둘레**는 `arcLength` 함수로 구할 수 있다.
    - 두번째 파라미터인 `closed`의 의미는 **폐곡선의 여부**로, 설정한 값이 True일 때는 컨투어의 시작점과 끝점을 이어 도형을 구성하고 그 둘레 값을 계산한다. False인 경우 시작점과 끝점을 잇지 않고 둘레를 계산한다.
- **컨투어를 둘러싸는 박스**는  `boundingRect` 함수로 구한다.
- **가로 세로 비율**은 바운딩 박스에서 구할 수 있다.

```py
## 면적
cv2.contourArea(c0)
## > 42355.0

# 둘레
cv2.arcLength(c0, closed=True), cv2.arcLength(c0, closed=False)
## > (2203.678272008896, 2199.678272008896)

# 둘러싸는 박스 
x, y, w, h = cv2.boundingRect(c0)
x, y, w, h
## > (18, 9, 371, 304)

# 가로세로 비율
aspect_ratio = float(w) / h
aspect_ratio
## > 1.2203947368421053
```

```py
# 박스와 그림을 같이 그릴 경우
plt.plot(x0, y0, c="b")
plt.plot(
    # x축과 y축, w(가로너비), h(세로높이)를 조합
    [x, x + w, x + w, x, x], 
    [y, y, y + h, y + h, y],
    c="r"
)
plt.show()
```

![](/assets/images/Preprocessing8_2.png)


- 컨투어 라인의 **중심점**과 **좌우상하의 끝점**은 아래처럼 구할 수 있다.

```py
cx = int(moment['m10'] / moment['m00'])
cy = int(moment['m01'] / moment['m00'])

# c0은 contours0의 변수로, 가장자리 값을 most변수로 구할 수 있다.
leftmost = tuple(c0[c0[:, :, 0].argmin()][0])
rightmost = tuple(c0[c0[:, :, 0].argmax()][0])
topmost = tuple(c0[c0[:, :, 1].argmin()][0])
bottommost = tuple(c0[c0[:, :, 1].argmax()][0])

plt.subplot(1,2,1)
plt.imshow(image, cmap='bone')
plt.title("컨투어의 중심점")
plt.axis('off')
plt.scatter([cx], [cy], c="r", s=30)

plt.subplot(1,2,2)
plt.imshow(img_raw, cmap='bone')
plt.axis("off")
plt.scatter(
    [leftmost[0], rightmost[0], topmost[0], bottommost[0]], 
    [leftmost[1], rightmost[1], topmost[1], bottommost[1]], 
    c="b", s=30)
plt.title("Extream Points")

plt.show()
```

## 3. 연습문제

**좌우상하의 끝점**을 구하는 코드를 이용해서 '영수증'사진의 꼭짓점을 구하는 코드를 작성해보도록 하자.

> 이전 게시글인 [[머신러닝] 데이터 전처리 (이미지) - 2. 이미지 필터링]({% post_url 2020-11-11-Preprocessing07%})에서 사용했던 연습문제 코드를 이용

- 이미지 가공

```py
# 이미지 로드
img = cv2.imread("./receipt.png")
img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img_raw.copy().astype('uint8')

# 임계처리
maxval = 255
thresh = 200
_, img = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)

# 중앙값 처리
img = cv2.medianBlur(img, 3)

# closing
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, 
                       cv2.getStructuringElement(
                           cv2.MORPH_ELLIPSE, (40, 40)))
```

![](/assets/images/Preprocessing8_3.png){: width="400" height="600"}{: .align-center}

- 이미지의 꼭짓점 위치 표시

```py
# 컨투어로 꼭짓점 찾기
contours, hierachy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
c0 = contours[0]

leftmost = tuple(c0[c0[:, :, 0].argmin()][0])
rightmost = tuple(c0[c0[:, :, 0].argmax()][0])
topmost = tuple(c0[c0[:, :, 1].argmin()][0])
bottommost = tuple(c0[c0[:, :, 1].argmax()][0])

# 시각화 코드
plt.imshow(img_raw, cmap='bone')
plt.axis("off")
plt.scatter(
    [leftmost[0], rightmost[0], topmost[0], bottommost[0]],
    [leftmost[1], rightmost[1], topmost[1], bottommost[1]],
    c="b", s=30)
plt.title("Extream Points")
plt.show()
```

![](/assets/images/Preprocessing8_4.png){: width="300" height="500"}{: .align-center}


## 4. 컨투어 추정

> 이하 컨투어 추정의 코드의 경우 생략한다. 추후 관심이 있을 경우 상단의 출처 페이지 ([데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.02.03%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%EC%BB%A8%ED%88%AC%EC%96%B4.html) )를 참조 바랍니다.

컨투어 추정은 Douglas-Peucker라는 알고리즘을 이용해 **컨투어 포인트의 수를 줄여** 실제 컨투어 라인과 유사한 라인을 그릴 때 사용한다. 이미지를 가지고 오브젝트 디렉팅을 할 때 사용한다고 하낟.

OpenCV에서는 `approxPolyDP` 라는 함수로 구현되어 있다. **입력값**으로는 컨투어 포인트 배열, 실제 컨투어 라인과 근사치의 최대거리, 폐곡선 여부 등을 받는다.   다음 코드는 실제 컨투어 라인과 **근사치의 최대거리를 0.01, 0.05로 설정**하여 실제 컨투어 라인과 비교 한다.

![](/assets/images/Preprocessing8_5.png)

### 1) Convex Hull
**Convex Hull**이란 컨투어 포인트를 모두 포함하는 볼록한 외곽선을 의미한다. 결과는 컨투어 추정과 비슷하지만 방법이 다르다. 먼저, `cv2.isContourConvex()`함수를 사용해 이미지의 컨투어가 볼록(convex)한지 확인 할 수 있다. 입력한 컨투어 배열이 볼록(convex)하다면 True, 아니라면 False 값을 반환한다. 예제로 활용한 말 그림은 볼록하지 않기에 False를 리턴한다. 

컨투어 라인이 볼록하지 않다면, `cv2.convexHull()` 함수를 사용해 컨투어라인을 볼록하게 만들 수 있다.

![](/assets/images/Preprocessing8_6.png)

### 2) Bounding Rectangle

Bounding Rectangle은 **컨투어 라인을 둘러싸는 사각형을 그리는 방법**이다. 사각형을 그리는 방법은 2가지가 있다.

- `boundingRect`: Straight Bounding Rectangle : 물체의 회전은 고려하지 않은 사각형
- `minAreaRect`: Rotated Rectangle : 물체의 회전을 고려한 사각형

![](/assets/images/Preprocessing8_7.png)


### 3) Minumum Enclosing Circle 과 Fitting Ellipse

각각 컨투어 라인을 완전히 포함하는 **가장 작은 원과 타원을 그리는 방법**이다.
- `minEnclosingCircle`: 물체를 포함한 원
- `fitEllipse`: 물체를 포함한 타원

![](/assets/images/Preprocessing8_8.png)
