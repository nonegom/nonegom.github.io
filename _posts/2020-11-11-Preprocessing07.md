---
title:  "[머신러닝] 데이터 전처리 (이미지)  - 2. 이미지 필터링"
excerpt: "이미지 필터링 - threshold, imagefilter, blur, Morphological Transformation 및 이미지 필터링 예제"

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

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.02.02%20%EC%9D%B4%EB%AF%B8%EC%A7%80%20%ED%95%84%ED%84%B0%EB%A7%81.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  


## 0.이미지 필터링

여러 수식을 이용해 이미지를 이루고 있는 픽셀 행렬을 다른 값으로 바꾸어 **이미지를 변형하는 것**을 말한다.

## 1. 임계처리 (Thresholding)
이미지 행렬에서 하나의 픽셀값을 **사용자가 지정한 기준값(threshold)을 사용하여 이진화(binarization)**하는 가장 단순한 필터다. OpenCV에서는 `threshold`라는 함수로 구현되어 있다. 

### threshold 함수
- `threshold(src, thresh, maxval, type)`
    - `src` : 그레이 스케일 이미지
    - `thresh` : 기준값
    - `maxval` : 기준값을 넘었을 때 적용할 최대값
    - `type` : 임계처리 유형
        - `THRESH_BINARY` : 기준값을 넘으면 최대값 아니면 0
        - `THRESH_BINARY_INV` : 기준값을 넘으면 0 아니면 최대값
        - `THRESH_TRUNC` : 기준값을 넘으면 기준값 아니면 최대값
        - `THRESH_TOZERO` : 기준값을 넘으면 원래값 아니면 0
        - `THRESH_TOZERO_INV` : 기준값을 넘으면 0 아니면 원래값

> 기본적으로 흑백 이미지만 처리가 가능하다.

```py
import cv2
from skimage.data import coins
img = coins()
maxval = 255
thresh = maxval / 2  # 최대값의 반을 thresh의 값으로 받음

# 원본나오는 것은 '_(언더바)' 변수 할당 안받는 것으로 설정
_, thresh1 = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY) # 배경 까맣게
_, thresh2 = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY_INV) # 객체를 까맣게
_, thresh3 = cv2.threshold(img, thresh, maxval, cv2.THRESH_TRUNC) 
_, thresh4 = cv2.threshold(img, thresh, maxval, cv2.THRESH_TOZERO) 
_, thresh5 = cv2.threshold(img, thresh, maxval, cv2.THRESH_TOZERO_INV)

titles = ['원본이미지', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
plt.figure(figsize=(9, 5))

for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i], fontdict={'fontsize': 10})
    plt.axis('off')

plt.tight_layout(pad=0.7)
plt.show()
```

![](/assets/images/Preprocessing7_1.png)

### 사용 예시
- 경계선을 추출할 때, 하얀색 아니면 까만색으로 만들어서 처리하기 편하게 만든다.


## 2. 적응임계처리

이미지 전체에 하나의 기준값을 적용한다. 일정한 영역 내의 **이웃한 픽셀들의 값들을 이용**해 해당 영역에 적용할 기준값을 자체적으로 계산한다. OpenCV에서는 `adaptiveThreshold` 함수로 구현되어 있다.

### adaptiveThreshold 함수
- `adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)`
    - `src` : 그레이스케일 이미지
    - `maxValue` – 기준값을 넘었을 때 적용할 값
    - `adaptiveMethod` : 영역 내에서 기준값을 계산하는 방법.
        - `ADAPTIVE_THRESH_MEAN_C`: 영역 내의 **평균값에서 C를 뺀 값**을 기준값으로 사용
        - `ADAPTIVE_THRESH_GAUSSIAN_C`: 영역에 추후 설명할 **가우시안 블러**를 적용한 후 C를 뺀 값을 기준값으로 사용

    - `thresholdType` : 임계처리 유형
        - `THRESH_BINARY`
        - `THRESH_BINARY_INV`

    - `blockSize` : 임계처리를 적용할 영역의 크기
    - `C` : 평균이나 가중평균에서 차감할 값

```py
from skimage.data import page

img = page()

maxval = 255
thresh = 126
ret, th1 = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)

k = 15
C = 20

th2 = cv2.adaptiveThreshold(
    img, maxval, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, k, C)
th3 = cv2.adaptiveThreshold(
    img, maxval, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, k, C)

images = [img, th1, th2, th3]
titles = ['원본이미지', '임계처리', '평균 적응임계처리', '가우시안블러 적응임계처리']

plt.figure(figsize=(8, 5))
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()
```
![](/assets/images/Preprocessing7_2.png)

## 3. 이미지 필터링
 **필터, 커널, 윈도우**라고 하는 **정방행렬을 정의**한다. 이 커널을 이동시키면서, 같은 이미지 영역과 곱하여 그 결과 값을 이미지의 해당 위치의 값으로 하는 새로운 이미지로 만드는 연산이다. 기호 $\otimes$로 표시한다.
- 커널의 값에 어떤 것을 주느냐에 따라 이미지가 뭉개질 수도 있고, 더 밝아질 수도 있다.

### filter2D 함수
openCV에서는 `filter2D` 함수를 사용한다

- `filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]])`
    - `src`: 이미지
    - `ddepth`: 이미지 깊이(자료형 크기). -1이면 입력과 동일
    - `kernel`: 커널 행렬 사이즈

```py
import cv2
from skimage.data import astronaut

img = astronaut()
img = cv2.resize(img, dsize=(150, 150))

plt.figure(figsize=(8,3))

for i, k in enumerate([2, 6, 11]):
# 커널 생성 및 이미지 필터링
    kernel = np.ones((k, k)) / k**2 
    filtering = cv2.filter2D(img, -1, kernel)
# 이미지 설정
    plt.subplot(1, 3, i+1)
    plt.imshow(filtering)
    plt.title("커널 사이즈 {}".format(k))
    plt.axis("off")
    
plt.show()
```

## 4. 블러 (Blur)
이미지 필터링을 사용해서 이미지를 흐리게 만드는 것을 말한다. 노이즈를 제거하거나 경계선을 흐리게 하기 위해 쓴다. 블러의 종류는 아래와 같이 있다.
- 평균 블러

- 중앙값 블러

- 가우시안 블러

- 양방향 블러


### 1) 평균블러
- 균일한 값을 가지는 커널을 이용한 이미지 필터링. 따라서 커널 영역 내의 평균값으로 픽셀을 대체한다.
- `blur(src, ksize)`
    - `src`: 원본 이미지
    - `ksize`: 커널 크기 (튜플 값으로)

```py
blur = cv2.blur(img, (5, 5))

## 이하 코드에서 그래프 출력 코드 생략
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("원본 이미지")
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(blur)
plt.title("blur 함수 적용")
plt.axis('off')

plt.tight_layout()
plt.show()
```

![](/assets/images/Preprocessing7_4.png)

### 2) 중앙값블러
- 평균이 아닌 중앙값으로 해당 픽셀을 대체한다. **점 모양의 잡음을 제거**하는데 효과적이다.
- `medianBlur(src, ksize)`
    - `src`: 원본 이미지
    - `ksize`: 커널 크기 (튜플 값으로)

```py
# 점 잡음 적용
img_noise = img.copy()

np.random.seed(0)
N = 500
idx1 = np.random.randint(img.shape[0], size=N)
idx2 = np.random.randint(img.shape[1], size=N)
img_noise[idx1, idx2] = 255

# 중앙값 블러로 잡음 제거
img_denoise = cv2.medianBlur(img_noise, 3)
```

![](/assets/images/Preprocessing7_5.png)


### 3) 가우시안 블러
- 가우시안 함수를 커널로 사용한다. 중앙 위치와 커널 위치의 **거리 차가 클수록 가중치가 작아진다**.
- 커널을 전체로 봐서 블러를 실시함으로 이미지 전체가 뭉개지게 된다.
- `GaussianBlur(src, ksize, sigmaX)`
    - `src`: 원본 이미지
    - `ksize`: 커널 크기 (튜플 값으로)
    - `sigmaX`: 표준편차

```py
# 백색 잡음 적용
img_noise = np.clip((img / 255 + np.random.normal(scale=0.1, size=img.shape)) * 255, 0, 255).astype('uint8')

# 가우시안 블러로 잡음 제거
img_denoise = cv2.GaussianBlur(img_noise, (9, 9), 2)
```

![](/assets/images/Preprocessing7_6.png)


### 4) 양방향 필터링
- 가우시안 필터링을 쓰면 이미지의 경계선도 흐려지게 된다. 따라서 명암값의 차이도 커널에 넣는다.
- 경계선은 살리면서, 경계선이 아닌 내부 영역만 블러처리를 하는 것을 말한다.

- `bilateralFilter(src, d, sigmaColor, sigmaSpace)`
    - `src` : 원본 이미지
    - `d` : 커널 크기
    - `sigmaColor` : 색공간 표준편차. 값이 크면 색이 많이 달라도 픽셀들이 서로 영향을 미친다.
    - `sigmaSpace` : 거리공간 표준편차. 값이 크면 멀리 떨어져있는 픽셀들이 서로 영향을 미친다.

```py
img_denoise1 = cv2.GaussianBlur(img_noise, (9, 9), 2)
img_denoise2 = cv2.bilateralFilter(img_noise, 9, 75, 75)
```

![](/assets/images/Preprocessing7_7.png)

> 가우시안 필터링을 적용한 이미지는 다 뭉개졌지만, 양방향 필터링을 적용한 이미지는 어느 정도 경계선이 살아있다는 것을 확인할 수 있다.

## 5. 형태학적 변환
형태학적 변환(Morphological Transformation)이란 **이미지 필터링을 사용**하여 영역을 변화시키는 방법이다. 변환에 적용할 커널은 `getStructuringElement` 함수로 생성한다. 모양과 크기를 변수로 입력받는다.

- `getStructuringElement(shape, ksize)`
    - `shape`: 커널 모양
        - `cv2.MORPH_RECT`: 사각형 모양
        - `cv2.MORPH_ELLIPSE`: 타원형 모양
        - `cv2.TMORPH_CROSS`: 십자 모양
    - `ksize`: 커널 크기

> 커널을 사용하지만 이진화가 된다.

### 침식기법
- 각 픽셀에 커널을 적용하여 **커널 영역 내의 최솟값(어두운 부분)**으로 해당 픽셀을 **대체**한다. 이진화된 이미지에서는 **0인 영역(어두운 부분)이 증가**한다.
- 어두운 영역이 하얀 영역을 침식해서(파고) 들어간다.
- 모든 영역을 '스트라이드(stride)'하면서 이미지를 변환시킨다.
- `erode(src, kernel)`
    - `src`: 원본 이미지
    - `kernel`: 커널

```py
# 데이터 로드
from skimage.data import horse
img = horse().astype('uint8')
img = np.ones(img.shape) - img
ksize = (20, 20)

# 커널 생성
kernel = {}
kernel[0] = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
kernel[1] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
kernel[2] = cv2.getStructuringElement(cv2.MORPH_CROSS, ksize)

title = ["사각형 커널", "타원 커널", "십자가 커널"]
plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("원본 이미지")
plt.axis('off')

for i in range(3):
# 침식 기법 적용
    erosion = cv2.erode(img, kernel[i])
    plt.subplot(2, 2, i+2)
    plt.imshow(erosion, cmap='bone')
    plt.title(title[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

![](/assets/images/Preprocessing7_8.png)

### 팽창기법
- 침식과 반대로 커널 영역의 최댓값(밝은 부분)으로 해당 픽셀을 대체한다.
- `dilate(src, kernel)`
    - `src`: 원본 이미지
    - `kernel`: 커널

```py
plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("원본 이미지")
plt.axis('off')
for i in range(3):
# 팽창기법 적용
    erosion = cv2.dilate(img, kernel[i])
    plt.subplot(2, 2, i+2)
    plt.imshow(erosion, cmap='bone')
    plt.title(title[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

![](/assets/images/Preprocessing7_9.png)

### 그래디언트, 오프닝, 크로징
**그레디언트**
- 그레디언트는 팽창으로 확장시킨 영역에서 침식으로 축소시킨 영역을 빼서 **윤곽선을 파악**하는 것이다.

**오프닝**
- **침식을 적용한 뒤 팽창을 적용**하는 것으로 영역이 점점 둥글게 되므로 **점 잡음, 작은 물체, 돌기 등을 제거하는데 적합**하다.

**클로징**
- 클로징은 반대로 **팽창을 적용한 뒤 침식을 적용**하여 영역이 영역이 붙기 때문에 **전체적인 윤곽을 파악**하는데 적합하다.

- `morphologyEx(src, op, kernel)`
    - `src`: 원본 이미지
    - `op`:
        - `cv2.MORPH_GRADIENT`: cv2.dilate(image) - cv2.erode(image)  
        - `cv2.MORPH_OPEN`: cv2.dilate(cv2.erode(image))
        - `cv2.MORPH_CLOSE`: cv2.erode(cv2.dilate(image))
        - `cv2.MORPH_TOPHAT`: image - opening(image)
        - `cv2.MORPH_BLACKHAT`: image - closing(image)
    - `kernel`: 커널

```py
# 20x20크기의 타원형 커널 사용
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN,
                           cv2.getStructuringElement(
                               cv2.MORPH_ELLIPSE, (20, 20))
                           )
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(
                               cv2.MORPH_ELLIPSE, (20, 20))
                           )
# 그레디언트의 경우 윤곽을 확인하기 위해 크기를 3x3으로 줌
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(
                                cv2.MORPH_ELLIPSE, (3, 3))
                            )
images = [img, gradient, opening, closing]
titles = ["원본 이미지", 'Gradient', 'Opening', 'Closing']
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```

![](/assets/images/Preprocessing7_10.png)

## 6. 연습문제
영수증 이미지를 필터링 처리해서 영수증 부분만 하얀색이 나오고, 다른 배경은 검은색으로 나오게 이진화하라.

```py
import cv2
# 이미지 로드
origin_img = cv2.imread("./receipt.png")

# 이미지 변환
img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)

# 임계처리
maxval = 255
thresh = 200
_, img = cv2.threshold(img, thresh, maxval, cv2.THRESH_BINARY)

# 중앙값 처리
img = cv2.medianBlur(img, 3)

# 클로징
img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,
                           cv2.getStructuringElement(
                               cv2.MORPH_ELLIPSE, (40, 40))
                           )
### 시각화 코드###
plt.figure(figsize=(20, 15))
plt.subplot(121)
plt.title("원본 이미지", fontsize= 25)
plt.imshow(origin_img, cmap="gray")
plt.axis('off')
plt.subplot(122)
plt.title("변형 이미지", fontsize= 25)
plt.imshow(img, cmap="gray")
plt.axis('off')
plt.show()
```

![](/assets/images/Preprocessing7_11.png)


### 해답
- 크기를 변경하는 게 굉장히 중요하다.
1. `threshhold`를 해서 흑색과 흰색으로 만들어야 한다.
2. `Closing`을 하게 되면 단어같은 것들은 사라진다.
3. 나머지 자잘한 것들은 `blur`를 통해서 없앨 수 있다.