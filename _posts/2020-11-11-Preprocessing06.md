---
title:  "[머신러닝] 데이터 전처리 (이미지)  - 1. 이미지 처리 기초
"
excerpt: "이미지를 구성하는 기본 요소 및 이미지 처리 라이브러리 - 픽셀, 색공간, 이미지 형식 / Pillow, OpenCV, Scikit-Image"

categories:
  - ML_PreProcessing
tags:
  - UnSupervisedLearning
  - 11월
toc: true
toc_sticky: true
toc_label: 페이지 목차
use_math: true
---

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.01.05%20%ED%99%95%EB%A5%A0%EB%A1%A0%EC%A0%81%20%EC%96%B8%EC%96%B4%20%EB%AA%A8%ED%98%95.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

# 이미지 처리 기초

이미지 데이터를 표현하는 방식과 이미지 데이터를 처리하기 위한 파이썬 패키지 `Pillow`, `Scikit-Image`, `OpenCV`에 대해 설명한다.

## 1. 픽셀
이미지 데이터는 **픽셀(pixel)**이라고 하는 작은 이미지를 직사각형 형태로 모은 것이다. 각 픽셀은 단색의 직사각형이고, 전체 이미지의 크기를 표현할 때는 (세로픽셀수 x 가로픽셀수) 형식으로 표현한다. 

이미지 데이터를 저장할 때는 픽셀의 색을 표현하는 **스칼라 값**이나 **벡터**를 2차원 배열로 표현한다. 파이썬에서는 `Numpy`의 `ndarray`클래스 배열로 표현한다.

## 2. 색공간
픽셀의 색을 숫자로 표현하는 방식을 **색공간(color space)**라고 한다. 대표적인 색공간으로는 **그레이스케일**(gray scale), **RGB**(Red-Green-Blue), **HSV**(Hue-saturation-Value)방식이 있다. 

### 1) 그레이스케일
그레이 스케일에서는 모든 색이 흑백이고, 각 픽셀은 명도를 나타내는 숫자(스칼라)로 표현된다. **0은 검은색**을 나타내고 숫자가 **커질수록 명도가 증가**하며 하얀색이 된다. 숫자는 보통 0~ 255의 8비트 부호없는 정수로 저장된다.

SciPy 패키지의 `misc` 서브 패키지의 `face` 명령은 이미지 처리용 샘플 이미지를 제공한다. 인수로 `(gray=True)`를 입력하면 그레이스케일 이미지를 반환한다. 이미지의 크기는 배열의 `shape` 속성으로 볼 수 있다. 

이 이미지 데이터는 (768, 1024) 크기의 uint8 자료형 2차원 배열이다. 좌측 상단의 15x15개 픽셀의 데이터만 보면 아래와 같다.

```py
# 1. 이미지 로드
import scipy as sp

img_gray = sp.misc.face(gray=True)
img_gray.shape
# > (768, 1024)

# 2. 히트맵 그리기
import matplotlib.pylab as plt
import seaborn as sns

sns.heatmap(img_gray[:15, :15], annot=True, fmt="d", cmap=plt.cm.bone)
plt.axis("off")
plt.show()
```

![](/assets/images/Preprocessing6_1.png)

### 2) RGB
RGB색공간은 Red, Green, Blue 3가지 색의 명도를 뜻하는 숫자 3개가 합쳐진 벡터로 표현된다. 8비트 부호없는 정수를 사용하는 경우 (255, 0, 0)은 빨간색, (0, 255, 0)은 녹색, (0, 0, 255), 파란색이다.

RGB는 (세로 픽셀수 x 가로픽셀수 x 색체널)의 3차원 배열의 형태로 저장한다. 세번째 축을 **색채널**이라고 부른다. 

```py
# 1. 샘플 이미지 로드
from sklearn.datasets import load_sample_images

dataset = load_sample_images()   
img_rgb = dataset.images[1]
img_rgb.shape

# 2. RGB별 사진 출력
plt.figure(figsize=(10, 2))

plt.subplot(141)
plt.imshow(img_rgb[50:200, 50:200, :])
plt.axis("off")
plt.title("RGB 이미지")

plt.subplot(142)
plt.imshow(img_rgb[50:200, 50:200, 0], cmap=plt.cm.bone)
plt.axis("off")
plt.title("R 채널")

plt.subplot(143)
plt.imshow(img_rgb[50:200, 50:200, 1], cmap=plt.cm.bone)
plt.axis("off")
plt.title("G 채널")

plt.subplot(144)
plt.imshow(img_rgb[50:200, 50:200, 2], cmap=plt.cm.bone)
plt.axis("off")
plt.title("B 채널")

plt.show()
```

![](/assets/images/Preprocessing6_2.png)

그림을 보면 붉은 기와 부분에서는 R채널의 값이 크고 하늘은 B채널의 값이 큰 것을 확인할 수 있다. (하얀색에 가까운 색이 나타날 수록 값이 크다.)


### 3) HSV

HSV(Hue, Saturation, Value) 색공간은 **색상, 채도, 명도** 세가지 값으로 표현된다. 값의 크기는 %로 나타나며, 색상의 스펙트럼의 경우 0~360도의 값을 가진다.

- **색상(Hue)**: 색상값 H는 가시광선 스펙트럼을 주파수 별로 고리모양으로 배치했을 때의 각도이다. 0°~360°의 범위를 갖고 360°와 0°는 빨강을 가리킨다.

- **채도(Saturation)**: 채도값 S는 특정한 색상의 진함의 정도를 나타낸다. 가장 진한 상태를 100%이고 0%는 같은 명도의 무채색이다.

- **명도(Value)**: 명도값 V는 밝은 정도를 나타낸다. 순수한 흰색, 빨간색은 100%이고 검은색은 0%이다.


`matplotlib` 패키지의 `rgb_to_hsv`, `hsv_to_rgb` 명령을 사용하면 RGB 색공간 표현과 HSV 색공간 표현을 **상호변환**할 수 있다.

HSV 색공간으로 표현된 파일은 `imshow` 명령으로 바로 볼 수 없다. 이외에도 RGB, HSV 색공간에 **투명도(transparency)를 표현**하는 **A(Alpha) 채널**이 추가된 RGBA, HSVA 등의 색공간도 있다.

```py
from matplotlib.colors import hsv_to_rgb

V, H = np.mgrid[0:1:100j, 0:1:360j]
S = np.ones_like(V)

# 채도가 100일 때
HSV_S100 = np.dstack((H, S * 1.0, V))
RGB_S100 = hsv_to_rgb(HSV_S100)

# 채도가 20일 때
HSV_S20 = np.dstack((H, S * 0.2, V))
RGB_S20 = hsv_to_rgb(HSV_S20)

HSV_S20.shape

####### 시각화 코드 생략 #######
```

![](/assets/images/Preprocessing6_3.png)



## 3. 이미지 파일 형식

`.bmp` 확장자를 가지는 **비트맵(bitmap) 파일**은 지금까지 설명한 다차원 배열정보를 그대로 담고있지만 비트맵 파일은 파일 용량이 크기 때문에 압축을 통해 용량을 줄인 `JPG`, `GIF`, `PNG` 등의 압축 파일 형식도 많이 사용한다.

- JPEG: 웹상 및 멀티미디어 환경에서 가장 널리 사용되고 있는 포맷. RGB 이미지의 모든 컬러 정보를 다 유지한다.
- GIF: 하나의 파일에 여러 비트맵을 저장해서 다중 프레임 애니메이션 구현 가능, 투명 이미지 지원 
- PNG: GIF 포맷을 대체하기 위해 개발된 파일 포맷, 비손실 압축방식으로 원본에 손상없이 파일의 크기를 줄여준다. 문자 혹은 날카로운 경계가 있는 이미지의 경우 `PNG`가 효과적이다.

# Python 이미지 처리 패키지

## 1. Pillow를 이용한 이미지 처리
`Pillow`는 이전에 사용되던 PIL(Python Imaging Library)패키지를 대체하기 위한 것이다. JPEG, BPM, GIF, PNG, PPM, TIFF 등의 **다양한 포맷을 지원**하고 **초보자**가 다루기 쉽다는 장점이 있다

### 이미지 읽고 쓰기

1. 인터넷에서 실습을 위한 이미지 파일을 내려받는다.

2. `Image` 클래스를 사용하면 여러가지 다양한 포맷의 이미지를 읽고 변환하여 저장할 수 있다. `open` 메서드는 이미지 파일을 열 수 있다.

3. 주피터 노트북에서는 Image 클래스 객체를 바로 볼 수 있다. (바로 이미지 변수 출력하면 됨)

```py
## 1. 파일 다운로드
import urllib.request
urllib.request.urlretrieve("https://www.python.org/static/community_logos/python-logo-master-v3-TM.png", filename="logo.png")

## 2. 이미지 열기
from PIL import Image
img_logo = Image.open("./logo.png")
img_logo.size()
# > (601, 203)

## 3. 이미지 출력
img_logo
```

![](/assets/images/Preprocessing6_4.png)

4. 파일로 저장할 때는 `save`메서드를 사용한다. 확장자를 지정하고 싶가면 해당 이미지 형식으로 자동 변환되어 저장된다.

5. 이미지 데이터 처리를 위해 `Image` 클래스 객체를 `NumPy` 배열로 변환할 때는 `np.array` 함수를 사용한다. `NumPy` 배열이 되면 `matplotlib`의 `imshow` 명령으로 볼 수 있다.

6. 반대로 `NumPy` 배열을 `Image` 객체로 바꿀 때는 `fromarray` 클래스 메서드를 사용한다.

```py
## 4. 이미지 저장  
img_logo.save("./logo.bmp")
img_logo_bmp = Image.open("./logo.bmp")

## 5. Image 객체에서 배열로 (np.array)
img_logo_array = np.array(img_logo_bmp)

plt.imshow(img_logo_array)
plt.axis("off")
plt.show()

## 6. Numpy 배열에서 Image객체로
Image.fromarray(img_logo_array)

```

### 이미지 크기 변환
이미지의 크기를 확대 또는 축소하려면 `resize`메서드를 사용한다. 인수로는 새로운 사이즈의 **튜플을 입력**한다.
```py
img_logo2 = img_logo.resize((300,100))
img_logo2
```

![](/assets/images/Preprocessing6_5.png)

썸네일 이미지를 만들고 싶다면 `Image` 객체의 `thumbnail`메서드를 사용한다. 단, `resize`메서드와는 다르게 원래 **객체 자체를 바꾸는** 인플레이스(in-place) 메서드이므로 주의해서 사용한다. 

```py
img_logo_thumbnail = img_logo.copy() # 원본 데이터 보호를 위해 깊은 복사
img_logo_thumbnail.thumbnail((150, 50))
img_logo_thumbnail
```

![](/assets/images/Preprocessing6_6.png)

### 이미지 회전

이미지를 회전하기 위해서는 `rotate` 메서드를 호출한다. 인수로는 (도(degree) 단위)**각도를 입력**한다. 입력 각도만큼 **반시계 방향으로 회전**한다.

```py
img_logo_rotated = img_logo_png.rotate(45)
img_logo_rotated
```

### 이미지 잘라내기
`crop` 메서드를 사용하면 이미지에서 우리가 관심이 있는 **특정 부분(ROI: region of interest)만 추출** 할 수 있다. 인수로 ROI의 **좌-상의 좌표**, **우-하의 좌표**를 받는다. 아래의 코드는 파이썬 로고이미지에서 파이썬의 마크만 잘라낸 것이다.

```py
img_logo_cropped = img_logo_png.crop((10, 10, 200, 200))
img_logo_cropped
```

![](/assets/images/Preprocessing6_8.png)

## 2. Scikit-Image

### 샘플 이미지
`Scikit-Image`는 `data`라는 모듈을 통해 **샘플 이미지 데이터를 제공**한다. 이미지는 **NumPy 배열 자료형**으로 사용한다.

```py
import skimage.data

img_astro = skimage.data.astronaut()
img_astro.shape
# > (512, 512, 3)
```

### 이미지 읽고 쓰기
`Scikit-Image` 패키지로 이미지를 읽고 쓸 때는 `io` 서브패키지의 `imsave`, `imread` 명령을 사용한다. 파일 확장자를 지정하면 해당 이미지 형식으로 자동 변환한다.

```py
skimage.io.imsave("astronaut.png", img_astro)
img_astro2 = skimage.io.imread("astronaut.png")
```

### 색공간 변환
`Scikit-Image`는 그레이스케일, RGB, HSV 등의 **색공간을 변환하는 기능**을 `color` 서브패키지에서 제공한다.

```py
from skimage import color

plt.subplot(131)
plt.imshow(img_astro)
plt.axis("off")
plt.title("RGB")

plt.subplot(132)
plt.imshow(color.rgb2gray(img_astro), cmap=plt.cm.gray)
plt.axis("off")
plt.title("그레이 스케일")

plt.subplot(133)
plt.imshow(color.rgb2hsv(img_astro))
plt.axis("off")
plt.title("HSV")

plt.show()
```

![](/assets/images/Preprocessing6_9.png)

## 3. OpenCV

OpenCV(Open Source Computer Vision)은 이미지 처리, 컴퓨터 비전을 위한 라이브러리이다. Windows, Linux, OS X(Mac OS), iOS, Android 등 다양한 플랫폼을 지원한다. **실시간 이미지 프로세싱**에 중점을 둔 라이브러리이며 **많은 영상처리 알고리즘**을 구현해 놓았다. 

> 앞으로 이미지 처리 포스팅을 하며 가장 많이 사용될 예정 

다양한 이미지 특징(feature) 처리 기능을 제공하는데 이 기능은 무료소스가 아니기 떄문에, 아나콘다나 pip 명령으로 받은 패키지에는 이 기능이 제외되어 있다.

### 파일 읽고 쓰기
이미지를 읽을 때는 `imread` 메서드를 사용하는데 인수로 **파일이름**과 함께 **`flag`**를 넣을 수 있다.

- `cv2.IMREAD_COLOR`: 이미지 파일을 **컬러**로 읽어들인다. 투명한 부분은 무시되며, flag디폴트 값이다.

- `cv2.IMREAD_GRAYSCALE`: 이미지를 **그레이스케일** 읽어 들인다. 실제 이미지 처리시 중간단계로 많이 사용한다.

- `cv2.IMREAD_UNCHANGED`: 이미지파일을 **알파 채널(투명도)**까지 포함하여 읽어 들인다.

각각 1, 0, -1로 표현된다.

```py
img_astro3 = cv2.imread("./astronaut.png")
img_astro3.shape
# > (512, 512, 3)
```

OpenCV도 이미지 데이터를 Numpy 배열로 저장한다. 하지만 **색 체널의 순서가 B-G-R 순서**로 되어 있다.  RGB값으로 사용하고 싶으면 채널을 분리해서 합해줘야 한다.

- `cvtColor`명령을 사용하면 더 간단하게 색공간을 변환할 수 있다. 

- 이미지 파일을 만들 때는 `imwrite`명령 사용

```py
# 1. bgr -> rgb 
b, g, r = cv2.split(img_astro3) # 각 채널을 분리
img_astro3_rgb = cv2.merge([r, g, b]) # b, r을 서로 바꿔서 Merge

# 2. cvtColor명령
img_astro3_gray = cv2.cvtColor(img_astro3, cv2.COLOR_BGR2GRAY)

# 3. 이미지 파일 저장
cv2.imwrite("./gray_astronaut.png", img_astro3_gray)
```

### 이미지 크기 변환
`resize()`명령으로 이미지의 크기를 변환할 수 있다.

```py
img_astro3_gray_resized = cv2.resize(img_astro3_gray, dsize=(50, 50))
img_astro3_gray_resized.shape
# > ((512, 512), (50, 50))

plt.subplot(121)
plt.imshow(img_astro3_gray, cmap=plt.cm.gray)
plt.title("원본 이미지")
plt.axis("off")

plt.subplot(122)
plt.imshow(img_astro3_gray_resized, cmap=plt.cm.gray)
plt.title("축소 이미지 (같은 크기로 표현)")
plt.axis("off")

plt.show()
```

![](/assets/images/Preprocessing6_10.png)