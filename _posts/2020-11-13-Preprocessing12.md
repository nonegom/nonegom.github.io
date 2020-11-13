---
title:  "[머신러닝] 데이터 전처리 (사운드)  - 2. 푸리에 변환과 스펙트럼"
excerpt: "정현파 조합을 분리하기 위한 방법 - Fourier Transform과 Spectrum 방법"

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

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.03.02%20%ED%91%B8%EB%A6%AC%EC%97%90%20%EB%B3%80%ED%99%98%EA%B3%BC%20%EC%8A%A4%ED%8E%99%ED%8A%B8%EB%9F%BC.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 0. 푸리에 변환과 스펙트럼
모음의 경우 '싱글톤'이 자주 나오고, 자음의 경우 '하모니'가 생기는 경향이 있다. 이는 장애음과 비장애음의 차이라고 볼 수 있다. 

음성(speech), 음악(music) 등의 음향(sound) 데이터에서 특징(feature)을 추출하는 방법 중 **푸리에 변환**(Fourier transfrom)과 **스펙트럼**(spectrum)이 있다.

> 들어가기에 앞서 이 분야는 제대로 파고들면 한 학기 분량이 나오므로, 개념에 대한 이해 정도만 다룬다. 

## 1. 정현파 조합
모든 신호는 주파수와 크기, 위상이 다른 정현파의 조합으로 나타낼 수 있다. **푸리에 변환**은 조합된 정현파의 합(하모니) 신호에서 그 신호를 구성하는 **정현파들을 각각 분리해내는 방법**이다.

```py
N = 1024
T = 1.0 / 44100.0
f1 = 697
f2 = 1209
t = np.linspace(0.0, N*T, N)
y1 = 1.1 * np.sin(2 * np.pi * f1 * t)
y2 = 0.9 * np.sin(2 * np.pi * f2 * t)
y = y1 + y2

plt.subplot(311)
plt.plot(t, y1)
plt.title(r"$1.1\cdot\sin(2\pi\cdot 697t)$")
plt.subplot(312)
plt.plot(t, y2)
plt.title(r"$0.9\cdot\sin(2\pi\cdot 1209t)$")
plt.subplot(313)
plt.plot(t, y)
plt.title(r"$1.1\cdot\sin(2\pi\cdot 697t) + 0.9\cdot\sin(2\pi\cdot 1209t)$")
plt.tight_layout()
plt.show()
```

![](/assets/images/Preprocessing12_1.png)

- 아래와 같은 정형화가 나올 때 어떤 feature를 뽑아낼 수 있는가?
> 즉 가장 밑에 있는 신호에서 조합된 위 두 개의 신호를 분리해내는 방법이 *푸리에 변환*이다.

## 2. 푸리에 변환
주기 $T$를 가지고 반복되는(cyclic) 모든 함수 $y(t)$는 **주파수와 진폭이 다른 몇 개의 사인 함수(정확히는 복수 지수함수)의 합**으로 나타낼 수 있다. 이 **사인 함수의 진폭을 구하는 과정**을 푸리에 변환(Fourier Transform)이라고 한다.
$$y(t) = \sum_{k=-\infty}^{\infty} A_k exp \left( i \cdot 2\pi \frac{k}{T}t \right)$$

이 식에서 $k$번째 사인 함수의 진폭 $A_n$은 아래와 같은 식으로 계산한다. 이것이 **푸리에 변환**이다.
어떤 **주파수(k)의 소리가 얼마나 포함되어 있는지**를 구하는 공식이다. 이를 통해 주파수가 있는지 없는지를 알아낼 수 있다.
$$A_k = \frac{1}{T} \int_{-\frac{T}{2}}^{\frac{T}{2}} y(t)\ exp \left(-i \cdot 2\pi \frac{k}{T}t \right)dt$$

### 이산 푸리에 변환 (DFT)
**이산 푸리에 변환**(Discrete Fourier Transform) 또는 DFT는 길이가 $N$인 **이산시간 시계열 데이터** $ y_0, y_1, \ldots, y_{N-1} $가 있을 때 이 이산시간 시계열이 주기 $N$으로 **계속 반복된다고 가정**하여 푸리에 변환을 한 것이다. 

이 때 원래의 이산시간 시계열 데이터는 다음 주파수와 진폭이 다른 **$N$개의 사인 함수의 합**으로 나타난다.
$$
y_n = \frac{1}{N} \sum_{k=0}^{N-1} Y_k \cdot \exp \left( i\cdot 2\pi\frac{k}{N} n \right)
$$

이 때 진폭 $Y_k$를 원래의 시계열 데이터에 대한 **푸리에 변환값**이라고 한다.
$$
Y_k = \sum_{n=0}^{N-1} y_n \cdot \exp \left( -i\cdot 2\pi\frac{k}{N} n \right)
$$

> 기존 푸리에 형식이 **적분**이었던 데 반해 이산 푸리에 변환은 **시그마**로 바뀌면서 컴퓨터로 쉽게 구할 수 있게 되었다.

### 고속 푸리에 변환 (FFT)
고속 푸리에 변환(Fast Fourier Transform, FFT)는 **아주 적은 계산량으로 DFT를 하는 알고리즘**을 말한다. 길이가 $2^N$인 시계열에만 적용할 수 있다는 단점이 있지만 보통의 DFT가 $O(N^2)$ 수준의 계산량을 요구하는데 반해 **FFT는 $O(N\log_2 N)$ 계산량으로 DFT를 구할 수 있다**.

실제로는 아래와 같이 계속 **반복되는 시계열에 대해 푸리에 변환**을 하는 것이다. 따라서 시계열의 시작 부분과 끝 부분이 너무 다르면 원래 시계열에는 없는 신호가 나올 수도 있는데 이를 **깁스 현상(Gibbs phenomenon)**이라고 한다.

- 깁스현상: 시계열의 시작 부분과 끝 부분이 너무 달라 원래 시계열에 없는 신호가 나오는 현상

![](/assets/images/Preprocessing12_2.JPG){: width="600" height="300"}{: .align-center}

- 빨간색으로 표시된 부분이 **깁스 현상이 발생한 부분**이다.

scipy 패키지의 `fftpack` 서브패키지에서 제공하는 `fft` 명령으로 이 신호에 담겨진 주파수를 분석하면 다음과 같이 692HZ와 1211Hz 성분이 강하게 나타나는 것을 볼 수 있다. 이와 같은 플롯을 **피리오도그램(periodogram)**이라고 한다.

```py
from scipy.fftpack import fft

yf = fft(y, N)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)

plt.stem(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.xlim(0, 3000)

plt.show()
```

![](/assets/images/Preprocessing12_3.png){: width="600" height="300"}{: .align-center}

- 성분이 강하게 나오는 부분 아래의 조금씩 튀는 부분역시 **깁스 현상**때문에 생긴 것이다.

### DCT
DCT(Discrete Cosine Transform)는 DFT와 유사하지만 **기저함수**로 복소 지수함수가 아닌 **코사인 함수를 사용**한다. DFT보다 **계산이 간단하고 실수만 출력한다는 장점**이 있어서 DFT 대용으로 많이 사용된다.
$$ 
Y_k = \sum_{n=0}^{N-1} y_n \cdot \cos \left( 2\pi\frac{k}{N} \left(\frac{2n+1}{4}\right)\right)  
$$
```py
from scipy.fftpack import dct

dct_type = 2
yf2 = dct(y, dct_type, N)

### 시각화 코드 생략
```
![](/assets/images/Preprocessing12_4.png){: width="600" height="300"}{: .align-center}

## 3. 스펙트럼
푸리에 변환은 결정론적인 시계열 데이터를 주파수 영역으로 변환하는 것을 말하지만 스펙트럼(spectrum)은 확률론적인 **확률과정(random process) 모형**을 주파수 영역으로 변환하는 것을 말한다. 따라서 푸리에 변환과 달리 **시계열의 위상(phase) 정보는 스펙트럼에 나타나지 않는다**.

스펙트럼을 추정할 때 사용하는 방법 중의 하나는 전체 시계열을 짧은 구간으로 나눈 뒤 깁스 현상을 줄위기 위해 **각 구간에 윈도우를 씌우고 FFT 계산으로 나온 값을 평균하는 방법**이다. 보통은 **로그 스케일**로 표현한다.

```py
import scipy.signal

f, P = sp.signal.periodogram(y, 44100, nfft=2**12)

plt.subplot(211)
plt.plot(f, P)
plt.xlim(100, 1900)
plt.title("선형 스케일")

plt.subplot(212)
plt.semilogy(f, P)
plt.xlim(100, 1900)
plt.ylim(1e-5, 1e-1)
plt.title("로그 스케일")

plt.tight_layout()
plt.show()
```

![](/assets/images/Preprocessing12_5.png){: width="600" height="300"}{: .align-center}
## 4. STFT
STFT(Short-Time Fourier Transform)는 주파수 특성이 시간에 따라 달라지는 사운드를 분석하기 위한 방법이다. **시계열을 일정한 시간 구간**으로 나누고 **각 구간에 대해 스펙트럼**을 구한 데이터이다. 시간-주파수의 2차원 데이터로 나타난다.

### liborsa 패키지
파이썬으로 STFT 스펙트럼 분석을 하려면 librosa 패키지를 사용한다. (사전에 설치를 해야 한다.)

주피터 노트북에서 librosa 패키지를 사용할 때는 `jupyter_notebook_config.py`파일의 `iopub_data_rate_limit` 설정을 `10000000`정도로 크게 해야 한다.

```py
import librosa
import librosa.display

# 이전에 만들었던 octave 파일을 사용
sr_octave, y_octave = sp.io.wavfile.read("octave.wav")

D_octave = np.abs(librosa.stft(y_octave))
librosa.display.specshow(librosa.amplitude_to_db(D_octave, ref=np.max), sr=sr_octave, y_axis='linear', x_axis='time')
plt.title('Octave')
plt.ylim(0, 2000)
plt.show()
```

![](/assets/images/Preprocessing12_6.png){: width="600" height="300"}{: .align-center}

## 5. 멜 스펙트럼
주파수의 단위를 아래 공식을 따라 멜 단위로 바꾼 스펙트럼을 말한다.
$$m = 2595\ log_{10} \left( 1 + \frac{f}{700} \right)$$

```py
S_octave = librosa.feature.melspectrogram(y=y_octave, sr=sr_octave, n_mels=128)
librosa.display.specshow(librosa.power_to_db(S_octave, ref=np.max), sr=sr_octave, y_axis='mel', x_axis='time')
plt.ylim(0, 4000)
plt.show()
```

### MFCC
MFCC(Mel-frequency cepstral coefficients)는 멜 스펙트럼을 40개의 주파수 구역(band)으로 묶은 뒤에 이를 **다시 푸리에 변환**하여 얻은 계수이다. 스펙트럼이 어떤 모양으로 되어 있는지를 나타내는 **특성값**이라고 생각할 수 있다.

```py
y, sr = librosa.load(librosa.util.example_audio_file(), offset=30, duration=5)

mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
librosa.display.specshow(mfccs, x_axis='time')
plt.title('MFCC')
plt.tight_layout()
plt.show()
```

![](/assets/images/Preprocessing12_8.png){: width="600" height="300"}{: .align-center}