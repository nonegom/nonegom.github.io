---
title:  "[머신러닝] 데이터 전처리 (사운드)  - 1. 사운드 프로세싱 기초"
excerpt: "사운드 데이터를 이해하기 위한 기본 개념 - 사인함수, 진폭, 주파수, 위상, 싱글톤과 듀얼톤"

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

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.03.01%20%EC%82%AC%EC%9A%B4%EB%93%9C%20%ED%94%84%EB%A1%9C%EC%84%B8%EC%8B%B1%20%EA%B8%B0%EC%B4%88.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 0. 사운드 프로세싱 기초

소리는 공기를 구성하는 입자들이 진동하는 현상이다. 공기의 진동을 측정하는 양적 기준을 **음압(sound pressure)**고 한다. **사운드 데이터**란 이 **음압의 변화를 기록한 시계열 데이터**이다.

## 1. 사인함수
**음압의 변화를 나타내는 시계열 데이터** 중 가장 단순한 형태가 '사인 함수'이다. 삼각 함수 중 하나인 사인 함수는 다음과 같은 **3가지 특징**을 가진다.

- **진폭**(amplitude) $A$: 위 아래로 움직이는 **폭**. **소리의 크기**로 인식된다.

- **주파수**(frequency) $w$ 또는 $f$: **진동 속도**. 주파수가 높으면 빠르게 진동한다. **소리의 높낮이**로 인식된다.

- **위상**(phase) $phi$: 사인 함수의 **시작 시점**. 위상 만큼 출발이 늦어진다. 위상의 차이는 **소리의 시간차**로 인식된다.
    - 사인 함수가 여러 개 있을 시에, 위상의 차이를 확인할 수 있다.


사인함수를 수식으로 다음과 같이 표현할 수 있다.
$$A\ sin(wt - \phi) \quad \text{ or }\quad  A\ sin(2\pi ft-\phi)$$

- 여기에서 $t$는 시간을 나타내며 보통은 초(second)단위이다.

**주파수(frequency)**는 초당 진동의 횟수를 나타낸다. 1회전의 각도가 라디안(radian)이므로 보통은 초당 라디안(radian per second) 단위로 표시하지만 이 값을 $2\pi$로 나누어 **헤르쯔(Hz) 단위***로 표시할 수도 있다. **1 Hz는 1초에 한 번 진동하는 주파수**이다. 위 식에서 $w$는 초당 라디안 단위의 주파수이고 $f$는 Hz 단위의 주파수이다.
$$w = 2\pi f$$

아래의 코드는 여러가지 사인함수를 보여주는 코드이다. 이 코드에서 **A는 진폭, w는 주파수, p는 위상**을 나타낸다. 

```py
t = np.linspace(0, 1, 100)

plt.plot(t, 1 * np.sin(2 * np.pi * t + 0), ls="-", 
        label=r"$\sin\left(2\pi{t}\right)$ 주파수 1Hz. 진폭 1, 위상 0") # 기준이 된다.
plt.plot(t, 2 * np.sin(2 * np.pi * t + 0), ls="--",
         label=r"$2\sin\left(2\pi{t}\right)$ 진폭이 2로 커진 경우")
plt.plot(t, 1 * np.sin(3 * np.pi * t + 0), ls=":", 
        label=r"$\sin\left(3\pi{t}\right)$ 주파수가 1.5H로 커진 경우")
plt.plot(t, 1 * np.sin(2 * np.pi * t - 0.3), ls="-.", 
        label=r"$\sin\left(2\pi{t} - 0.3\right)$ 위상이 늦춰진 경우")

plt.ylim(-2.2, 3)
plt.xticks(np.linspace(0, 1, 5))
plt.legend()
plt.title(r"$A\sin\left(\omega{t}+\phi\right)$")
plt.show()
```

코사인 함수는 사인 함수와 위상이 90도 $= \frac{\pi}{2}$라디안 차이가 있으므로 사인 함수의 일종으로 볼 수 있다.
$$cos(2\pi t) = sin(2\pi t - \frac{\pi}{2})$$


## 2. 주기와 주파수의 관계
주파수 $f$의 **역수를 주기 $T$**라고 한다. 주기는 1번의 진동에 필요한 시간을 뜻한다.
$$f = \frac{1}{T} \quad \text{ or }\quad  w = \frac{2\pi}{T}$$

따라서 주기를 사용할 경우 사인 함수는 다음처럼 쓸 수도 있다. ($f$를 $T$로 바꿔)
$$A sin(\frac{2\pi}{T}t - \phi)$$

## 3. 싱글 톤 (single tone)

주파수는 사람에게 음의 높이(tone)로 인식된다. 사인파처럼 **주파수가 일정한 음압 시계열**은 사람에게 음높이가 **일정한 기계음**으로 들리기 때문에 하나의 사인파로 이루어진 싱들 톤(single tone)이라고 한다. 참고로 통화연결음(ring tone)은 보통 400Hz~450Hz의 싱글 톤을 사용한다.
- `single_tone(frequency, sampling_rate, duration)`
    - `frequency`: 주파수
    - `sampling_rate`: 초당 샘플링 데이터 수. 디폴트 44100
    - `duration`: 지속 시간. 단위 초. 디폴트 1초

```py
def single_tone(frequency, sampling_rate=44100, duration=1):
    t = np.linspace(0, duration, int(sampling_rate))
    y = np.sin(2*np.pi*frequency*t)
    return y
y = single_tone(400)

plt.plot(y[:400])
plt.show()
```
- **sampling data**는 1초에 몇 번 끊느냐를 나타내는 것이다. 즉, 시계열의 개수가 44000개라는 뜻이다.
- **주파수**가 400이라는 것은 400번 반복되었다는 것
- 한 번 반복에 sampling data/frequency(주파수)이므로 대략 110.25라고 할 수 있다.

주피터 노트북에서는 다음 코드로 사운드 데이터를 표시한다.
```py
from IPython.display import Audio, display
display(Audio(y, rate=44100))
```

### 음계 표시
음계에서 기준이 되는 가온다(middle C)음은 **261.62Hz의 싱글 톤**이다. 반음(semitone, half tone, half step)이 올라갈 때마다 $2^{\frac{1}{12}}$배만큼 주파수가 높아지고 12반음 즉, 1옥타브(octave)가 올라가면 **주파수는 2배**가 된다.

다음 코드는 가온 다(C)로부터 한 옥타브의 음에 대해 계산한 주파수이다.

```py
notes = 'C,C#,D,D#,E,F,F#,G,G#,A,A#,B,C'.split(',')
freqs = 261.62 * 2**(np.arange(0, len(notes)) / 12.)
notes = list(zip(notes, freqs))
notes
'''
[('C', 261.62),
 ('C#', 277.17673474627884),
 ('D', 293.6585210786982),
 ('D#', 311.1203654270119),
 ('E', 329.6205450734967),
 ('F', 349.2208026479644),
 ('F#', 369.98655218804913),
 ('G', 391.9870974524774),
 ('G#', 415.29586321592035),
 ('A', 439.9906403173536),
 ('A#', 466.1538452797511),
 ('B', 493.87279536756927),
 ('C', 523.24)]
'''
```
```py
octave = np.hstack([single_tone(f) for f in freqs])
display(Audio(octave, rate=44100))
```

### 화음 표시

**복수의 싱글톤을 더하여** 한번에 소리를 내면 화음이 된다. 예를 들어 도(C) 미(E) 솔(G) 3도 화음은 다음과 같다. 사인파 3개가 합쳐져 복잡한 소리가 된다.

```py
tone_C = single_tone(261.62)
tone_E = single_tone(329.62)
tone_G = single_tone(392)
harmony = tone_C + tone_E + tone_G

plt.plot(harmony[:10000])
plt.show()
```
![](/assets/images/Preprocessing11_1.png)

```py
display(Audio(harmony, rate=44100))
```

## 4. wave 형식 파일
wave 파일은 음압 시계열 데이터를 저장하는 가장 기본적인 파일 형식으로 `.wav` 확장자로 표시한다. wave 파일은 보통 **초당 44100번 음압을 측정**하고 $−32768∼32767(=2^{15})$ 까지의 2바이트(bytes) 숫자로 기록한다.  
파이썬에서 wave 파일을 쓰거나 읽기 위해서는 scipy 패키지의 `io.wavfile` 서브패키지에서 제공하는 `read`, `write` 명령을 사용한다. 
 
아래의 코드는 위에서 만든 octave 파일을 저장하고 다시 읽는 과정을 거친다.

```py
import scipy.io.wavfile

# 초당 샘플링 데이터 수 
sampling_rate = 44100
sp.io.wavfile.write("octave.wav", sampling_rate, octave)

sr, y_read = sp.io.wavfile.read("octave.wav")
# sr == sampling_rate
# y_read == 우리가 읽은 신호 자체

Audio(y_read, rate=sr)
```

## 5. DTMF
전화는 DTMF(Dual-tone multi-frequency) 방식이라는 **두 싱글톤 조합의 음향 신호**로 전화번호를 입력받는다. (집 전화나 공중전화에 사용된다.)

![](/assets/images/Preprocessing11_2.JPG){: .align-center}

해당 주파수의 싱글톤들을 조합해 듀얼톤으로 만든 후 조합하면 전화번호를 출력할 수도 있다.