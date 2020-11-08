---
title:  "[머신러닝] 데이터 전처리 (자연어)  - 2. KoNLPy 한국어 처리 패키지 "
excerpt: "NLTK 패키지를 사용한 자연어 전처리 Text 클래스를 사용한 FreqDist, WordCloud 출력"

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

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.01.02%20KoNLPy%20%ED%95%9C%EA%B5%AD%EC%96%B4%20%EC%B2%98%EB%A6%AC%20%ED%8C%A8%ED%82%A4%EC%A7%80.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

# KoNLPy 한국어 처리 패키지

KoNLPy는 **한국어 정보처리**를 위한 파이썬 패키지이다.

## 1.한국어 말뭉치
KoNLPy에서는 대한민국 헌법 말뭉치인 `kolaw` 와 국회법안 말뭉치인 `kobill` 을 제공한다. 각 말뭉치가 포함하
는 파일의 이름은 `fields` 메서드로 알 수 있고 `open` 메서드로 해당 파일의 텍스트를 읽어들인다.

- 한국어 코퍼스의 경우 역사에 비해 데이터가 적은데, 그 이유는 구축했더라도 저작권 문제로 사용하지 못하는 자료들이 많기 때문이다. 

```py
### 헌법 말뭉치
from konlpy.corpus import kolaw
kolaw.fileids()  # ['constitution.txt']

c = kolaw.open('constitution.txt').read()
print(c[:20])
'''
대한민국헌법

유구한 역사와 전통에 
'''

### 법안 말뭉치
from konlpy.corpus import kobill
kobill.fileids() # ['1809890.txt', ... '1809899.txt']

d = kobill.open("1809890.txt").read()
print(d[:20])
'''
지방공무원법 일부개정법률안

(정의화
'''
```

## 2. 형태소 분석
KoNLPY는 다양한 **형태소 분석, 태깅 라이브러리**를 파이썬에서 쉽게 사용할 수 있도록 모아놓았다.
- Hannanum: 한나눔
- Kkma: 꼬꼬마
- Open Korean Text: 오픈 소스 한국어 분석기. 과거 트위터 형태소 분석기

> 아래에서는 위 라이브러리의 형태소만 예제러 포함했다.
- 추가적으로 Komoran이나 Mecab라는 라이브러리도 있다.

```py
from konlpy.tag import *

hannanum = Hannanum()
kkma = Kkma()
okt = Okt()
komoran = Komoran()
mecab = Mecab()
```

위 클래스들은 다음과 같은 메서드를 기본적으로 제공한다.
- `nouns` : 명사 추출
- `morphs` : 형태소 추출
- `pos` : 품사 부착

### 1) 명사 추출
문자열에서 명사만 추출하려면 `noun`명령을 사용한다.

```py
hannanum.nouns(c[:40])
kkma.nouns(c[:40])
okt.nouns(c[:40])
```

### 1) 명사 추출
문자열에서 명사만 추출하려면 `noun`명령을 사용한다.

```py
hannanum.nouns(c[:40])
kkma.nouns(c[:40])
okt.nouns(c[:40])
# 출력 예제 생략
```

### 2) 형태소 추출
명사 뿐만 아니라 모든 품사의 형태소를 알아내려면 `morphs` 명령을 사용한다.

```py
hannanum.morphs(c[:40])
kkma.morphs(c[:40])
okt.morphs(c[:40])
# 출력 예제 생략
```

### 3) 품사부착
`pos` 명령을 사용하면 품사 부착을 한다. 

한국어 품사 태그 세트로는 "21세기 세종계획 품사 테그세트"를 비롯해 다양한 품사 태그세트가 있다. 형태소 분석기마다 **사용하는 품사 태그가 다르므로** 각 형태소 분석기에 대한 문서를 참조한다.

```py
hannanum.pos(c[:40])
'''
[('대한민국헌법', 'N'),
('유구', 'N'),
('하', 'X'),
('ㄴ', 'E'),
...
'''
kkma.pos(c[:40])
'''
[('대한민국', 'NNG'),
('헌법', 'NNG'),
('유구', 'NNG'),
('하', 'XSV'),
('ㄴ', 'ETD'),
'''
okt.pos(c[:40])
'''
[('대한민국', 'Noun'),
('헌법', 'Noun'),
('\n\n', 'Foreign'),
('유구', 'Noun'),
('한', 'Josa'),
'''
```

부착되는 태그의 기호와 의미는 `tagset` 속성으로 확인할 수 있다.
```py
okt.tagset
'''
{'Adjective': '형용사',
'Adverb': '부사',
'Alpha': '알파벳',
'Conjunction': '접속사',
'Determiner': '관형사',
'Eomi': '어미',
'Exclamation': '감탄사',
'Foreign': '외국어, 한자 및 기타기호',
'Hashtag': '트위터 해쉬태그',
'Josa': '조사',
'KoreanParticle': '(ex: ㅋㅋ)',
'Noun': '명사',
...
'''
```

## 3. NLTK 기능 사용
NLTK의 `Text`클래스를 결합해서 NLTK 기능을 사용할 수도 있다.
```py
from nltk import Text
kolaw = Text(okt.nouns(c), name="kolaw")
kolaw.plot(30)
plt.show()
```

![](assets/images/Preprocessing2_1.png)

```py
from wordcloud import WordCloud

# 자신의 컴퓨터 환경에 맞는 한글 ttf폰트 경로 설정
font_path = "C:\Windows\Fonts\malgun.ttf"
wc = WordCloud(width=1000, height=600,
               background_color="white", font_path=font_path)

plt.imshow(wc.generate_from_frequencies(kolaw.vocab()))
plt.axis("off")
plt.show()
```

![](assets/images/Preprocessing2_2.png)

## 참고 (사용자 사전)
**사용자 사전**을 추가할 수가 있다.

- 신조어나 고유명사 등을 추가할 수 있다.
- 한나눔, 꼬꼬마, 오픈 코리아마다 사용자 사전 추가 방법이 다르다. 