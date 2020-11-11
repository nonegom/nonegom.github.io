---
title:  "[머신러닝] 데이터 전처리 (자연어)  - 1. NLTK 자연어 처리 패키지 "
excerpt: "NLTK 패키지를 사용한 자연어 전처리 Text 클래스를 사용한 FreqDist, WordCloud 출력"

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

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.01.01%20NLTK%20%EC%9E%90%EC%97%B0%EC%96%B4%20%EC%B2%98%EB%A6%AC%20%ED%8C%A8%ED%82%A4%EC%A7%80.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

# NLTK 자연어 처리 패키지
NLTK(Natural Language Toolkit) 패키지는 교육용으로 개발된 **자연어 처리 및 문서 분석용** 파이썬 패키지다. 다양한 기능과 예제를 가지고 있어 실무 및 연구에서 많이 사용된다.

**주요기능**
- 말뭉치
- 토큰 생성
- 형태소 분석
- 품사 태깅

## 0. 말뭉치
말뭉치(Corpus)는 자연어 분석 작업을 위해 만드는 샘플 문서 집합을 말한다. 단순한 문서부터 형태소나 품사 등을 분석 및 정리해 놓은 것을 포함한다. NLTK 패키지의 `corpus` 서브패키지에서는 다양한 연구용 말뭉치를 제공한다.(전체 `corpus`의 일부)  

저작권이 말소된 문학작품을 포함하는 `gutenberg`말뭉치 또한 있다. 아래 실습에서는 jane austen의 emma 문서를 통해 진행하도록 하겠다.


```py
## NLTK 파일 불러오기
import  nltk
nltk.download("book")
from nltk.book import *

## gutenberg 말뭉치
nltk.corpus.gutenberg.fileids()
#> ['austen-emma.txt',
#      ...
#  'whitman-leaves.txt']
```

중 '제인 오스틴의 엠마' 문서를 살펴보면 원문 형태를 그대로 포함하고 있다.

```py
emma_raw = nltk.corpus.gutenberg.raw("austen-emma.txt")
print(emma_raw[:1000]) 
# 원문 중 1000단어까지만 출력
```

## 1. 토큰 생성
자연어 문서를 분석하기 위해 긴 문자열을 작은 단위로 나누어야 하는데, 이 단위를 **토큰(token)**이라고 한다. 그리고 문자열을 토큰으로 나누는 작업을 **토큰 생성(tokenizing)**이라고 한다. 영문의 경우 문장이나 단어 등을 토큰으로 사용하고, **정규 표현식**을 사용할 수 있다.  

문자열을 토큰으로 분리하는 함수를 **토큰 생성 함수(tokenizer)**라고 한다. 토큰 생성함수는 **문자열을 입력받아 토큰 문자열의 리스트를 출력**한다. 

```py
# 문장으로 토큰화 
from nltk.tokenize import sent_tokenize
print(sent_tokenize(emma_raw[:1000])[3])

# 단어로 토큰화 
from nltk.tokenize import word_tokenize
word_tokenize(emma_raw[50:100])

# 정규 표현식을 사용해 토큰화
from nltk.tokenize import RegexpTokenizer
retokenize = RegexpTokenizer("[\w]+")
retokenize.tokenize(emma_raw[50:100])
```

## 2. 형태소 분석
형태소 (morpheme)는 언어학에서 일정한 의미가 있는 가장 작은 말의 단위를 의미한다. 보통 자연어 처리에서는 토큰으로 **형태소**를 이용한다. **형태소 분석**은 단어로부터 어근, 접두사, 접미사, 품사 등 다양한 언어적 속성을 파악해 이를 이용해 형태소를 찾아내 처리하는 작업이다. 형태소 분석의 예로는 아래와 같은 작업이 있다.
- 어간 추출(stemming)
- 원형 복원(lemmatizing)
- 품사 부착(Part-Of-Speech tagging)

### 어간 추출과 원형 복원
**어간 추출(stemming)**은 변화된 단어의 접미사나 어미를 제거하여 같은 의미를 가지는 형태소의 기본형을 찾는 방법이다. NLTK는 `PorterStemmer`나 `LancasterStemmer` 등을 제공한다. 어간 추출법은 단순히 어미를 제거할 뿐이므로 단어의 원형의 정확히 찾아주지는 않는다.

```py
from nltk.stem import PorterStemmer, LancasterStemmer

# 가장 단순한 알고리즘
st1 = PorterStemmer()
st2 = LancasterStemmer()

words = ["fly", "flies", "flying", "flew", "flown"]


print("Porter Stemmer   :", [st1.stem(w) for w in words])
print("Lancaster Stemmer:", [st2.stem(w) for w in words])
# Porter Stemmer : ['fli', 'fli', 'fli', 'flew', 'flown']
# Lancaster Stemmer: ['fly', 'fli', 'fly', 'flew', 'flown']

# 한계가 존재하기 때문에 모든 것을 찾아내지 못한다.
```
### 참고 사항
- 정규화: 의미와 쓰임이 같은 단어를 같은 토큰으로 구분
- 대소문자 통합, 어간 추출, 표제어 추출, 품사 부착, 불용어(StopWords)구분 등을 수행할 수 있다.  



**원형 복원(lemmatizing)**은 같은 의미를 가지는 여러 단어를 사전형으로 통일하는 작업이다. 품사(part of speech)를 지정하는 경우 좀 더 정확한 원형을 찾을 수 있다.

```py
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
[lm.lemmatize(w, pos="v") for w in words] # 품사를 동사로 설정

# > ['fly', 'fly', 'fly', 'fly', 'fly']
```

## 3. 품사 태깅
품사(POS, part-of-speech)는 낱말을 문법적인 기능이나 형태, 뜻에 따라 구분한 것이다. 예를 들어 NLTK에서는 **펜 트리뱅크 태그세트(Penn Treebank Tagset)**라는 것을 이용한다. 다음은 NLTK에서 사용하는 품사의 예이다.
- NNP: 단수 고유명사
- VB: 동사
- VBP: 동사 현재형
- TO: to 전치사
- NN: 명사(단수형 혹은 집합형)
- DT: 관형사

`nltk.help.upenn_tagset` 명령으로 자세한 설명을 볼 수 있다.

> 동일한 철자의 단어가 다른 의미나 다른 품사로 쓰이는 경우, 다른 토큰으로 토큰화해야 한다.

```py

# 샘플 데이터 (sentence)생성
from nltk.tag import pos_tag
sentence = "Emma refused to permit us to obtain the refuse permit"
tagged_list = pos_tag(word_tokenize(sentence))
tagged_list

# 앞 뒤의 문맥을 다 보면서 품사를 찾아 품사를 붙인다.
# 최종 결과물은 튜플의 리스트로 나온다.
'''
[('Emma', 'NNP'),
 ('refused', 'VBD'),
 ('to', 'TO'),
 ('permit', 'VB'),
 ('us', 'PRP'),
 ('to', 'TO'),
 ('obtain', 'VB'),
 ('the', 'DT'),
 ('refuse', 'NN'),
 ('permit', 'NN')]
'''
```

품사 태그 정보를 사용하면 해당 품사의 토큰만 선택할 수 있다. 아래의 코드는 명사인 토큰만 선택하는 코드이다. 또한 `untag` 명령을 사용하면 태그 튜플을 제거할 수 있다.
```py
nouns_list = [t[0] for t in tagged_list if t[1] == "NN"]
nouns_list
# > ['refuse', 'permit']

from nltk.tag import untag
untag(tagged_list)
'''
['Emma',
'refused',
'to',
'permit',
'us',
'to',
'obtain',
'the',
'refuse',
'permit']
'''
```

Scikit-Learn 등에서 자연어 분석을 할 때는 **같은 토큰이라도 품사가 다르면 다른 토큰으로 처리해야 하는 경우**가 많은데 이 때는 원래의 토큰과 품사를 붙여서 새로운 토큰 이름을 만들어 사용하면 철자가 같고 품사가 다른 단어
를 구분할 수 있다.

```py
# 튜플로 나온 값을 품사로 붙여서 토큰화 시킨다.
def tokenizer(doc):
  return ["/".join(p) for p in tagged_list]
tokenizer(sentence)
'''
['Emma/NNP',
'refused/VBD',
'to/TO',
'permit/VB',
'us/PRP',
'to/TO',
'obtain/VB',
'the/DT',
'refuse/NN',
'permit/NN']
'''
```

## 4. Text클래스

NLTK의 `Text`클래스는 문서 분석에 유용한 여러가지 메서드를 제공한다. 토큰열을 입력해 Text클래스를 생성할 수 있다.  

```py
from nltk import Text
text = Text(retokenize.tokenize(emma_raw))
```
### 사용가능한 메서드
- `plot` : 단어 사용빈도 그래프
- `dispersion_plot`: 단어 위치 그래프
- `concordance`: 콩코던스 분석
- `similar`: 유사 단어 검색
- `common_contexts`: 공통 문맥 인쇄

- `plot` 메서드를 사용하면 **각 단어(토큰)의 사용 빈도를 그래프**로 그려준다.

```py
# 20개 출력
text.plot(20)
plt.show()
```
![](/assets/images/preprocessing1_1.png)

- `dispersion_plot` 메서드는 단어가 사용된 위치를 시각화해서 나타내준다. 소설 엠마의 각 등장인물에 대해 적용하면 다음과 같은 결과를 얻을 수 있다. 'Word Offset'은 단어의 순서이다.

```py
text.dispersion_plot(["Emma", "Knightley", "Frank", "Jane", "Harriet", "Robert"])
```
![](/assets/images/preprocessing1_2.png)

- `concordance` 메서드로 단어가 사용된 위치를 직접 표시하면 문맥(context)이 어떤지 볼 수 있다. 여기에서 문맥은 해당 단어의 앞과 뒤에 사용된 단어를 뜻한다.
 앞 뒤의 문맥을 '윈도우'라고 한다.

```py
text.concordance("Emma")
'''
Displaying 25 of 865 matches:
Emma by Jane Austen 1816 VOLUME I CHAPTER

Jane Austen 1816 VOLUME I CHAPTER I Emma Woodhouse handsome clever 
 both daughters but particularly of Emma Between _them_ it was more the 
                                  ...
'''
```

- `similar`메서드는 같은 문맥에서 주어진 단어 대신 사용된 횟수가 높은 단어들을 찾는다.

```py
text.similar("Emma")
'''
> she it he i harriet you her jane him that me and all they them herself
  there but be isabella
'''
```

- `common_contexts`메서드는 두 단어의 공통 문맥을 보여준다.
```py
text.common_contexts(["Emma", "she"])
'''
but_was and_could that_should said_and which_could whom_knew
which_particularly and_imagined that_could said_i that_began
and_thought do_was but_could than_had said_but manner_was this_could
as_saw possible_could
'''
```

## 5. FreqDist
`FreqDist`클래스는 문서에 사용된 **단어(토큰)의 사용빈도 정보**를 담는 클래스이다. `Text`클래스의 `vocab`메서드로 추출할 수 있다.

```py
## 1. Text 클래스에서 생성
fd = text.vocab()
type(fd)
# > nltk.probability.FreqDist
```

또는 '토큰 리스트'를 넣어서 직접 만들 수도 있다. 아래의 코드에서는 Emma 말뭉치에서 **사람의 이름만 모아**서 `FreqDist`클래스 객체를 만들었다.
품사 태그 중 `NNP`(고유대명사)이면서 필요없는 단어(stopwords)는 제거했다.

```py
## 2. 토큰 리스트에서 생성
from nltk import FreqDist

stopwords = ["Mr.", "Mrs.", "Miss", "Mr", "Mrs", "Dear"]
emma_tokens = pos_tag(retokenize.tokenize(emma_raw))
names_list = [t[0] for t in emma_tokens if t[1] == "NNP" and t[0] not in stopwords]
fd_names = FreqDist(names_list)
```
`FreqDist`클래스는 단어를 키(Key), 출현빈도를 값(value)으로 가지는 사전 자료형과 유사하다. 다음 코드는 전체 단어의 수, 'Emma'라는 단어의 출현 횟수, 확률을 각각 계산한다.

```py
## 3. 단어 빈도 분석
fd_names.N(), fd_names["Emma"], fd_names.freq("Emma")
# > (7863, 830, 0.10555767518758744)
```

`most_common` 메서드를 사용하면 가장 출현 횟수가 높은 단어를 찾는다.

```py
fd_names.most_common(5)
'''
[('Emma', 830),
 ('Harriet', 491),
 ('Weston', 439),
 ('Knightley', 389),
 ('Elton', 385)]
'''
```

## 6. 워드 클라우드
wordcloud 패키지를 사용하면 단어의 사용 빈도수에 따라 **워드클라우드(Word Cloud) 시각화**를 할 수 있다.  

```py
from wordcloud import WordCloud

wc = WordCloud(width=1000, height=600, background_color="white", random_state=0)
plt.imshow(wc.generate_from_frequencies(fd_names)) # frequency객체 삽입
plt.axis("off") # 여백 자르기
plt.show()
```
![](/assets/images/preprocessing1_3.png)

## 7. 연습문제 (총정리)

아래 코드는 위에서 배운 것들을 적용해서 '성경' 등장인물 이름의 워드 클라우드를 제작하는 코드이다.

```py

# 1. 코퍼스 파일 로드
import nltk
nltk.corpus.gutenberg.fileids()
bible_raw = nltk.corpus.gutenberg.raw("bible-kjv.txt")

# 2-1. Text 클래스에서 생성 (vocab 메서드 사용)
from nltk import Text
from nltk.tokenize import word_tokenize
text = Text(word_tokenize(bible_raw))
fd = text.vocab()

# 2-2. 토큰 리스트에 생성 (불용어(stopwords) 제거)
from nltk.tag import pos_tag
from nltk import FreqDist
stopwords = ["Mr.", "Mrs.", "Miss", "Mr", "Mrs", "Dear"]
bible_tokens = pos_tag(word_tokenize(bible_raw))
names_list = [t[0] for t in bible_tokens if t[1] =="NNP" and t[0] not in stopwords]
fd_names = FreqDist(names_list)

# 3-1. 워드 클라우드 출력 (vocab)
from wordcloud import WordCloud

wc = WordCloud(width=1000, height=600, background_color="white", random_state=0)
plt.imshow(wc.generate_from_frequencies(fd)) # frequency객체 삽입
plt.axis("off") # 여백 자르기
plt.show()

```
![](/assets/images/preprocessing1_4.png)

```py
# 3-2. 워드 클라우드 출력 (토큰 리스트 사용)
wc = WordCloud(width=1000, height=600, background_color="white", random_state=0)
plt.imshow(wc.generate_from_frequencies(fd_names)) # frequency객체 삽입
plt.axis("off") # 여백 자르기
plt.show()
```
![](/assets/images/preprocessing1_5.png)

- `vocab`메서드를 사용해 워드클라우드를 생성했을 때에는 관사나 접속사 등 빈도가 높은 다른 단어들이 많이 나온다. 따라서 구하고자 하는 '등장인물'들의 단어의 빈도가 상대적으로 낮게 나와 워드클라우드에 나타나지 않게 된다.  
- 하지만 전처리 작업으로 `stopwords`를 제거하고 워드클라우드 생성 시 '등장인물'과 관련된 단어의 빈도가 훨씬 뚜렷하게 나타나는 것을 확인할 수 있다.  

> 자신이 원하는 목표에 맞게 언어 데이터를 전처리해서 분석하는 작업이 필요하다. 