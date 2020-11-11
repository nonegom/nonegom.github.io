---
title:  "[머신러닝] 데이터 전처리 (자연어)  - 5. 확률론적 언어모형
"
excerpt: "문장의 단어 열이 실제로 현실에서 사용될 수 있는 지 판별하는 모형인 확률론적 언어모형에 대한 이해 - N-gram과 확률 추정"

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

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.01.05%20%ED%99%95%EB%A5%A0%EB%A1%A0%EC%A0%81%20%EC%96%B8%EC%96%B4%20%EB%AA%A8%ED%98%95.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 0.확률론적 언어모형

확률론적 언어 모형(Probabilistic Language Model)은 m개의 단어 $w_1,w_2, \dots ,w_m$ 열(word sequence)이 주어졌을 때 문장으로써 성립될 **확률 $p(w_1,w_2, \dots ,w_m)$ 을 출력**함으로써 이 단어 열이 실제로 현실에서 사용될 수 있는 문장(sentence)인지를 **판별하는 모형**이다. 확률론적 언어모형은 '철자 및 문법 교정', '음성 인식', '자동 교정', '자동 요약', '챗봇' 등 광범위한 분야에 사용된다.

이 확률은 각 단어의 확률과 단어들의 조건부 확률을 이용하여 다음과 같이 계산할 수 있다.
$$
\begin{eqnarray}
P(w_1, w_2, \ldots, w_m) &=& P(w_1) \cdot P(w_2 \;|\; w_1) \cdot  P(w_3 \;|\; w_1, w_2) P(w_4 \;|\; w_1, w_2, w_3) \cdots P(w_m\;|\; w_1, w_2, \ldots, w_{m-1})
\end{eqnarray}
$$

여기에서 $P(w_m\;|\; w_1, w_2, \ldots, w_{m-1})$ 은 지금까지 $w_1, w_2, \ldots, w_{m-1}$라는 단어 열이 나왔을 때, 그 다음 단어로 $w_m$이 나올 조건부 확률을 말한다. 여기에서 지금까지 나온 단어를 **문맥(context) 정보**라고 한다.

이 때 조건부 확률을 모형화하는 방법에 따라 아래와 같이 나뉘어진다.

- 유니그램 모형(Unigram Model)

- 바이그램 모형(Bigram Model)

- N그램 모형(N-gram Model)

> 유니그램과 바이그램은 N-그램을 간략화한 모형이라고 할 수 있다.

## 1. 조건부 확률 모형화 방법
### 1) 유니그램 모형
만약 모든 **단어의 활용이 완전히 서로 독립**이라면, 단어 열의 확률은 다음과 같이 **각 단어의 확률의 곱**이 된다. 이러한 모형을 **유니그램 모형**이라고 한다.
$$ P(w_1, w_2, \ldots, w_m) = \prod_{i=1}^m P(w_i) $$

> 너무 단순한 모델이기 때문에 잘 사용되지 않는다.

### 2) 바이그램 모형
만약 **단어의 활용이 바로 전 단어에만 의존**한다면 단어 열의 확률은 다음과 같다. 이러한 모형을 **바이그램 모형** 또는 **마코프 모형(Markov Model)**이라고 한다.
$$ P(w_1, w_2, \ldots, w_m) = P(w_1) \prod_{i=2}^{m} P(w_{i}\;|\; w_{i-1}) $$

> 딱 한 단어만 사용해 앞에 있는 문맥의 확률을 측정해서, 전체 문장의 확률을 예측한다.

### 3) N그램 모형
만약 **단어의 활용이 바로 전 $n-1$개의 단어에만 의존**한다면 단어 열의 확률은 다음과 같다. 이러한 모형을 **N그램 모형**이라고 한다.
$$ P(w_1, w_2, \ldots, w_m) = P(w_1) \prod_{i=n}^{m} P(w_{i}\;|\; w_{i-1}, \ldots, w_{i-n}) $$


## 1. NLTK의 N그램 기능
NLTK 패키지에는 바이그램과 N-그램을 생성하는 `bigrams`, `ngrams` 명령이 있다.

```py
from nltk import bigrams, word_tokenize
from nltk.util import ngrams

sentence = "I am a boy."
tokens = word_tokenize(sentence)

bigram = bigrams(tokens)
trigram = ngrams(tokens, 3) # n을 3으로 줌.

print("\nbigram:")
for t in bigram:
    print(t)

print("\ntrigram:")
for t in trigram:
    print(t)

'''
bigram:
('I', 'am')
('am', 'a')
('a', 'boy')
('boy', '.')

trigram:
('I', 'am', 'a')
('am', 'a', 'boy')
('a', 'boy', '.')
'''
```

### 특별토큰(SS, SE)
조건부 확률을 추정할 때는 문장의 시작과 끝이라는 조건을 표시하기 위해 모든 문장에 문장의 시작과 끝을 나타내는 특별 토큰을 추가한다. 예를 들어 **문장의 시작은 SS**, **문장의 끝은 SE** 이라는 토큰을 사용할 수 있다.

예를 들어 [“I”, “am”, “a”, “boy”, “.”]라는 토큰열(문장)은 [“SS”, “I”, “am”, “a”, “boy”, “.”, “SE”]라는 토큰열이 된다. `ngrams` 명령은 `padding` 기능을 사용하여 이런 특별 토큰을 추가할 수 있다.

```py
bigram = ngrams(tokens, 2, pad_left=True, pad_right=True, 
        left_pad_symbol="SS", right_pad_symbol="SE")
for t in bigram:
    print(t)
'''
('SS', 'I') # SS: 시작
('I', 'am')
('am', 'a')
('a', 'boy')
('boy', '.')
('.', 'SE') # SE: 끝
'''
```

## 2. 조건부 확률 추정 방법
NLTK 패키지를 사용하면 바이그램 형태의 조건부 확률을 쉽게 추정할 수 있다.

- ConditionalFreqDist 클래스로 각 **문맥별 단어 빈도를 추정**
- ConditionalProbdist 클래스로 **조건부 확률을 추정**

```py
from nltk import ConditionalFreqDist

sentence = "I am a boy."
tokens = word_tokenize(sentence)
bigram = ngrams(tokens, 2, pad_left=True, pad_right=True, 
        left_pad_symbol="SS", right_pad_symbol="SE")
cfd = ConditionalFreqDist([(t[0], t[1]) for t in bigram]) # 단어 빈도 추정

### ConditionalFreqDist 클래스는 문맥을 조건으로 가지는 사전 자료형과 비슷하다.

cfd.conditions()
# >['SS', 'I', 'am', 'a', 'boy', '.']

cfd["SS"]
# > FreqDist({'I': 1})
```

## 3. 바이그램 확률 추정 예제
아래에서는 nltk 패키지의 샘플 코퍼스인 `movie_reviews`의 텍스트로부터 바이그램 확률을 추정하는 방법을 코드와 같이 알아보겠다.

### 토큰화
```py
### Step1: 말뭉치 바이그램 토큰화
from nltk.corpus import movie_reviews
from nltk.util import ngrams
sentences = []
# pad_symbol은 문장의 '시작과 끝을 알려주는 스패셜 토큰'을 넣어준다.
for tokens in movie_reviews.sents():
    bigram = ngrams(tokens, 2, pad_left=True, pad_right=True,
                    left_pad_symbol="SS", right_pad_symbol="SE")
    sentences += [t for t in bigram]
sentences[:17]


### Step2: ConditionalFreqDist 클래스 객체 생성
from nltk import ConditionalFreqDist
cfd = ConditionalFreqDist(sentences)

cfd["SS"].most_common(5) 
# > [('the', 8071), ('.', 3173), ('it', 3136), ('i', 2471), ('but', 1814)]
cfd["SS"].plot(5, title = "문장 첫단어 분포")
plt.show()

cfd["i"].most_common(5)
# > [("'", 1357), ('was', 506), ('can', 351), ('have', 330), ('don', 276)]
cfd["i"].plot(5, title = "i 다음 오는 단어들의 분포")
plt.show()

cfd["."].most_common(5)
# > [('SE', 63404), ('"', 1854), (')', 535), ("'", 70), (']', 10)]
cfd["."].plot(5, title = ".(마침표) 다음에 올 수 있는 단어")
plt.show()
```

![](/assets/images/Preprocessing5_1.png)
![](/assets/images/Preprocessing5_2.png)
![](/assets/images/Preprocessing5_3.png)

### 빈도 추정
빈도를 추정하면 각각의 조건부 확률은 기본적으로 다음과 같이 추정할 수 있다.
$$P(w | w_{c}) = \frac{C((w_c, w))}{C((W_c))}$$

위 식에서 $C(w_c,w)$은 전체 말뭉치에서 $(w_c,w)$라는 바이그램이 나타나는 횟수이고 $C(w_c)$은 전체 말뭉치에서 $(w_c)$라는 유니그램(단어)이 나타나는 횟수이다. 

NLTK의 `ConditionalProbDist` 클래스에 `MLEProbDist` 클래스 팩토리를 인수로 넣어 위와 같이 빈도를 추정할 수 있다. (MLEProbdist: MaximumLikeliEstimaion)

- 학습이 끝나면 조건부 확률의 값을 보거나 샘플 문장을 입력해서 **문장의 로그 확률**을 구할 수 있다.

```py
### Step3: 조건부 확률 추정
from nltk.probability import ConditionalProbDist, MLEProbDist

cpd = ConditionalProbDist(cfd, MLEProbDist)

# cpd - 문맥, prob - 다음에 나오는 단어
print(cpd["i"].prob("am"), cpd["i"].prob("is"))
print(cpd["we"].prob("are"), cpd["we"].prob("is"))
# > 0.018562267971650354 0.0002249971875351558
#   0.08504504504504505  0.0
```

### 확률 계산

- 조건부 확률을 알게 되면 각 문장의 확률을 구할 수 있다.
- 토큰열을 N-그램 형태로 분해하고. 바이그램 모형에서는 전체 문장의 확률은 조건부 확률의 곱으로 나타난다.

```py
### Step4: 문장 확률 계산

def sentence_score(s):
    p = 0.0
    for i in range(len(s) - 1):
        c = s[i]
        w = s[i+1]
        p += np.log(cpd[c].prob(w) + np.finfo(float).eps) # finfo는 실수에서 사용할 수 있는 가장 작은 수
    return np.exp(p)

test_sentence = ["i", "like", "the", "movie", "."]
sentence_score(test_sentence)
# 말이 되는 문장의 경우 높은 확률이 나온다.
# > 2.740764134071561e-06

test_sentence = ["like", "i", "the", "movie", "."]
sentence_score(test_sentence)
# 말이 안 되는 문장은 확률이 굉장히 낮게 된다.
# > 1.0088151944997699e-20

test_sentence = ["hate", "this", "movie", "."]
sentence_score(test_sentence)
# > 0.0006470744497711514
```
### 무작위 문장 생성
- 처음에 "SS"에서 나올 수 있는 문장을 선택 만약 "I"를 발견했을 시
- "I"에서 나올 수 있는 문장을 선택 후 쭉 가다가 "SE"를 만났을 때 멈추고, 문장 생성

```py
### Step5: 무작위 문장 생성
def generate_sentence(seed=None):
    if seed is not None:
        import random
        random.seed(seed)
    c = "SS"
    sentence = []

    while True:
        if c not in cpd:
            break
        w = cpd[c].generate() # generate 함수 이용

        if w == "SE":
            break

        else:
            w2=w

        if c=="SS":
            sentence.append(w2.title())
        else:
            sentence.append(" " + w2)

        c = w
    return "".join(sentence)

generate_sentence(0)

# 출력 결과는 매번 달라진다.
```

## 4. 한글 자료를 이용한 코퍼스
이번에는 한글 자료를 이용해보자 코퍼스로는 아래의 웹사이트에 공개된 Naver sentiment movie corpus 자료를 사용한다.

[https://github.com/e9t/nsmc]

# 0. 데이터 다운로드
```py
import urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/e9t/nsmc/master/ratings_train.txt", filename="ratings_train.txt")

# 0-1. 코덱을 이용한 데이터 open
import codecs
with codecs.open("ratings_train.txt", encoding='utf-8') as f:
    data = [line.split('\t') for line in f.read().splitlines()]
    data = data[1:]   # header 제외

docs = [row[1] for row in data]
len(docs)
# > 150000

# 1. tokenizer 생성
import warnings
warnings.simplefilter("ignore")

from konlpy.tag import Okt

tagger = Okt()

def tokenize(doc):
    tokens = ['/'.join(t) for t in tagger.pos(doc)]
    return tokens

tokenize("그 영화는 아주 재미있었어요.")
# > ['그/Noun', '영화/Noun', '는/Josa', '아주/Noun', '재미있었어요/Adjective', './Punctuation']

# 2. n-그램 (바이 그램)
from tqdm import tqdm
sentences = []
for d in tqdm(docs):
    tokens = tokenize(d)
    bigram = ngrams(tokens, 2, pad_left=True, pad_right=True, left_pad_symbol="SS", right_pad_symbol="SE")
    sentences += [t for t in bigram]
sentences[:8]
'''
[('SS', '아/Exclamation'),
 ('아/Exclamation', '더빙/Noun'),
 ('더빙/Noun', '../Punctuation'),
 ('../Punctuation', '진짜/Noun'),
 ('진짜/Noun', '짜증나네요/Adjective'),
 ('짜증나네요/Adjective', '목소리/Noun'),
 ('목소리/Noun', 'SE'),
'''

# 3. 빈도 추정
cfd = ConditionalFreqDist(sentences)
cpd = ConditionalProbDist(cfd, MLEProbDist)

def korean_most_common(c, n, pos=None):
    if pos is None:
        return cfd[tokenize(c)[0]].most_common(n)
    else:
        return cfd["/".join([c, pos])].most_common(n)

korean_most_common("나", 10)
'''
[('는/Josa', 831),
 ('의/Josa', 339),
 ('만/Josa', 213),
 ('에게/Josa', 148),
 ('에겐/Josa', 84),
 ('랑/Josa', 81),
 ('한테/Josa', 50),
 ('참/Verb', 45),
 ('이/Determiner', 44),
 ('와도/Josa', 43)]
'''

# 4. 무작위 문자 생성
## 위에 방법 처럼 문장을 랜덤하게 생성할 수 있다. (코드 생략)
```

