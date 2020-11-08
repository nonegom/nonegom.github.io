---
title:  "[머신러닝] 데이터 전처리 (자연어)  - 4. 한국어 처리를 위한 Soynlp"
excerpt: "형태소 분석 없이 토큰화 기능을 제공하는 `soynlp`패키지 - Cohension, Branching Entropy, Accessor Variety"

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

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.01.04%20soynlp.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

# Soynlp 

`koNLPy`에서 제공하는 형태소 분석기는 새롭게 만들어진 미등록 단어들은 인식이 잘 되지 않으며, 이를 해결하기 위해 사용자 사전에 단어를 등록하는 절차를 거쳐야 한다. 이를 해결하기 위해 사용자 사전이나 형태소 분석 없이 `cohension` 기반으로 토큰화할 수 있는 기능을 제공하는 게 `soynlp` 패키지이다.  

- `soynlp`는 패키지를 설치해서 사용한다.
```py
pip install soynlp
```

## 1. 말뭉치 다운로드
soynlp는 koNLPy와 달리 패키지 내에서 말뭉치를 제공하지 않는다. 대신 github repo에 **예제 말뭉치 파일**이 있으므로 이를 다운로드 받아서 사용한다. 

이 파일은 하나의 문서가 한 줄로 되어 있고 각 줄 내에서 문장은 두 개의 공백으로 분리되어 있는 형식의 말뭉치이다. `DoublespaceLineCorpus`클래스로 이 말뭉치를 사용할 수 있다.

```py
# 0. 데이터 다운로드
import urllib.request
urllib.request.urlretrieve("https://raw.githubusercontent.com/lovit/soynlp/master/tutorials/2016-10-20.txt", filename="2016-10-20.txt")


# 1. 말뭉치 생성
from soynlp import DoublespaceLineCorpus

## 문서 단위 말뭉치 생성
corpus = DoublespaceLineCorpus("2016-10-20.txt")
len(corpus) # 문서의 갯수
# > 30091

## 문장 단위 말뭉치 생성
corpus = DoublespaceLineCorpus("2016-10-20.txt", iter_sent=True)
len(corpus) # 문장의 갯수
# > 223357
```

## 2. 단어 추출

`WordExtractor` 클래스를 사용하면 **형태소에 해당하는 단어를 분리하는 학습**을 수행한다. 실제로는 각 단어 후보에 대해 `cohesion` 방법을 사용 

`extract()` 메서드로 각 cohesion, branching entropy, accessor variety 등의 통계 수치를 계산할 수 있다.

```py
%%time
from soynlp.word import WordExtractor

word_extractor = WordExtractor()
word_extractor.train(corpus)

### 트레이닝이 끝나면, extractor를 통해 단어를 세가지 방법으로 통계 수치 계산
word_score = word_extractor.extract()
'''
all cohesion probabilities was computed. # words = 223348
all branching entropies was computed # words = 360721
all accessor variety was computed # words = 360721
'''
```

### 1) Cohesion
문자열을 글자단위로 분리하여 부분문자열(substring)을 만들 때 왼쪽부터 문맥을 증가시키면서 각 **문맥이 주어졌을 때 그 다음 글자가 나올 확률을 계산해 누적곱**을 한 값이다. **조건부 확률**의 값을 이용한다.
$$\text{cohesion}(n) = \left( \prod_{i=1}^{n-1} P(c_{1:i+1} | c_{1:i}) \right)^{\frac{1}{n-1}}$$

하나의 단어를 중간에서 나눈 경우, 다음 글자를 예측하기 쉬우므로 조건부확률의 값은 크다. 하지만 **단어가 종료된 다음**에 여러가지 조사나 결합어가 오는 경우에는 다양한 경우가 가능하므로 조건부확률의 값이 작아진다. 따라서 `cohesion`값이 **가장 큰 위치가 하나의 단어**를 이루고 있을 **가능성이 높다**.

```py
word_score["연합"].cohesion_forward
# > 0.1943363253634125

word_score["연합뉴"].cohesion_forward
# > 0.43154839105434084

word_score["연합뉴스"].cohesion_forward
# > 0.5710254410737682

word_score["연합뉴스는"].cohesion_forward
# > 0.1535595043355021
```

### 2) Branching Entropy
조건부 확률의 값이 아니라 **확률분포의 엔트로피값**을 사용한다. 

만약 하나의 단어를 중간에서 끊으면 다음에 나올 글자는 쉽게 예측이 가능하다.
즉, 여러가지 글자 중 특정한 하나의 글자가 확률이 높다. 따라서 엔트로피값이 0에 가까운 값으로 작아진다. 하지만 **하나의 단어가 완결되는 위치**에는 다양한 조사나 결합어가 올 수 있으므로 **여러가지 글자의 확률이 비슷하게 나오고 따라서 엔트로피값이 높아진다**.

```py
word_score["연합"].right_branching_entropy
# > 0.42721236711742844

## '연합뉴' 다음에는 항상 '스'만 나온다.
word_score["연합뉴"].right_branching_entropy
# > -0.0

word_score["연합뉴스"].right_branching_entropy
# > 3.8967810761022053

word_score["연합뉴스는"].right_branching_entropy
# > 0.410116318288409
```

### 3) Accessor Variety
확률분포를 구하지 않고 단순히 **특정 문자열 다음에 나올 수 있는 글자의 종류만 계산**한다. 글자의 종류가 많다면 엔트로피가 높아지리 것이라고 **추정**하는 것이다.

```py
word_score["연합"].right_accessor_variety
# > 42

## '연합뉴' 다음에는 항상 '스'만 나온다.
word_score["연합뉴"].right_accessor_variety
# > 1

word_score["연합뉴스"].right_accessor_variety
# > 158

word_score["연합뉴스는"].right_accessor_variety
# > 2
```

`soynlp`는 이렇게 계산된 통계수치를 사용하여 **문자열을 토큰화하는 방법도 제공**한다. soynlp가 제공하는 토큰화 방법은 두 가지다.

- 띄어쓰기가 잘 되어 있는 경우: **L-토큰화**

- 띄어쓰기가 안 되어 있는 경우: **Max Score 토큰화**

## 3. L-토큰화
우리나라에서만 가능한 토큰화 방법이다. 한국어의 경우 공백(띄어쓰기)으로 분리된 하나의 문자열은 'L 토큰 + R 토큰' 구조인 경우가 많다. 왼쪽에 오는 L토큰은 체언(명사, 대명사)이나 동사, 형용사 등이고 오른쪽에 오는 R토큰은 동사, 형용사, 조사 등이다. 여러가지 길이의 L토큰 점수를 비교해 가장 **점수가 높은 L 단어를 찾는 것**이 **L-토큰화(L-Tokenizing)**이다. soynlp에서는 `LTokenizer`클래스로 제공한다.

```py
from soynlp.tokenizer import LTokenizer

scores = {word:score.cohesion_forward for word, score in word_score.items()}
l_tokenizer = LTokenizer(scores=scores)

l_tokenizer.tokenize("안전성에 문제있는 스마트폰을 휴대하고 탑승할 경우에 압수한다", flatten=False)

'''
[('안전', '성에'),
 ('문제', '있는'),
 ('스마트폰', '을'),
 ('휴대', '하고'),
 ('탑승', '할'),
 ('경우', '에'),
 ('압수', '한다')]
'''
```

## 4. 최대 점수 토큰화
**띄어쓰기가 되어 있지 않는 긴 문자열**에서 가능한 모든 종류의 부분문자열을 만들어서 **가장 점수가 높은 것을 하나의 토큰**으로 정한다. 이 **토큰을 제외하면** 이 위치를 기준으로 전체 문자열이 다시 더 작은 문자열들로 나누어지는데 **이 문자열들에 대해** 다시 한번 **가장 점수가 높은 부분문자열**을 찾는 것을 반복한다.

```py
from soynlp.tokenizer import MaxScoreTokenizer

maxscore_tokenizer = MaxScoreTokenizer(scores=scores)
maxscore_tokenizer.tokenize("안전성에문제있는스마트폰을휴대하고탑승할경우에압수한다")

'''
['안전',
 '성에',
 '문제',
 '있는',
 '스마트폰',
 '을',
 '휴대',
 '하고',
 '탑승',
 '할',
 '경우',
 '에',
 '압수',
 '한다']
'''
```