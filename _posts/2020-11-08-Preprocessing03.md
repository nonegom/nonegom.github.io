---
title:  "[머신러닝] 데이터 전처리 (자연어)  - 3. Scikit-Learn의 문서 전처리 기능"
excerpt: "Scikit-Learn의 문서 전처리 클래스 - CountVectorizer, TFidfVectorizer, HashongVectorizer"

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

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.01.03%20Scikit-Learn%EC%9D%98%20%EB%AC%B8%EC%84%9C%20%EC%A0%84%EC%B2%98%EB%A6%AC%20%EA%B8%B0%EB%8A%A5.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 0. BOW 인코딩
문서를 숫자 벡터로 변환하는 가장 기본적인 방법이 바로 **BOW(Bag of Words)**인코딩 방법이다. BOW 인코딩 방법에서는 {$d_1, d_2, \ldots, d_n$}를 구성하는 고정된 단어장 {$t_1, t_2, \ldots, t_m$}를 만들고 $d_i$라는 **개별 문서**에 **단어장**에 해당하는 단어들이 포함되어 있는지 표시하는 방법이다.
$x_{i, j}$는 아래와 같이 표시한다.

$$ x_{i,j} = \text{문서 $d_i$내의 단어 $t_j$의 출현 빈도} $$
$$
\begin{split} x_{i,j} = 
\begin{cases}
0, & \text{만약 단어 $t_j$가 문서 $d_i$ 안에 없으면} \\
1, & \text{만약 단어 $t_j$가 문서 $d_i$ 안에 있으면}
\end{cases}
\end{split}$$


## 1. Scikit-Learn의 문서 전처리 기능

Scikit-Learn의 `feature_extraction` 서브패키지와 `feature_extraction.text`서브패키지는 다음과 같은 **문서 전처리용 클래스**를 제공한다.

- `DictVectorizer`:
    - 각 단어의 수를 세어놓은 사전에서 BOW 인코딩 벡터를 만든다.
- `CountVectorizer`:
    - 문서 집합에서 단어 토큰을 생성하고 각 단어의 수를 세어 BOW인코딩 벡터를 만든다.
- `TFidfVectorizer`:
    - CountVectorizer와 비슷하지만 TF-IDF방식으로 단어의 가중치를 조정한 BOW인코딩 벡터를 만든다.
- `HashongVectorizer`:
    - 해시함수를 사용하여 적은 메모리와 빠른 속도로 BOW인코딩 벡터를 만든다.

### Vectorizer 클래스 사용법

1. 클래스 객체 생성
2. 말뭉치를 넣고 `fit`메서드 실행
3. `voacbulary_`속성에 단어장이 자동 생성됨
4. `transform` 메서드로 다른 문서를 **BOW인코딩**
5. BOW인코딩 결과는 **Spare행렬**로 만들어지므로 `toarray` 메서드로 **보통 행렬로 변환**시킨다.

- Sapre행렬은 0인 값은 제외하고, 값이 있는 것에만 데이터가 할당된 행렬 (데이터를 효율적으로 처리하기 위해서)

## 2. DictVectorizer 
`DictVectorizer` 는 `feature_extraction` 서브패키지에서 제공한다. 문서에서 단어의 사용 빈도를 나타내는 딕셔너리 정보를 입력받아 BOW 인코딩한 수치 벡터로 변환한다.

- 단어와 빈도가 리스트 안에 dict 형태로 나타나있어야 한다. `dictVectorizer`를 만드려면 문서의 단어를 dict 형식으로 만들어야 한다.
- 코퍼스를 만들 때 없었던 단어는 추후에 나타나도 없는 것으로 간주한다.
- 따라서 사실상 **쓰이기가 힘들다.**

```py
from sklearn.feature_extraction import DictVectorizer
vec1 = DictVectorizer(sparse=False)
text = [{'A': 1, 'B': 2}, {'B': 3, 'C': 1}]
X = vec1.fit_transform(text)
'''
array(
  [[1., 2., 0.],
   [0., 3., 1.]])
'''

vec1.feature_names_
# ['A', 'B', 'C']

vec1.transform({'C': 4, 'D': 3})
# array([[0., 0., 4.]])
## D는 코퍼스를 만들 때 없었으므로, 없는 것으로 간주한다.
```

## 3. CountVectorizer
`CountVectorizer` 는 세 가지 작업을 수행한다.
1. 문서를 **토큰 리스트로 변환**한다.
2. 각 문서에서 **토큰의 출현 빈도**를 센다.
3. 각 문서를 **BOW 인코딩 벡터로 변환**한다.

### 예제 코드
```py
### Step1: 말뭉치만들기
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
'This is the first document.',
'This is the second second document.',
'And the third one.',
'Is this the first document?',
'The last document?',
]
# 사전을 만들 때 최겡 최대한 많은 단어를 넣어둬야 한다.

### Step2:인코더 객체 생성
vect = CountVectorizer()

### Step3: 말뭉치 학습 및 단어장 생성
vect.fit(corpus)
vect.vocabulary_ # - CountVectorizer가 만든 사전
'''
{'this': 9, # 수는 단어의 빈도수가 아닌, 사전 내 인덱스의 위치이다. 
 'is': 3,
 'the': 7,
 'first': 2,
 'document': 1,
 'second': 6,
 'and': 0,
 'third': 8,
 'one': 5,
 'last': 4}
'''

### Step4: 문장을 BOW인코딩
vect.transform(["This is the second document."]).toarray()
# > array([[0, 1, 0, 1, 0, 0, 1, 1, 0, 1]]) 원핫 인코딩처럼 단어가 있는 위치에 True 설정
```

### 생각해볼 문제
- 단어가 많아져 `vocabluary_`의 개수가 많아지게 되면,
- 위처럼 단어장 생성을 하지 않고 `HashVectorizer`를 사용

### CountVectorizer의 인수

- `stop_words` : 문자열 {‘english’}, 리스트 또는 None (디폴트). 
    - stop words 목록.‘english’이면 영어용 스탑 워드 사용.

- `analyzer` : 문자열 {‘word’, ‘char’, ‘char_wb’} 또는 함수
    - **단어** n-그램, **문자** n-그램, 단어 내의 문자 n-그램

- `token_pattern` : string
    - 토큰 정의용 **정규 표현식**

- `tokenizer` : 함수 또는 None (디폴트)
    - 토큰 **생성 함수**

- `ngram_range` : (min_n, max_n) 튜플
    - n-그램 범위

- `max_df` : 정수 또는 [0.0, 1.0] 사이의 실수. 디폴트 1
    - 단어장에 포함되기 위한 최대 빈도

- `min_df` : 정수 또는 [0.0, 1.0] 사이의 실수. 디폴트 1
    - 단어장에 포함되기 위한 최소 빈도

#### Stop Words (불용어) 
Stop Words 는 문서에서 단어장을 생성할 때 무시할 수 있는 단어를 말한다. 보통 영어의 관사나 접속사, 한국어의 조사 등이 여기에 해당한다. `stop_words` 인수로 조절.

```py
### Step5: 불용어 제외
vect = CountVectorizer(stop_words=["and", "is", "the", "this"]).fit(corpus)
vect.vocabulary_
# > {'first': 1, 'document': 0, 'second': 4, 'third': 5, 'one': 3, 'last': 2}

vect = CountVectorizer(stop_words="english").fit(corpus)
vect.vocabulary_
# > {'document': 0, 'second': 1}
```

#### 빈도수
문서에서 토큰이 나타난 횟수를 기준으로 단어장을 구성할 수도 있다. 토큰의 빈도가 max_df로 지정한 값을 초과 하거나 min_df로 지정한 값보다 작은 경우에는 무시한다. 인수 값은 **정수인 경우 횟수**, **부동소수점인 경우 비중**을 뜻한다.

```py
### Step6: 빈도수
vect = CountVectorizer(max_df=4, min_df=2).fit(corpus)
vect.vocabulary_, vect.stop_words_
# > ({'this': 3, 'is': 2, 'first': 1, 'document': 0},
# > {'and', 'last', 'one', 'second', 'the', 'third'})
```

#### N그램
N그램은 단어장 생성에 사용할 토큰의 크기를 결정한다. **모노그램**(monogram)은 토큰 하나만 단어로 사용하며 **바이그램**(bigram)은 두 개의 연결된 토큰을 하나의 단어로 사용한다.

```py
### Step7: N-그램 적용
vect = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
vect.vocabulary_
'''
{'this is': 12,
 'is the': 2,
 'the first': 7,
 'first document': 1, ...
'''
```

#### 토큰
`analyzer`, `tokenizer`, `token_pattern` 등의 인수로 사용할 토큰 생성기를 선택할 수 있다.

```py
vect = CountVectorizer(analyzer="char").fit(corpus)
vect.vocabulary_
'''
{'t': 16,
 'h': 8,
 'i': 9,
 's': 15, ...
'''

vect = CountVectorizer(token_pattern="t\w+").fit(corpus)
vect.vocabulary
'''
{'this': 2, 'the': 0, 'third': 1}
'''

import nltk

vect = CountVectorizer(tokenizer=nltk.word_tokenize).fit(corpus)
vect.vocabulary_
'''
{'this': 11,
 'is': 5,
 'the': 9,
 'first': 4,
 'document': 3,
 '.': 0, ...
'''
```

## 4. TF-IDF
TF-IDF(Term Frequency – Inverse Document Frequency) 인코딩은 단어를 갯수 그대로 카운트하지 않고 **모든 문서에 공통적으로 들어있는 단어의 경우** 문서 구별 능력이 떨어진다고 보아 **가중치를 축소**하는 방법이다.
- 중요도와 비율이 적용된 값이 들어간다.

구체적으로 문서 $d$와 단어 $t$에 대해 아래와 같이 계산한다.
$$\text{tf-idf}(d, t) = tf(d, t) \cdot idf(t) $$

여기서 
- tf($d,t$): 특정한 단어의 빈도수 term frequency이고
- idf($t$): 특정한 단어가 들어 있는 문서의 수에 반비례하는 수 inverse document frequency
$$\text{idf(d,t)} = log \frac{1+\text{df}(t)}{n}$$

- $n$: 전체 ㅁ누서의 수
- df($t$): 단어 $t$를 가진 문서의 수

```py
from sklearn.feature_extraction.text import TfidfVectorizer
tfidv = TfidfVectorizer().fit(corpus)
tfidv.transform(corpus).toarray()
```

![](/assets/images/Preprocessing3_1.png)

## 5. Hahsging Trick
`CountVectorizer` 는 모든 작업을 메모리 상에서 수행하므로 처리할 문서의 크기가 커지면 속도가 느려지거나 실행이 불가능해진다. 이 때 `HashingVectorizer`를 사용하면 해시 함수를 사용하여 **단어에 대한 인덱스 번호를 생성**하기 때문에 메모리 및 실행 시간을 줄일 수 있다.

- 해시함수: 글자를 집어넣으면, 자체적인 알고리즘을 돌려서 숫자를 출력한다.


**특징**
- 단어에 대한 인덱스 번호를 수식으로 생성
- 사전 메모리가 없고 실행시간 단축가능
- 단어의 충돌이 발생할 수 있음

```py
from sklearn.datasets import fetch_20newsgroups
twenty = fetch_20newsgroups()
len(twenty.data)
# > 11314

# CountVectorizer 시간
%time
CountVectorizer().fit(twenty.data).transform(twenty.data)
# > Wall time: 20.3 s...

# HashingVectorizer 시간
from sklearn.feature_extraction.text import HashingVectorizer
hv = HashingVectorizer(n_features=300000)

%time hv.transform(twenty.data)
# > Wall time: 3.9 s

## HashingVector를 사용했을 때 확실히 시간이 단축된을 확인할 수 있다.
```
## 6. Gensim 패키지
DIctionary클래스 이용한다. TFidfModel클래스를 이용시 TF-IDF인코딩도 가능
- 주제를 가져다가 토픽을 뽑아낼 수 있는 **토픽 모델링**에 사용될 수 있다.

```py
### Step1: 말뭉치 만들기
corpus = [
'This is the first document.',
'This is the second second document.',
'And the third one.',
'Is this the first document?',
'The last document?',
]

### Step2: 토큰 리스트 생성
# split명령을 이용한 torkenizer
token_list = [[text for text in doc.split()] for doc in corpus]
token_list

### Step3: Dictionary 객체생성
from gensim.corpora import Dictionary

dictionary = Dictionary(token_list)
dictionary.token2id # voca가 token2id로 인덱스 된다.

### Step4: BOW인코딩
term_matrix = [dictionary.doc2bow(token) for token in token_list]
term_matrix
## spots Matrix에는 각 단어가 몇 번 나왔는지 나타난다.
'''
[[(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)],
 [(0, 1), (1, 1), (3, 1), (4, 1), (5, 2)],
 [(4, 1), (6, 1), (7, 1), (8, 1)],
 [(2, 1), (4, 1), (9, 1), (10, 1), (11, 1)],
 [(10, 1), (12, 1), (13, 1)]]
'''

### Step5: TF-IDF인코딩
from gensim.models import TfidfModel
    
# CountMatrix를 TfidfModel에 넣어주면 변하게 된다.
tfidf = TfidfModel(term_matrix)

for doc in tfidf[term_matrix]:
    print("doc:")
    for k, v in doc:
        print(k, v)

# idf가 곱해져서 스케일링된 값을 확인할 수 있다.
```

## 7. 토픽 모델링

```py
### Step1: 텍스트 데이터 다운로드
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(
    categories= ["comp.graphics", "rec.sport.baseball", "sci.med"])

### Step2: 명사 추출
%%time
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

tagged_list = [pos_tag(word_tokenize(doc)) for doc in newsgroups.data]
nouns_list = [[t[0] for t in doc if t[1].startswith("N")] for doc in tagged_list]

### Step3: 표제어 추출
from nltk.stem import WordNetLemmatizer

lm = WordNetLemmatizer()
nouns_list = [[lm.lemmatize(w, pos="n") for w in doc] for doc in nouns_list]

### Step4: 불용어 제거
import re
token_list = [[text.lower() for text in doc] for doc in nouns_list]
token_list = [[re.sub(r"[^A-za-z]+", '', word) for word in doc] for doc in token_list]
```

## 8. 웹사이트 내 특정 단어 빈도 
```py
import warnings
warnings.simplefilter("ignore")

import json
import string
from urllib.request import urlopen
from konlpy.utils import pprint
from konlpy.tag import Hannanum

# 1. 데이터 로드
hannanum = Hannanum()

f = urlopen("https://www.datascienceschool.net/download-notebook/708e711429a646818b9dcbb581e0c10a/")
json = json.loads(f.read())

cell = ["\n".join(c["source"]) for c in json["cells"] if c["cell_type"] == "markdown"]
docs = [
    w for w in hannanum.nouns(" ".join(cell)) 
    if ((not w[0].isnumeric()) and (w[0] not in string.punctuation))
]
``` 

여기에서는 하나의 문서가 하나의 단어로만 이루어져 있다. 따라서 `CountVectorizer`로 이 문서 집합을 처리하면 각 문서는 **하나의 원소만 1이고 나머지 원소는 0인 벡터**가 된다. 이 **벡터의 합으로 빈도**를 알아보았다.

```py
# 2. 전처리 및 데이터 분석
vect = CountVectorizer().fit(docs)
count = vect.transform(docs).toarray().sum(axis=0)
idx = np.argsort(-count)
count = count[idx]

feature_name = np.array(vect.get_feature_names())[idx]
plt.bar(range(len(count)), count)
plt.show()
```

![](/assets/images/Preprocessing3_2.png)

```py
# 3. 빈도수 확인
pprint(list(zip(feature_name, count))[:10])
'''
[('컨테이너', 83),
 ('도커', 41),
 ('명령', 34),
 ('이미지', 34),
 ('사용', 26),
 ('가동', 14),
 ('중지', 13),
 ('mingw64', 13),
 ('다음', 12),
 ('삭제', 12)]
'''
```