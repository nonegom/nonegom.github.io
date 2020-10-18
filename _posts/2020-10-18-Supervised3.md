---
title:  "[머신러닝] 지도학습 - 3.2. 나이브베이즈 분류모형"
excerpt: "생성모형 중 나이브베이즈 분류 모형"

categories:
  - MachinLearning
tags:
  - SupervisedLearning
  - 10월
toc: true
toc_sticky: true
toc_label: 페이지 목차
use_math: true
---
> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/11.02%20%EB%82%98%EC%9D%B4%EB%B8%8C%EB%B2%A0%EC%9D%B4%EC%A6%88%20%EB%B6%84%EB%A5%98%EB%AA%A8%ED%98%95.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

## 0.조건부 독립
확률변수 A, B가 독립이면 A,B의 결확확률은 주변확률의 곱과 같다.(일반적인 독립 - 무조건부독립)
$$P(A,B) = P(A)P(B)$$

**조건부 독립**은 조건이 되는 **별개의 확률변수 C**가 존재해야 독립이 됨을 의미한다. A, B의 **결합 조건부 확률**이 C에 대한 A,B의 조건부확률의 곱과 같으면 A와 B가 C에 대해 **조건부독립**이라고 한다.
$$P(A, B|C) = P(A|C)P(B|C)$$

- 조건부독립과 비교해 일반적인 독립은 **무조건부독립**이라고 한다.

만약, A, B가 C에 대해 조건부독립이면 다음도 만족한다.
$$P(A|B, C) = P(A|C)$$
$$P(B|A, C) = P(B|C)$$
- 위 식은 나중에 '확률적 그래프 모델'에서 사용된다.

위 식을 다시 말하자면, C의 값이 주어졌을 경우 A에 대한 확률을 구하는데 B값은 필요없다는 뜻이다. 또한 B의 확률을 구하는데도 A의 값은 필요없다.
 
- 주의할 점은 **조건부독립**과 **무조건부독립**은 **관계가 없다**는 점이다. 두 확률변수가 독립이라고 항상 조건부독립이 되는 것도 아니고 조건부독립이라고 꼭 독립이 되는 것도 아니다.

### 예제 데이터 
조건부독립의 관계를 다음의 예제를 통해 알아보자.
한 동물의 어미와 새끼들로 구성된 무리가 있다고 가정하자.  
어미의 몸무게는 $x$를 기댓값으로 하고, $5kg$으로 표준편차를 가진다.  
이 동물의 새끼 중 2마리의 몸무게를 각각 A, B라고 하고 어미의 몸무게를 C라고 하자. 이 경우 어미 표본과 각각의 어미에 대해 2마리의 새끼 표본을 만들어보도록 
하자.

- 만약 '어미의 몸무게'를 모른다면, A와 B는 무조건부 상관관계를 갖는다.
- 여기서 어미의 몸무게를 알고 있다면, 기존 A와 B의가 가지고있던 몸무게에 대한 상관관계가 없어진다.
- 즉, C가 주어지면 A와 B의 상관관계는 없어진다.

### 예제 코드
``` py

## 예제 데이터 생성
np.random.seed(0)
C = np.random.normal(100, 15, 2000)
A = C + np.random.normal(0, 5, 2000) # 형 개체
B = C + np.random.normal(0, 5, 2000) # 동생 개체

## 무조건부 상관관계일 때와 조건부 상관관계일 때

plt.figure(figsize=(8, 4))

plt.subplot(121)
plt.title("B와 C의 무조건부 상관관계")
plt.scatter(A, B)
plt.xlabel("A")
plt.ylabel("B")
plt.xlim(30, 180)
plt.ylim(30, 180)

plt.subplot(122)
plt.title("B와 C의 조건부 상관관계")
idx1 = (118 < C) & (C < 122)
idx2 = (78 < C) & (C < 82)
plt.scatter(A[idx1], B[idx1], label="C=120")
plt.scatter(A[idx2], B[idx2], label="C=80")
plt.xlabel("A")
plt.ylabel("B")
plt.xlim(30, 180)
plt.ylim(30, 180)
plt.legend()
plt.tight_layout()
plt.show()
```
![](/assets/images/Supervised3_1.png){: .align-center}

## 1. 나이브 가정
- 배경

 독립변수 x가 D차원($x = (x_1, \dots, x_D)$)이라고 가정할 때, '가능도함수'는 $P(x \mid y=k) = P(x_1, \dots, x_D \mid y= k)$가 된다. 하지만**차원 D가 커지면** 가능도함수의 추정이 현실적으로 어려워진다.
따라서 나이즈베이즈 분류모형에서는 **모든 차원의 개별 독립변수가 서로 조건부 독립이라는 가정**을 사용한다. 이러한 가정을 **나이브 가정**이라고 한다.

- 나이브 가정을 사용하면 벡터 **$x$의 결합확률분포함수**는 **개별 스칼라 원소 $x_d$의 확률분포함수의 곱**이 된다.
- 스칼라 원소 $x_d$의 확률분포함수는 결확확률분포함수보다 추정하기가 훨씬 쉽다.

### 나이브 가정
$$P(x_1, ..., x_D | y=k) =\prod^D_{d=1} P(x_d | y=k)$$

- 그리고 역시 **가능도 함수**를 추정한 후에는 베이즈정리를 사용해 조건부 확률을 계산할 수 있다.

### 나이브베이즈 분류 모형
$$P(y=k|x$) = {\left( \prod^D_{d=1} P(x_d | y=k)\right)P(y=k)\over P(x)}$$

## 2. 나이브베이즈 분류 모형의 종류

### 1) 정규분포 가능도 모형
 **x벡터의 원소가 모두 실수**이고 **클래스마다 특정한 값 주변에서 발생**한다고 하면 가능도 분포로 **'정규분포'**를 사용한다. 각 독립변수 $x_d$마다, 그리고 클래스 k마다 정규 분포의 기댓값$\mu_{d,k}$ , 표준 편차 $\sigma_{d,k}^2$가 달라진다. 
- QDA 모형과는 달리 모든 독립변수들이 서로 조건부독립이라고 가정한다.

- 이 때, 가능도함수는 '스칼라' 즉, **단변수 정규분포를 곱해놓은 값**이 된다. 왜냐하면 앞선 모델과 달리 표준편차가 다 달라지기 때문에 행렬 값이 아니게 된다. 

### 2) 베르누이 분포 가능도 모형
 베르누이분포 가능도 모형에서는 각각의 $x = (x_1, \dots, x_D)$의 각 원소 $x_d$가 0 또는 1이라는 값만을 가질 수 있다. 독립변수는 D개의 독립적인 **베르누이 확률변수**, 즉 동전(x)으로 구성된 동전세트(y)로 표현할 수 있다. 이 동전들의 모수 $\mu_d$는 동전마다 다르다. 또한 클래스 $y= k(k=1, \dots, K)$마다도 확률이 다르다. 그러므로 동전들의 모수 $\mu_{d, k}$는 동전($d$)마다 다르고 클래스 ($k$)마다도 다르다.
 - 결론적으로 전체 $D\ X\ K$의 동전이 존재하며 같은 클래스에 속하는 $D$개의 동전이 하나의
동전 세트를 구성하고 이러한 동전 세트가 $K$개가 있다고 생각할 수 있다.

 따라서 베르누이분포 가능도 모형을 기반으로 하는 나이브베이즈 모형은 **동전 세트를 N번 던진 결과로 부터 $1, ..., K$ 중 어느 동전 세트를 던졌는지를 찾아내는 모형**이라고 할 수 있다.

### 3) 다항분포 가능도 모형
 $x$벡터가 **다항분포의 표본**이라고 가정한다. 즉 D개의 면을 가지는 주사위를 $\Sigma^D_{d=1}x_d$번 던져서 나온 결과로 본다.
 - 다항분포 가능도 모형을 기반으로 하는 나이브베이즈 모형은 주사위를 던진 결과로부터 $1, ..., K$ 중 어느 주사위를 던졌는지를 찾아내는 모형이라고 할 수 있다.

### 사이킷런에서 제공하는 나이브베이즈 모형
사이킷런의 `baive_bayes` 서브패키지
- `GaussianNB`(정규분포 나이브베이즈): x값이 실수일 때
- `BernoulliNB`(베르누이분포 나이브베이즈): x값이 0, 1일 때
- `MuntinomialNB`(다항분포 나이브베이즈): x값이 정수로 되어있을 때

#### 클래스 속성값
- classes_: 종속변수 Y의 클래스(라벨)
- class_count_: 종속변수 Y의 값이 특정한 클래스인 표본 데이터의 수
- class_prior_: 종속변수 Y의 무조건부 확률분포 (정규분포의 경우에만) [= 사전확률]
- class_log_prior_: 종속변수 Y의 무조건부 확률분포의 로그 (베르누이분포나 다항분포의 경우에만) [= 로그값을 취한 사전확률]

## 3. 정규분포 나이브베이즈 모형
가우시안 나이브베이즈 모형 `GaussianNB`은 가능도 추정과 관련해 다음과 같은 속성을 가진다.
- theta_: 정규분포의 기댓값 $\mu$벡터값
- sigma_: 정규분포의 분산 $\sigma^2$

### 예제 1
- 실수인 두 개의 독립변수 $x_1, x_2$와 두 종류의 클래스 $y=0, 1$을 가지는 분류문제
- 두 독립변수의 분포는 **정규분포**이고 y의 클래스에 따라 모수(공분산행렬)가 달라진다.
- 데이터는 $y=0$인 데이터가 40개, $y=1$인 데이터가 60개 주어졌다.

```py

## 데이터 생성
np.random.seed(0)
rv0 = sp.stats.multivariate_normal([-2, -2], [[1, 0.9], [0.9, 2]])
rv1 = sp.stats.multivariate_normal([2, 2], [[1.2, -0.8], [-0.8, 2]])
X0 = rv0.rvs(40)
X1 = rv1.rvs(60)
X = np.vstack([X0, X1])
y = np.hstack([np.zeros(40), np.ones(60)])

## 가우시안 나이브베이즈 모형
from sklearn.naive_bayes import GaussianNB
model_norm = GaussianNB().fit(X, y)

# 클래스 종류 (2개)
model_norm.classes_

# 각 클래스 별 데이터의 개수 (40, 60)
model_norm.class_count_

# 사전확률 (x를 안 보여줬을 때의 확률)
model_norm.class_prior_

# 각 클래스별 기댓값과 분산
model_norm.theta_[0], model_norm.sigma_[0]
model_norm.theta_[1], model_norm.sigma_[1]

# 각 클래스별 확률분포 생성
rv0 = sp.stats.multivariate_normal(model_norm.theta_[0], model_norm.sigma_[0])
rv1 = sp.stats.multivariate_normal(model_norm.theta_[1], model_norm.sigma_[1])

```

### 모델을 통한 $x$데이터에 따른 $y$값 예측
`predict_proba`메서드로 각 클래스의 값이 나올 확률을 구할 수 있다. (값이 나오게 되는 과정 생략)

```py
### 위 코드에서 이어짐

x_new = [0, 0] # 입력 데이터 x값.
model_norm.predict_proba([x_new])
# > array([[0.48475244, 0.51524756]])
### x가 [0,0]일 때, y=0일 확률은 '0.48', y=1일 확률은 '0.52'이다.
```

## 4. 베르누이분포 나이브베이즈 모형
전자우편과 같은 **문서 내에 특정한 단어가 포함되어 있는지 여부**는 **베르누이 확률변수**로 모형화할 수 있다.
- `feature_count`: 각 클래스 k에 대해 d번째 동전이 앞면(x=1)이 나온 횟수 $N_{d,k}$
- `feature_log_prob`: 베르누이분포 모수의 로그

여기에서 N_k은 클래스 k에 대해 동전을 던진 총 횟수이다. 표본 데이터의 수가 적은 경우에는 모수에 대해 다음처럼 **스무딩**할 수도 있다.

### 손으로 풀 수 있어야 한다.

### 스무딩(Smoothing)
 표본 데이터의 수가 적은 경우 베르누이 모수가 0또는 1이라는 극단적인 모수 추정값이 나올 수도 있지만, 보통은 그럴 가능성이 적다. 따라서 베르누이 모수가 0.5인 가장 일반적인 경우를 가정하여 0이 나오는 경우와 1이 나오는 경우, 두 개의 **가상 표본 데이터(가짜 데이터)를 추가**한다.
 
- 그러면 0이나 1과 같은 극단적인 추정값이 0.5에 가까운 값으로 변한다. 이를 **'라플라스 스무딩'** 또는 **'애드원 스무딩'**이라고 한다. 
 $$\hat\mu_{d_k} = \frac {N_{d,k} + \alpha}{N_{k} + 2\alpha}$$

- 여기서 데이터를 얼마나 섞을 지 **가중치**$\alpha$라는 값으로 스무딩의 정도를 조절할 수도 있다. 만약 가중치 $\alpha$값이 1인 경우는 **무정보 사전확률**을 사용한 베이즈 모수추정의 결과와 같다.

### 예제 2
- 4개의 키워드를 사용해 정상 메일(y=0) 4개와 스팸 메일(y=1) 6개를 BOW인코딩한 데이터이다.
- 예를 들어 첫번째 메일은 '정상메일'이고, 1번, 4번 키워드는 포함하지 않지만 2번, 3번 키워드를 포함한다고 볼 수 있다.

```py
### 예제 데이터
X = np.array([
[0, 1, 1, 0],
[1, 1, 1, 1],
[1, 1, 1, 0],
[0, 1, 0, 0],
[0, 0, 0, 1],
[0, 1, 1, 0],
[0, 1, 1, 1],
[1, 0, 1, 0],
[1, 0, 1, 1],
[0, 1, 1, 0]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

## 모델 생성
from sklearn.naive_bayes import BernoulliNB
model_bern = BernoulliNB().fit(X, y)

## 클래스의 속성
model_bern.classes_
model_bern.class_count_
np.exp(model_bern.class_log_prior_) # 로그값 취한 사전확률

## 각 클래스 k별, 각 독립변수 d별 베르누이 확률변수의 모수
fc = model_bern.feature_count_
"""
> 출력값
array([[2., 4., 3., 1.],
      [2., 3., 5., 3.]])
"""
## 스무딩이 적용안된 베르누이 모수값
fc / np.repeat(model_bern.class_count_[:, np.newaxis], 4, axis=1)

## 스무딩이 적용된 베르누이 모수값
np.exp(model_bern.feature_log_prob_)

### 테스트 1) 1번, 2번 키워드 포함

x_test1 = np.array([1, 1, 0, 0])
model_bern.predict_proba([x_test1])
# > array([[0.72480181, 0.27519819]])
## 정상메일일 확률이 3배 가량 높다.

### 테스트 1) 3번, 4번 키워드 포함

x_test2 = np.array([0, 0, 1, 1])
model_bern.predict_proba([x_test1])
# > array([[0.09530901, 0.90469099]])
## 스펨메일일 확률이 90% 가량.

```

## 5. 다항분포 나이브베이즈 모형
다항분포 나이브베이즈 모형 클래스 `MultinomialNB` 는 가능도 추정과 관련하여 다음 속성을 가진다.

- `feature_count_` : 각 클래스 에서 번째 면이 나온 횟수
- `feature_log_prob_` : 다항분포의 모수의 로그

여기에서 $N_k$은 클래스 $k$에 대해 주사위를 던진 총 횟수를 뜻한다. (스무딩 적용)

### 예제 3
- 스팸 메일 필터링을 예제로 하되, BOW인코딩을 할 때, 각 키워드가 출현한 빈도를 '입력 변수'로 사용
- 정상메일 4개, 스팸메일 6개

```py
### 예제 데이터
X = np.array([
[3, 4, 1, 2],
[3, 5, 1, 1],
[3, 3, 0, 4],
[3, 4, 1, 2],
[1, 2, 1, 4],
[0, 0, 5, 3],
[1, 2, 4, 1],
[1, 1, 4, 2],
[0, 1, 2, 5],
[2, 1, 2, 3]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

## 모델 생성
from sklearn.naive_bayes import MultinomialNB
model_mult = MultinomialNB().fit(X, y)

## 모델의 속성
model_mult.classes_
model_mult.class_count_
np.exp(model_mult.class_log_prior_)

## 각 클래스(0,1) 별 특정 면(1, 2, 3, 4)이 나온 횟수[사면체 주사위라 생각] 
fc = model_mult.feature_count_
""" 
> array([[12., 16., 3., 9.],
        [ 5., 7., 18., 18.]])
"""

## 모수 추정치 (스무딩 적용)
np.exp(model_mult.feature_log_prob_)
""" 
> array([[0.29545455, 0.38636364, 0.09090909, 0.22727273],
        [0.11538462, 0.15384615, 0.36538462, 0.36538462]])
"""

## 테스트 1) 각 키워드가 10번씩 나왔을 경우

x_test = np.array([10, 10, 10, 10])
model_mult.predict_proba([x_test])
# > array([[0.38848858, 0.61151142]])
## 스펨메일일 확률이 높다.

```

## 6. 뉴스그룹 분류
뉴스그룹 데이터에 대해 나이브베이즈 분류모형 적용

### Pipeline의 기능
보통 데이터 분류 모델의 생성과정은 아래와 같은 일련의 과정을 거친다.
1. 데이터 분석을 하기 위해서는 Preprocessor(전처리기)와 Model(분류모형)이 존재해야 한다.
2. Train데이터의 경우 Preprocessor로 fit()하는 과정과 transform()을 해서 Model에 집어넣는 과정을 거친다. 그리고 그 Model이 fit()하는 과정을 거쳐야 한다.
3. Test데이터의 경우 fit()을 하지 않고 transform()과정만 거쳐서 Model에 집어넣어지고, 그 값을 통해 predict()한다.

그런데 `Pipeline([ ])`을 사용하면, 위의 과정을 자동적으로 해준다. 따라서 우리는 주어진 모델에서 fit()과 predict()만 하면 된다.
- `pipeline`내의 인수명은 임의로 줄 수 있다.

```py
## 데이터 로드
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset="all")
X = news.data
y = news.target

## 전처리기와 모델을 합치기(Pipeline)
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer

model1 = Pipeline([ # CountVectorizer 전처리기 사용
    ('vect', CountVectorizer()),
    ('model', MultinomialNB()),
])

model2 = Pipeline([ # TfidVectorizer 전처리기 사용
    ('vect', TfidfVectorizer()),
    ('model', MultinomialNB()),
])
model3 = Pipeline([ # 불용어 'english' 제거
    ('vect', TfidfVectorizer(stop_words="english")),
    ('model', MultinomialNB()),
])
model4 = Pipeline([ # 정규화를 통해 쓸데없는 문장들 추가 제거
    ('vect', TfidfVectorizer(stop_words="english",
                             token_pattern=r"\b[a-z0-9_\-\.]+[a-z][a-z0-9_\-\.]+\b")),
    ('model', MultinomialNB()),
])

## 각 모델에 대해 KFold 방법으로 측정 훅 cross_val_score비교
%%time
from sklearn.model_selection import cross_val_score, KFold
for i, model in enumerate([model1, model2, model3, model4]):
  scores = cross_val_score(model, X, y, cv=5)
  print(("Model{0:d}: Mean score: {1:.3f}").format(i + 1, np.mean(scores)))
"""
> 출력값
Model1: Mean score: 0.855
Model2: Mean score: 0.856
Model3: Mean score: 0.883
Model4: Mean score: 0.888
CPU times: user 5min 52s, sys: 8.16 s, total: 6min
Wall time: 6min 5s
"""
### 전처리 작업을 할 수록 성능이 좋아짐을 확인할 수 있다.
```