---
title:  "Python - 입문2"
excerpt: "반복문 / 함수 / 데이터 구조"
categories:
  - ComputerEngineering
tags:
  - 12월
og_image: "/assets/images/green.jpg"
toc: true
toc_sticky: true
toc_label: 페이지 목차
use_math: true
---

> 게시물 내 이미지 자료의 저작권은 SmartLearn에 있습니다.

# 1강) 반복문

## 반복문 for

- for: 리스트나 튜플, 문자열의 첫 번째 요소부터 마지막 요소까지 차례로 변수에 대입되어 문장들이 수행된다.

```py
'''
for 변수 in 리스트(또는 튜플, 문자열):
    문장
'''

for x in range(5): 
    print("Hello Python")

# 5번 반복이 된다.
```

### range() 함수
`range([start,] stop [, step])`
- range() 함수는 지정된 범위의 값을 반환
- range(start, stop)와 같이 호출하면 start부터 시작해서 (stop-1)까지의 정수가 생성
- **stop은 포함되지 않는다.**
- range(10)하면 0부터 9까지의 정수가 생성된다. 

### 리스트
- 데이터의 구조로 데이터들의 목록이다.
-리스트에는 다양한 데이터타입을 넣을 수 있다.

```py
for x in [0, 1, 3, 4, 5]:
    print(x, end=" ") # print문에 end매개변수 삽입 가능

for name in ["A", "B", "C", "D"]:
    print(name)

for string in "가나다라마바":
    print(string)    
```

### while
- 조건의 결과에 따라 특정 부분의 처리를 반복 실행하는 제어문장
- 조건문이 참인 동안 while문 아래에 속하는 문장들이 반복 수행된다.

```py
i = 0
while i < 5:
    print(Hello Python!)
    i += 1
print("반복 종료")
```

## 분기문(Jump Statement)

- break문: 반복을 탈출한다

```py
for i in range(1, 101):
    print("for문을 %d번 실행" % i)
    break
```

- continue문: 무조건 블록의 남은 부분을 건너뛰고 반복문의 처음으로 돌아간다.

```py
# 1~100의 숫자 중 3의 배수를 제외한 합
sum, i = 0, 0
for i in range(1, 101):
    if i % 3 == 0:
        continue
    sum += i
```

## 반복문 실습

- Q1. 1부터 사용자가 입력한 수 n까지 더해서 계산하는 프로그램 작성

```py
sum = 0
num = int(input("어디까지 계산할까요? "))
for i in range(1, num+1):
    sum += i
print("1부터 %d까지의 정수의 합 = %d", % (num, sum))
```

- Q2. 팩토리얼 계산
    - 팩토리얼 n!은 1부터 n까지의 정수를 모두 곱한 것을 의미

```py
fact = 1
num = int(input("정수를 입력하시오: "))

for i in range(1, n+1):
    fact *= i
print(fact)
```

- Q3. 정수안의 각 자리수의 합을 계산하는 프로그램 작성

```py
number = 1234
su =0 
while number > 0:
    digit = number % 10
    sum += digit
    number = number // 10
print(sum)

############
sum = 0
num = (input("정수를 입력하시오: "))

for i in num:
    sum += int(i)
```

- Q4. 숫자 맞추기 게임
    - 컴퓨터가 선택한 숫자를 사용자가 맞추는 게임

```py
import random
tries = 0
number = random.randint(1, 100)

print("1부터 100 사이의 숫자를 맞추시오")

while tries < 10:
    guess = int(input("숫자를 입력하시오: "))
    tries += 1
    if(guess < number):
        print("숫자가 낮음!")
    elif (guess > number): 
        print("숫자가 높음!")
    else:
        break
if guess == number:
    print("정답!! 시도횟수: %d" % tries)

print("정답은 %d" %number)
```

---

# 2강) 함수와 모듈

## 함수 개념
- 함수(function)는 독립적으로 수행하는 프로그램 단위로 특정 작업을 수행하는 명령어들의 모음에 이름을 붙인 것

- 프로그램에서 반복적으로 수행되는 기능을 함수로 만들어 호출
- 함수는 작업에 필요한 데이터(매개변수)를 전달받을 수 있으며, 작업이 완료된 후에는 작업의 결과를 호출자에게 반환할 수 있다.

### 함수의 필요성
- 함수는 문제 해결의 방법
- 함수로 구성된 프로그램은 읽기 쉽고, 이해하기 쉽다. 또한 이미 정의된 함수는 여러 번 호출 가능하므로 소스의 중복을 최소화해 프로그램의 양을 줄이는 효과가 있다.

## 함수 정의
- 함수 정의는 `def`로 시작하고 콜론(`:`)으로 끝낸다.
- 함수의 시작과 끝은 들여쓰기로 구분하고, 시작과 끝 명시해줄 필요가 없다.

```py
def function_name(Argument list ...):
    Statement
    # ...
    return return_value
##########
def add(a, b):
    return a + b
```
- 입력값과 반환 값이 없는 함수, 입력값은 없고 반환 값이 있는 함수도 만들 수 있다.

### 함수 작성 예시
- 정수의 거듭제곱값을 계산하여 **반환하는 함수** 작성(**연산자 사용가능)

```py
def power(x, y):
    result = 1
    for i range(y):
        result = result*x
    return result
print(power(10, 2))

# > 100
```

### 함수 이용시 주의사항
- 파이썬 인터프리터는 함수가 정의되면 함수 안의 문장들은 즉시 실행하지 않음
- 함수 정의가 아닌 문장들은 즉시 실행
- main()함수를 만들어 호출해서 활용할 수 있다.

## 함수 예제 실습
- 성적처리
    - 데이터를 입력하고, 총점 및 평점을 구하고 성적별로 정렬 `swap()`

```py
# 함수
def add(a, b):
    return a+b
def swap(a, b):
    a, b = b, a
    return a, b # 2개 이상의 값을 return할 수 있다.

a, b = 10, 20
print(a, b) # 10 20
a, b = swap(a, b) # 리턴하는 값을 받아줄 변수가 필요하다.
print(a, b) # 20 10
```

---

# 3강) 데이터 구조
## 리스트
- 리스트는 여러 개의 데이터가 저장되어 있는 장소
- 리스트는 여러 개의 데이터를 하나의 이름으로 관리할 수 있는 데이터 구조이다.

- 선언: `리스트이름 = [값1, 값2, 값3]`
- 리스트는 문자열이나 숫자 등을 원소로 가질 수 있다. (문자와 숫자가 섞여도 된다)
- `emptylist = []`처럼 원소를 안 넣고도 만들 수도 있다.
- 리스트의 첫 번호는 '0'번이다. 

- ls[1]처럼 인덱스를 통해 원소에 접근할 수 있다.

### 리스트와 연산자
- `in & not in`: 리스트에 element가 있는지 없는지를 확인하는 연산자

- for문으로 리스트 순회도 가능하다.

### 리스트 사용 가능 함수
![](/assets/images/ComputerEngineering/PB_3.PNG)

### 리스트 예제
- Q. 학생들의 성적을 처리하는 프로그램 작성
    1. 학생들의 성적 입력해서 리스트에 저장
    2. 리스트를 통해 값 확인

```py
STUDENTS = 5
scores = []
scoreSum = 0

for i in range(STUDENTS):
    value = int(input("성적을 입력하시오: "))
    scores.append(value)
    scoreSum += value

scoreAvg = scoreSum / len(cores)
highScoreStudents = 0 
for i in range(len(scores)):
    if scores[i] >= 80:
        highScoreStudents += 1
print("성적 평균: ", scoreaAbg)
print("80점 이상 학생: ", highScoreStudents)
```

- Q.학생들의 이름을 입력받아 처리하는 프로그램

```py
names = []
While True:
    name = input ("학생이름을 입력하세요(입력이 끝나면 공백을 입력하세요): ")
    if name == "":
        break
    names.append(name)
print("학생이름: ")
for n in names:
    print(n)
```

## 데이터 구조
- 프로그램에서 자료들을 저장하는 여러가지 구조들이 있는데, 이를 자료 구조(data structure)라고 한다.
- 파이썬에는 리스트, 튜플, 딕셔너리, 문자열 등 다양한 데이터 구조를 기본으로 사용할 수 있게 제공한다.

### 리스트 예제
- 연락처 관리 프로그램

```py
menu = 0
friends = []
while menu !=9:
    print("-----------------")
    print("1. 친구 리스트 출력")
    print("2. 친구 추가")
    print("3. 친구 삭제")
    print("4. 이름 변경")
    print("9. 종료")

    if menu ==1:
        print(friends)
    if menu ==2:
        name = input("이름을 입력: ")
        friends.append(name)
    if menu ==3:
        del_name = input("삭제하고 싶은 이름 입력: ")
        if del_name in friends:
            friends.remove(del_name)
        else:
            print("이름이 발견되지 않음")
    if menu ==4:
        old_name = input("변경하고 싶은 이름 입력: ")
        if old_name in friends:
            index = friends.index(old_name)
            new_name = input("새로운 이름을 입력: ")
            friends[index] = new_name
        else:
            print("이름이 발견되지 않음")
```

## 튜플(tuple)
- 튜플( `()`)은 변경할 수 없는 리스트로, 변경되면 안 되는 데이터의 경우 튜플형식으로 만든다.
- 값의 수정이나 삭제가 불가능하다.
`t1 = (1, 2, 3, 4)`

## 딕셔너리(dict)
- 딕셔너리는 키(key)와 값(value)의 쌍을 저장할 수 있는 객체

```py
dictionary = {'Kim': '01012345678', 'Moon': '01057392910'}
dictionary['Kim']
# > '01012345678'

dictionary.get('Kim')
# > '01012345678'

## 항목 순회
for item in dictionary.items():
    print(item)

# ('Kim', '01012345678')
# ('Moon', '01057392910')
```

## 문자열(str)
- 문자열은 문자들의 시퀀스(나열)로 정의
- ''(single quote), ""(double quote)모두 사용가능하다.
- 문자끼리의 `+` 연산이 가능하다.

### 슬라이싱
- 문자열이나 리스트를 잘라서 사용할 수 있는 기능이다.
```py
word = 'Pythoon'
word[0:2]
#> 'Py'
## [0:2]의 경우 0, 1까지만 자른다. 슬라이싱은 end-1까지만 출력한다.
word[2:5]
#> 'tho'
```
### in연산자 & not in연산자
- `in`연산자와 `not in`연산자도 사용가능하다.
- 대소문자를 구분한다.
