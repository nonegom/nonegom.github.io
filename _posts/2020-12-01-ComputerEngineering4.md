---
title:  "컴퓨터 공학(입문) - 4. 연산자"
excerpt: "연산자(사칙, 관계, 논리 연산자) 개념, 실습 예제"
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

# 연산자(사칙, 관계, 논리 연산자) 개념 (1강)

## 연산자의 개념
- 연산자(operator)
연산자는 산술 연산자 +, -, * 기호와 같이 이미 정의된 연산을 수행하는 키워드를 의미한다.

- 피연산자
연산(operation)에 참여하는 변수나 값을 의미한다. `10 + 20`에서 '10'과 '20'은 피연산자, '+'은 연산자이다.

![](/assets/images/ComputerEngineering/CE4_1.PNG)

## 대입 연산자 (assignment operator)
- 변수의 저장 값을 대입하는 `=`기호가 대입(할당) 연산자이다.
- =연산자 오른쪽 수식을 먼저 계산하고 결과 값을 왼쪽 변수에 대입하는 기능 (왼쪽 부분에는 반드시 변수만 가능)
- 수식의 결과 값을 변수에 대입하지 않으면 프로그램에는 영향이 없다.

## 관계 연산자 (relational operator)
- 관계 연산자는 2개의 피연산자 관계(크기)를 비교하기 위한 연산자 (`>, <, ==, !=, >=, <=`)
- 관계 연산자가 포함된 수식의 결과는 1(참)이 아니면 0(거짓)
- `printf()`를 사용하여 출력할 경우 결과가 참이면 정수 1, 거짓이면 0을 출력

## 논리 연산자 (logical operator)
- 논리 연산자는 두 개 또는 하나의 논리값을 0이나 1의 논리값으로 평가하기 위한 연산자
- AND(`&&`), OR(`||`), NOT(`!`)

### 생각하기
- C언어는 참과 거짓이라는 상수는 없으며, 0, 0.0, '\0'는 거짓으로 0이 아닌 모든 값을 참으로 간주한다.
- `(3 %& 4)` -> 1 (True)

## 연산자 우선순위(★)

![](!assets/images/ComputerEngineering/CE4_2.PNG)

# 연산자 실습 (2강)

## 사칙연산 예제 
- 2개의 값을 입력받아 `+, -, *, /, %`연산 후 출력

```c
#include <stdio.h>
int main(void)
{
    int x, y, add, sub, mul, div, mode;
    printf("정수 1 입력하세요: ");
    scanf("%d", &x);
    printf("정수 2 입력하세요: ");
    scanf("%d", &y);

    add = x + y;
    sub = x - y;
    mul = x * y;
    div = x / y;
    mod = x % y;

    printf("%d + %d = %d 입니다 \n", x, y, add);
    printf("%d - %d = %d 입니다 \n", x, y, sub);
    printf("%d * %d = %d 입니다 \n", x, y, mul);
    printf("%d / %d = %d 입니다 \n", x, y, div);
    printf("%d %% %d = %d 입니다 \n", x, y, mod); // '%'를 프린트문 안에서 입력하고 싶을 때는 %%식으로 표현한다. 
    return 0;
}

```

## 관계연산 예제

- 문자와 정수를 입력 받아 관계 연산 결과 출력

```c
#include <stdio.h>
int main(void)
{
    int x, y;
    char c;

    printf("문자1개 입력하세요: ");
    scanf("%c", &c);
    printf("입력한 문자 %c의 아스키 10진수는 $d 입니다 \n", c, c); // %c는 문자를, %d는 숫자를 출력
    printf("입력한 문자 %c의 다음 문자는 %c입니다. \n", c, c+1);   // 문자라도 숫자처럼 연산 가능
    printf("(%c < %c)는 %d입니다.\n", c, c+1, (c < c+1>));

    printf("정수 2개 입력하세요: ");
    scanf("%d%d", &x, &y);
    printf("(%d >= %d)는 %d입니다 \n", x, y, (x >= y));
    printf("(%d == %d)는 %d입니다 \n", x, y, (x == y));
    printf("(%d != %d)는 %d입니다 \n", x, y, (x != y));
    return 0;
}
```
> 문자를 숫자로 출력할 수도 있다.

## 논리연산 예제

- 문자와 정수를 입력 받아 관계 연산 결과 출력

```c
#include <stdio.h>
int main(void)
{
    printf("(3>2) && (1==2)의 결과는 %d 입니다. \n", (3>2) && (1==2)); // 0
    printf("(3>2) || (1==2)의 결과는 %d 입니다. \n", (3>2) || (1==2)); // 1
    printf("(5>2) != (1==2)의 결과는 %d 입니다. \n", (5>2) != (1==2)); // 1
    printf("('A' < 'B') || 0.0의 결과는 %d 입니다. \n", (5>2) != (1==2)); // 1
    return 0;
}
```