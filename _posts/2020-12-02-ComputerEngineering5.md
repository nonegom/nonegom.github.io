---
title:  "컴퓨터 공학(입문) - 5. 조건문"
excerpt: "조건문과 문제해결 예제"
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

# 조건문과 문제해결 예제 (1강)

## 일상생활문재 (성적 처리)
학생들의 점수를 입력 받아 점수에 따라 성적을 산출하는 일을 컴퓨터를 활용해 해결하고자 한다.

## 조건문 (if)
- 조건의 결과(참 또는 거짓)에 따라 프로그램의 흐름을 제어하는 문장
- 어떠한 조건을 만족하면 그에 해당하는 일이 처리되는 문장

```c
if (expression)
    statement1;
next_statementl
```

## 조건문 (if-else)
- if에서 결과가 거짓인 경우 수행해야 할 문장이 있다면 키워드 else 사용

```c
if (expression)
    statement1;
else
    next_statementl;
```

### if-else 예제: 3개의 값을 입력받아 작은 값(min)을 출력

```c
#include <stdio.h>
int main(void)
{
  int x, y, z, min;
  printf("Input three integers: ");
  scanf("%d%d%d", &x, &y, &z);
  if(x < y){
    min = x;
  }
  else{
    min = y;
  }
  if(z < min) //else-if로 사용할 수 있다.
  {
    min = z;
  }
  printf("The minimum value is %d \n", min)l;
  return 0;
}
```

## else-if 예제
- 조건 여러 개를 비교해 조건에 맞는 문장을 수행
- 중첩된 if문에서 else이후에 if문을 실행하는 구문

```c
if (expression)
{
    statement1;
}
else if (expression2)
{
    statement2;
}
else
{
    statement3;
}
```

> `if-else`를 이용해 '성적에 따른 학점'을 배정할 수 있다. 

# 조건문과 문제해결 예제 (2강)

## 일상생활 문제 (윤년 계산)
2월의 마지막 날이 어떤 해는 28일, 어떤 해는 29일인 경우가 있는데 년과 월을 입력했을 때 해당하는 월의 말일을 알고 싶다면?

### 문제분석하기
- 윤년이란 무엇인가?
- 윤년 계산 방법, 알고리즘 조사하기

## switch
선택해야 할 조건이 여러 개 있을 경우 조건에 맞는 문장을 선택하여 수행 

expression, value: 정수 또는 정수 수식이어야 한다.

- switch case: 필수 항목
- break, default: 선택 항목

- break문을 만나면 switch문을 종료한다. 만약 break문이 없다면 조건에 맞는 case를 다 실행한다.

```c
switch(expression)
{
  case value1:
    statement1;
    break;
  case value2:
    statement2;
    break;
  default: 
    statement;
    break;
}
next_statement;
```

## switch 예제(윤년 계산)

- switch문 활용(2월을 제외한 달)

```c
switch(month)
{
  case1:
  case3:
  case5:
  case7:
  case8:
  case10:
  case12:
    maxDay = 31;
    break;

  case4:
  case6:
  case9:
  case11:
    maxDay = 30;
    break;

  return 0;
}
```

### 윤년 계산 활용
- 기원연수가 4로 나누어 떨어지는 해는 윤년 (`year % 4 ==0`)
- 그 중에서 100으로 나누어 떨어지는 해는 평년 (`year % 100 !=0`)
- 다만 400으로 나누어 떨어지는 해는 다시 윤년으로 정하였다. (`year % 400 ==0`)

```c
((year % 4 == 0) && (year % 100 !=0) || (year % 400 == 0))
```

---
### 최종 코드

```c
int main(void) {
int year=0, month =0, maxDay =0;
printf("\n* 년과 월을 입력하세요: ");
scanf("%d %d", &year, &month);

switch(month){
  case1:
  case3:
  case5:
  case7:
  case8:
  case10:
  case12:
    maxDay = 31;
    break;

  case4:
  case6:
  case9:
  case11:
    maxDay = 30;
    break;

  case2:
    if((year % 4 == 0) && (year % 100 !=0) || (year % 400 == 0))
      {maxDay = 29;
      prinf("%d년 %d월의 말일은 %d일(윤년)입니다.", year, month, maxDay);
      break;
      }
    else:
    { maxDay = 28;
      prinf("%d년 %d월의 말일은 %d일입니다.", year, month, maxDay);
      break;
    }
  default:
    printf("입력이 잘못 되었습니다! \n");
  }
  return 0;
}
```