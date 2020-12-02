---
title:  "컴퓨터 공학(입문) - 6. 반복문"
excerpt: "제어문과 반복문(for) / 반복문 (while & do-while)"
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

# 제어문과 반복문 for (1강)

## 일상생활문제: 성적처리(확장)
- 여러 학생들의 점수를 입력받아 학점을 처리하는 문제
- 같은 기능을 반복 수행하도록 컴퓨터에게 반복시킬 수 있는 명령어(for, while)

## 제어문 - 프로그램의 흐름 제어
- **순차**(Sequence): 위에서 아래로 한 문장씩 순차적으로 수행
- **선택**(Selection): 조건에 따라 흐름 제어 // **조건문**
- **반복**(Repltition): 조건에 따라 반복 수행 // **반복문**
- **분기**(Jump): 정해진 위치로 이동 // **분기문**

### 제어문 종류
- 조건문: if, if-else, switch-case
- 반복문: for, while, do-while
- 분기문: goto, return, break, continue

### 키워드
- 약속된 의미의 단어 (데이터 타입, 제어 명령 등)
- 식별자(변수명)로 사용할 수 없고, 약속된 의미로만 사용된다.

## 반복문 종류

![](/assets/images/ComputerEngineering/CE6_1.PNG)

## 반복문 (for)

- 조건의 결과에 따라 특정 부분의 처리를 반복 실행하는 제어 문장

```c
for (초기화; 조건검사; 증감연산)
{
  statement1; //참
}
next_statement; //거짓
```

### for 예제
- 1부터 777까지의 합 구하기

```c
#include <stdio.h>
int main(void)
{
  int i, sum;
  sum = 0; // 지역변수 초기화
  for( i= 1; i <=777 ; i++){
    sum += i; //sum = sum + 1
  }
  printf(" 1부터 777까지의 합: %d \n", sum);
  return 0;
}
```

## 일상생활문제: 알고리즘 설계
1. 수강인원 입력받는다.
2. 반복기능: 수강 인원수만큼 반복
  - 학생의 학번과 점수
  - 점수를 총점에 더함
  - 학생의 점수를 학점으로 계산
  - 학번과 계산된 학점 출력
3. 평균 산출 후 출력

```c
int main(void){
int i, stuNum, stuID, score;  //변수 선언
char grade; //학점
float total = 0; //총점
printf("수강인원 입력")
sacnf("%d", &stuNum);

for(i=0; i< stuNum; i++){
  printf("학번과 점수 순서대로 입력: ");
  scanf("%d%d", &stuID, score);
  total += score;
  if(score >= 90)
    grade = 'A';
  else if(score >= 80)
    grade = 'B';
  else if(score >= 70)
    grade = 'C';
  else if(score >= 60)
    grade = 'D';
  else
    grade = 'F';
  printf("학번: %d, 학점: %c \n", stuID, grade);
}
printf("과목평균: %5.2f\n", total/stuNum) //소수점을 두 자리로 표현 (입출력 함수)
return 0;
}
```

# 반복문 while과 do-while (2강)

## 반복문 (while, do-while)

- 조건을 만족하는동안 특정 작업을 반복하여 처리함.
- while문의 경우 조것이 거짓인 경우 statement를 한번도 하지 않지만, do-while의 경우 반드시 한 번은 statement를 수행한다. (가장 큰 차이점)

```c
while(expression)
{
  statement1; //참
  증감연산;
}
next_statement; // 거짓

/**********************/

do {
  statement1;
} while(expression)

```

### while 예제
- 1부터 888까지의 합 구하기

```c
#include <stdio.h>
int main(void)
{
  int i=1, sum=0;
  while(i<=888){
    sum +=i;
    i++;
  }
  printf("1부터 888까지의 합: %d \n", sum);
  return 0;
}
```

### d0-while 예제
- 1부터 999까지의 합 구하기

```c
#include <stdio.h>
int main(void)
{
  int i=1, sum=0;
  do{
    sum +=i;
    i++;
  }while(i<=999);
  printf("1부터 999까지의 합: %d \n", sum);

return 0;
}
```