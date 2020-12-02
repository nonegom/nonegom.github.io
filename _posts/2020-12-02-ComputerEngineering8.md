---
title:  "컴퓨터 공학(입문) - 8. 배열과 구조체"
excerpt: "배열과 함수 / 배열과 구조체"
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

# 배열과 함수 (1강)

## 배열(array)
- 배열의 필요성: 많은 자료를 처리해야 할때 여러 개를 한번에 선언하고 각각의 데이터에 접근하여 처리할 수 있는 방법의 필요

- 배열
동일한 자료형의 데이터가 여러 개 연속적으로 저장되어 있는 데이터 저장 장소
`int score[5];` (자료형 / 배열이름 / [배열크기])

### 배열의 초기화
1. 초기화
`int score[5] = {1, 2, 3, 4, 5};`

2. 배열의 크기 없이 초기화
`int score[] = {1, 2, 3, 4, 5};`
자동적으로 초기값의 원소 개수 만큼 배열 크기 생성

3. 일부만 초기화
`int score[5] = {90, 80};` //나머지 원소들은 0으로 초기화

4. 0으로 초기화
`int score[5] = {0}` //모든 원소에 0 삽입

### 변수선언과 메모리

```c
// 주소를 10진수로 출력(%d), 16진수로 출력(%p)
int score1 = 89
printf("score1: %d, &score1: %d, 16&score1: %p", score1, &score1, &score1)
```

![](/assets/images/ComputerEngineering/CE8_1.PNG)

- 배열명(배열 시작 주소)은 주소 상수이므로 변경이 불가능하다.
- 원소명: 첨자는 0부터 시작한다.

```c
a == &a[0] //배열 시작 주소
a+1 == &a[1] a[1]원소의 주소
//주소에 +1은 그 다음 칸 주소를 의미한다.

*(a+2) = 70;
//a+2주소에 역참조연산자를 이용해서 원소값을 변경할 수 있다. 
```

## 배열 예제
1. 원소 값 출력
2. 원소 주소값 출력
3. 원소 합 출력

```c
int main(void)
{
  int score[5] = {10, 20, 30, 40, 50};
  int i, n, sum=0;

  n = sizeof(score) / sizeof(int); // 배열 원소의 수 (전체 바이트 수 / 배열타입 바이트 수 )

  printf("\n ** score 배열 ** \n");
  for(i = 0; i < n; ++i){
    printf("score[%d] : %d \n", i, score[i]); //배열원소 값
  }

  printf("\n ** score 배열 주소 ** \n");
  for(i = 0; i < n; ++i){
    printf("&score[%d] : %d \n", i, &score[i]); //배열원소 주소값
  }
  for(i = 0; i < n; ++i)
  {
    sum += score[i];
  }
  printf("\n 배열합: %4d\n", sum);
  return 0;
}
```

## 배열과 함수
- 함수를 이용한 배열의 합 구하기 문제
x배열과 y배열을 선언하고, 해당 배열의 각각 원소를 더해 다른 배열에 결과를 저장하고 출력하는 프로그램을 작성하는데, 단, `add_arrays`함수를 만들어서 처리하시오.

- CallByAddress를 사용해야 한다.
```c
#include<stdio.h>
int main(void)
{
  int x[5] = {10, 20, 30, 40, 50};
  int y[5] = {4, 55, 33, 28, 35};
  int xysum[5] = {0};
  int i = 0; n = 5;
  
  printf("\n x배열원소 출력: ");
  for(i = 0; i < n; ++i){
    printf("%3d", x[i]);}

  printf("\n y배열원소 출력: ");
  for(i = 0; i < n; ++i){
    printf("%3d", y[i]);}

  add_arrays(x, y, xysum, n); //함수 호출

  printf("\n\n x + y 결과 출력: ");
  for(i = 0; i < n; ++i){
    printf("%3d", xysum[1]);
  }
  return 0;
}

void add_arrays(const int a[], const int b[], int absum[], int n)
{
  int 1;
  for (i = 0; i < n; ++i){
    // main의 xysmu[i]값을 직접 바꾼다.
    absum[i] = a[i] + b[i];
  }
}
```
- 매개변수에서 배열형식([])으로 받아줄 경우, 포인터 `*`를 쓰지 않아도 ([]) 연산자를 사용해 원본의 배열을 직접 접근하여 읽거나 쓸 수 있는 기능을 제공한다. 따라서 원본 배열의 값이 쉽게 바뀔 수 있으므로 `const` 사용해서 제어를 권장한다.
- 위 코드를 보면 `add_arrays`에서 값이 변하면 안되는 배열 a[], b[에는 `const`를 붙였고, 값을 변화시키는 배열 absum[]에는 `const`를 붙이지 않았다.

# 배열과 구조체 (2강)

## 사용자 정의 자료형(User-defined data structure)
- 기본 자료형: 프로그래밍 언어에서 기본적으로 제공하는 자료형 (int, float, double, char 등)
- 사용자 정의 자료형: 해결하려는 문제와 가장 가까운 자료구조를 사용자가 직접 자료형으로 만들어서 문제를 해결할 수 있는 자료형
    - 구조체: struct

## 구조체(structure)

- **구조체의 필요성**
동일한 자료형의 데이터가 여러 개 필요한 경우에 배열을 사용할 수 있지만, 서로 다른 자료형을 가진 데이터를 함께 저장하고 처리하기 위해 새로운 자료형이 필요했음

- **구조체 정의**
다양한 자료형의 연관된 데이터를 묶어서 선언할 수 있도록 **사용자 정의 자료형**을 만드는 것 
탬플릿(templet)과 같은 역할을 하며, 구조체 정의는 **메모리에 변수를 생성하지 않는다**.

```c
struct stu{ //구조체 자료형 이름
  int ID;
  float kor, eng, math;
  float avg;
  char grade;
}
```

- **구조체 변수**
구조체 정의 후, 구조체 자료형을 사용해 변수를 선언한다. 구조체 변수를 선언하면 구조체 멤버의 크기 만큼 메모리에 할당

```c
struct stu{ //구조체 자료형 이름
  int ID;
  float kor, eng, math;
  float avg;
  char grade;
}
struct stu s1 //구조체 변수 선언

/********************************/
typedef struct stu stu; //타입이름 변경

stu s1 // 구조체 변수 선언
```

- **구조체 변수 선언**

```c
// 위에서 구조체 선언됐다고 가정
struct stu s1 = {1001, 99.5, 88.7, 77.9}; 
struct stu s2; //구조체 변수 선언

s2.ID = 10002; //. 연산자를 사용해 구조체 멤버에 접근 가능
s2.kor = 90.5;
s2.eng = 80.3;
s2.math = 95.4;
```

> 다른 프로그래밍 언어의 '객체 지향'의 전신이 '구조체'라고 할 수 있다. 

- **구조체 배열 선언**

```c
// 위에서 구조체 선언됐다고 가정

struct stu s[3]; 
s[0].ID = 10001;
s[0].kor = 90.5;
s[0].eng = 80.3;
s[0].math = 95.4;
```

- **구조체 배열 출력**

```c
struct stu s[3] ={
  {1001 85 98 89},
  {1002 77 00 88},
  {1003 97 79 88}
};
for (i=0; i<3; i++){
  printf("%d %5.2f %5.2f %5.2f %5.2f \n", s[i].ID, s[i].kor, s[i].eng, s[i].math);
}
```

## 구조체 배열 예제

1. 구조체 정의
2. 구조체 변수(배열) 선언
3. 일반 변수 선언
4. 학번, 점수 입력
5. 입력된 점수 출력
6. 평균 계산
7. 과목 총점 계산
8. 학점 계산
9. 학번, 평균, 학점 출력
10. 과목별 평균 출력

```c
#include <stdio.h>
#define MAX 3 

//이후에 나오는 MAX는 3으로 처리하겠다는 뜻

//1. 구조체 정의
struct stu{
  int ID;
  float kor, eng, math;
  float avg;
  char grade;
}

int main(void){
  //2. 구조체 변수(배열) 선언
  struct stu s[MAX];

  //3. 변수 선언
  int i, j, korsum=0, engsum=0, mathsum=0;

  //4. 학번, 점수 입력
  printf("학번, 점수(국, 영, 수)를 입력하세요: \n");
  for(i=0;i< MAX;i++){
    scanf("%d %f %f %f", &s[i].ID, &s[i].kor, &s[i].eng, &s[i].math);
  }

  //5. 입력된 점수 출력
  printf("\n 입력된 점수 \n")
  for (i=0; i<3; i++){
  printf("%d %5.2f %5.2f %5.2f %5.2f \n", s[i].ID, s[i].kor, s[i].eng, s[i].math);
  }
  //6. 평균계산
  for (i=0; i< MAX; i++){
    s[i].avg = (s[i].kor + s[i].eng + s[i].math)/3.0;
    //7. 과목 총점계산
    korsum+= s[i].kor;
    engsum+= s[i].eng;
    mathsum+= s[i].math;

  //8. 학점계산
  for(i=0; i< MAX; i++){
    if(s[i].avg >= 90)
      s[i].grade = "A";
    else if(s[i].avg >= 80)
      s[i].grade = "B";
    else if(s[i].avg >= 70)
      s[i].grade = "C";
    else if(s[i].avg >= 60)
      s[i].grade = "D";
    else 
      s[i].grade = "F";
  }
  //9. 학번, 평균, 학점 출력
  printf("\n 성적 출력 \n");
  for(i=0; i< MAX; i++){
    printf("학번: %5d\t평균:%5.2f\t 학점:%c\n", s[i].ID, s[i].avg, s[i].grade); 
    // \t는 tab누른 길이만큼 띄운다.
    // %5.2f 처럼 형식을 지정하지 않으면 소수점 6째 자리까지 나옴
  }
  //10. 과목별 평균 출력
  printf("\n 과목별 평균 \n");
  printf("국어: %5.2f\t 영어: %5.2f\t 수학: %5.2f\n", korsum/3.0, engsum/3.0, mathsum/3.0);
  }
  return 0;
}
```