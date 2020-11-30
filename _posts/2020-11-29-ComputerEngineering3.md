---
title:  "컴퓨터 공학(입문) - 3. 함수"
excerpt: "사용자 정의 함수"
categories:
  - ComputerEngineering
tags:
  - 11월
og_image: "/assets/images/green.jpg"
toc: true
toc_sticky: true
toc_label: 페이지 목차
use_math: true
---

# 사용자 정의 함수 (1강)

## 함수의 개념
- 함수: 독립적으로 수행하는 프로그램 단위
C언어는 여러 개의 함수들로 이루어진다. 프로그램에서 반복적으로 수행되는 기능을 함수로 만들어 호출한다.

- 주어진 문제를 작은 문제, 여러 함수로 나누어 생각할 수 있으므로 함수를 만드는 것은 문제 해결의 하나의 방법

- 함수 이용의 장점
    - 함수로 구성된 프로그램은 함수 단위로 구성되어 있어 읽기 쉽고, 이해하기 쉽다.
    - 이미 정의된 함수는 여러 번 호출이 가능하므로 소스의 중복 최소화 가능

- 함수 원형(function prototype, 함수 선언)
    - 함수를 사용(호출)하기 이전에 함수의 머리(헤더) 부분을 기술하는 단계계

> 함수(function, 모듈, 메소드(JAVA, C++))

## C프로그램 함수의 종류
1) 주(main) 함수: 프로그램의 시작과 종료를 나타내는 함수로, 프로그램에 main()함수는 꼭 있어야 한다. **사용자 정의 함수라고 할 수 있다.**
2) 사용자 정의 함수: 사용자(프로그래머)가 문제를 분석해 필요한 기능으로 분류해 기능별로 코딩하고자 할 때 만드는 함수 (ex. add(), swap())
3) 시스템 라이브러리 함수: 많이 사용하는 기능의 함수들을 시스템에서 미리 만들어 놓고 사용자가 사용할 수 있도록 제공하는 함수(ex. 입출력함수 [scanf(), printf()])

## 함수의 정의(Function definition)
함수 만들기 예제: 두 개의 정수를 매개변수로 입력받아 더한 값을 리턴 하는 함수 만들기
![](/assets/images/ComputerEngineering/CE3.PNG)

## 사용자 정의 함수 만들기
- C에서 변수는 **데이터 저장 공간**
변수 선언시 해당 변수에 해당하는 저장 공간이 생긴다. 그리고 해당 변수에는 변수명 말고, 숫자로 된 메모리의 address가 할당된다. 
address는 컴파일 할 때마다 값이 달라진다. 

- `\n`: newline 삽입 키보드
- `&` : 주소 연산자 (scanf사용시 변수에 주소연산자를 달아줘야 한다.)



# 함수와 매개변수 (2강)

## 매개변수 없는 함수 만들기
- 매개변수가 없는 함수는 print만 하고 돌아오거나, 간단한 설명을 출력하고자 할 때 사용할 수 있다.

```c
void hello(void) //함수원형

void main()
{
    hello;  // 함수 호출
}

//함수 정의
void hello()
{
    printf("Hello C!");
    return;  // 생략 가능 (void이기 떄문에)
}
```

## 배개변수 있는 함수 만들기
- 함수 안에서 선언된 변수는 함수 안에서만 쓸 수 있다. (지역변수 규칙)
- return의 경우 1개의 값만 리턴이 가능하다.

> return은 main함수로 돌아갈 때 가져가는 값이다.

## 용어정리
- 매개변수(Parameter, 인자(argument)): 함수와 함수 사이에 주고 받는 값
- 변수(Variable): 프로그램에서 데이터를 저장하는 공간
- 지역 변수: 함수 지역 안에서 선언된 변수

![](/assets/images/ComputerEngineering/CE3_1.PNG)


# 함수 호출 방법 (3강)

## 일상문제 해결(성적 처리를 위한 주요 함수 만들기)
- 성적 처리 특성 분석
- 총점 구하기(add())
- 총점을 반영하여 성적순으로 정렬
- 정렬을 위해 두 변수의 값을 서로 바꾸는 함수 필요(swap())

### swap함수 만들기
- 두 변수(a, b)의 값을 바꾸기 위해 (값에 의한 호출)
- temp이라는 임시 변수에 값을 저장시켜놓고, 값을 바꾸면 된다.

> 정렬 알고리즘에서 사용된다.

```c
int a, b, temp;
temp = a;
a = b;
b = temp;
```

### 함수를 통해 두 개의 변수 값을 서로 바꾸는 방법
- 문제 해결하기

main 안에서 변수 값을 temp 변수를 사용해 서로 변경하면 바뀌지만, 다른 함수로 두 변수 값을 call by value로 보내서 바꾼다면 특성상 main 함수 영역에 있는 변수 값을 두 개 모두 바꿀 수 없다. 왜냐하면 return은 1개의 값만 가져올 수 없기 때문이다.

> 이 해결방법은 **주소에 의한 호출(call by address)**이다.

## 주소에 의한 호출(call by address)
- main 함수에 있는 변수의 주소값을 가지고 `swap()`함수를 호출하면 `swap()`함수에서 main 함수의 변수 값을 바꿀 수 있다.(**포인터 변수**)
- 바꾸기 위해서는 역참조 연산자를 사용해야 한다.

### `*`의 두가지 용법
1) 포인터 변수 선언: 주소를 값으로 가지는 변수, 주소값을 저장하기 위해서는 포인터 변수가 필요하다.
2) 역참조 연산자: 포인터 변수 선언 후에 문장 중에 포인터 변수 앞에 `*`가 오면 (`*p`) 포인터 변수가 가리키는 main 변수의 값을 변경할 수 있다.

```c
// 포인터 변수 선언
int *p, int *q

// 포인터 변수는 주소값만 값으로 가질 수 있다.
p = &a
```

![](/assets/images/ComputerEngineering/CE3_2.PNG)

## 함수 호출 예제 sum(), swap()

```c
#include <stdio.h>

int sum(int a, int b); // 함수원형
void swap(int *p, int *q) //함수원형

/*********************************/
int main(void)
{
    int a, b, total;
    printf("Input two integers :");
    scanf("%d%d", &a, &b);

    total = sum(a,b); //call by value
    printf("**sum function call** \n");
    printf("%d + %d = %d \n", a, b, total);

    swap(&a, &b);   //call by address
    printf("**swap function call** \n");
    printf("a:%d, b:%d = %d \n", a, b);
    return 0;
}
/*********************************/
int sum(int a, int b)
{
    int total;
    total = a + b;
    return total;
}

void swap(int *p, int *q)
{
    int temp;
    temp - *p;
    *p = *q;
    *q = temp;
}
```
