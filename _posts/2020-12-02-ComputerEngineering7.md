---
title:  "컴퓨터 공학(입문) - 7. 파일 입출력"
excerpt: "파일 입출력과 함수 / 실습 예제"
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

# 파일 입출력과 함수 (1강)

## 일상 생활 문제: 성적처리(파일 입출력 확장)
- 파일에 점수를 모두 저장하여 소스코드에서 입력 받고 평균과 학점을 계산하여 파일로 출력하는 문제
- 파일로 점수를 저장, 파일로 입력 받기, 파일로 출력하는 기능이 필요하다.

![](/assets/images/ComputerEngineering/CE7_1.PNG)

## 파일의 개념
- 파일(File)은 데이터의 모임으로써 보조기억장치에 저장된 것
- 텍스트(Text) 파일은 사람이 알아볼 수 있는 문자나 숫자 등으로 이루어진 파일

### 텍스트 파일과 소스코드 파일
- 텍스트 파일 작성: 메모장에서 확장자 `.txt`로 저장
- 소스코드 작성: 확장자 `.C`로 저장

- 소스코드에서 파일을 입력 받을 경우, **소스코드와 입력 파일을 같은 폴더**에 넣고 저장
    - 다른 코드에 있을 경우 소스코드에서 경로를 지정해줘야 한다.

## 파일 입출력 처리 순서

1. 파일 연결 (input.txt, output.txt)
    - 파일의 주소를 저장할 수 있는 **파일 포인터 변수** 선언
    - `FILE* inData, outData` //FILE이 데이터 타입이 된다.
2. 파일 열기
    - `fopen()` 함수 사용
3. 파일의 데이터 읽어 오기
    - `fscanf()` 함수 사용
4. 읽어온 데이터로 성적 처리
    - if, else 등의 명령어 사용
5. 파일 닫기
    - `fclose()` 함수 사용

## 문제 해결방법의 설계

- 분할 정복 (divide and conquer) 알고리즘
- 함수호출 방법(값에 의한 호출, 주소에 의한 호출)

![](/assets/images/ComputerEngineering/CE7_2.PNG)

- 구조도

# 파일 입출력과 함수 예제 실습 (2강)

## 구조도
![](/assets/images/ComputerEngineering/CE7_3.PNG)

![](/assets/images/ComputerEngineering/CE7_4.PNG)

- 주소를 넘기고, scanf로 입력을 받아서 진행한다.

## 코드

- 성적처리 `main()`

1. 변수 선언
2. 파일열기
3. 반복하기
4. 파일 닫기

```c
int main(void)
{ //변수선언
  FILE* spStu; // 입력파일의 주소를 저장할 포인터 변수
  FILE* spGrades; // 출력파일의 주소를 저장할 포인터 변수

  int stuID, exam1, exam2, final //학번, 과제1, 과제2, 기말
  int avrg;   // 평균
  int grade;  // 학점
  printf("Begin student grading \n");

  // 파일열기
  if (!(spStu = fopen("input.txt", "r"))) //읽기모드 열기 (열렸다면 true리턴)
  {
    printf("Error opening student file \n"); // 에러 처리
    return 100; // 프로그램이 종료가 됨
  }
  /** fopen시 폴더 내에 파일이 없으면, 생성하게 된다 **/
  if (!(spGrades = fopen("output.txt", "w"))) //쓰기모드 열기
  {
    printf("Error opening grades file \n"); // 에러 처리
    return 102;
  }

  // 입력 데이터가 여러 줄이므로 반복해서 입력, 계산, 출력 함수 호출
  while(getStu(spStu, &stuID, &exam1, &exam2, &final)) //정상적으로 데이터가 들어오면 True리턴
  {
    //점수는 변하면 안되므로 CallByValue, 점수와 평점은 CallByAddress로 
    calcGrade(exam1, exam2, final, &avrg, &grade); 
    writeStu(spGrades, stuID, avrg, grade);
  } //while

  fclose(spStu); //연결된 파일 포인터 변수 닫기
  fclose(spGrades);
  printf("End student grading\n");
  return 0;
} // main
```

- 데이터 입력 `getStu()`

1. 매개변수 주소 전달받음
2. 변수 선언
3. 파일로부터 입력 받기
4. 파일 입력 에러 확인

```c
//main에서 선언된 변수의 주소가 매개변수로 넘어온다.
int getStu(FILE* spStu, int* stuID, int* exam1, int* exam2, int* final)
{
  int ioResult; //입력데이터 에러여부 확인용 변수

  //데이터를 파일로부터 입력받아 main 영역 변수에 저장 (fscanf)
  ioResult = fscanf(spStu, "%d%d%d%d", stuID, exam1,exam2, final); // 파일 포인터변수가 필요
  //fscanf는입력 받는 데이터의 개수 리턴
   
  if(ioResult == EOF) //End of File: 파일의 끝 확인
    return 0;
  else if(ioResult != 4) // 데이터가 4개가 아니면 에러처리
    {
      printf("Error reading data \n");
      return 0; // 파일 데이터가 끝났거나 개수가 다른 경우 0값 리턴
    }
  else
    return 1; // 정상 입력 시 1값을 리턴
}
```

- 성적 계산 `calcGrade()`

1. 매개변수 전달받음 (exam1,2 final 변수값, avrg, grade 주소값)
2. 평균 계산 (역참조연산자 사용)
3. 학점 계산

```c
void calcGrade(int exam1, int exam2, int final, int* avrg, char* grade)
{ // 역참조 연산자 사용해 main함수의 avrg변수 값 직접 바꿈 // 입력 시에 주소에 해당하는 값을 리턴함
  *avrg = (exam1 + exam2 + final) / 3;
  if(*avrg >= 90) // 평균값을 사용해 학점 계산
    *grade = "A"; // 역참조 사용해 main 함수의 grade값 직접 바꿈
  else if(*avrg >= 80)
    *grade = "B";
  else if(*avrg >= 70)
    *grade = "C";
  else if(*avrg >= 60)
    *grade = "D";
  else 
    *grade = "F";
}
```

- 성적 출력 `writeStu()`
1. 매개변수 전달받음
2. 파일로 출력

```c
// 점수를 입력 받아 평균과 학점을 계산하는 함수
// 실행 전: 출력파일 포인터, 학번, 평균, 학점 매개변수 입력
// 실행 후: 학번, 평균, 학점을 output.txt파일로 직접 출력

void writeStu(FILE* spGrades, int stuID, int avrg, char grade)
    //output.txt 파일로 실행 결과 직접 출력, 모니터에는 나오지 않음
    fprintf(spGrades, "%04d" %d %c\n", stuID, avrg, grade) //%04d - 4개의 칸에 학번 출력
    //spGrade: output.txt의 주소를 저장
```

