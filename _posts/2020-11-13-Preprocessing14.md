---
title:  "[머신러닝] 데이터 전처리 (지리정보)  - 2. Geocoding"
excerpt: "주소를 좌표로 바꾸거나, 좌표를 주소로 바꾸는 기능을 하는 `Geocoding`"

categories:
  - ML_PreProcessing
tags:
  - PreProcessing
  - 11월
toc: true
toc_sticky: true
toc_label: 페이지 목차
use_math: true
---

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.04.02%20%EC%A7%80%EC%98%A4%EC%BD%94%EB%94%A9.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

# Geocoding
Geopandas가 제공하는 유용한 기능인 `geocoding`는 **주소를 좌표로 바꾸거나, 좌표를 주소로 바꾸는 기능**을 한다.  

여기에서는 geocoding을 위해서 google의 map API를 사용한다. 이 때, API를 사용하기 위해서 **접근 키**가 필요한데 이는 실습을 원한다면 **본인이 직접 만들어야 한다**.  
만드는 방법은 이 페이지를 참고한다.(https://developers.googleblog.com/2016/03/introducing-google-api-console.html) (이 페이지에 나와 있는 Google Maps Android API 대신에, Geocoding API를 사용하도록 한다.)

> 간단하게 개념과 사용방법에 대해서 알아보도록 하자.

## 연습문제

다음 예제는 서울열린데이터광장에서 제공하는 “서울시 행정구역 읍면동 위치정보” 데이터를 이용해 **서울의 지도를 시각화**하고, “서울시 전통시장 현황” 데이터에 있는 **전통시장의 주소를 좌표로 바꾸어 지도 위에 점**으로 나타내는 것이다.

- 데이터 다운로드 링크
    - [“서울시 행정구역 읍면동 위치정보”](https://data.seoul.go.kr/)

    - [“서울시 전통시장 현황”](http://data.seoul.go.kr/dataList/datasetView.do?infId=OA-1176&srvType=F&serviceKind=1&currentPageNo=1) (위 링크의 “File” → “contents.xlsx”을 다운로드 받는다.)


> 데이터를 받았다는 가정 하에 진행한다.

- 데이터 불러오기

```py
import geopandas as gpd

# 서울시 지도 데이터 시각화
seoul_file = "/current_directory/TL_SCCO_EMD_2015_W_SHP/TL_SCCO_EMD_2015_W.shp"
seoul = gpd.read_file(seoul_file, encoding='euckr')
seoul.tail()
'''
    EMD_CD   EMD_KOR_NM EMD_ENG_NM    ESRI_PK  SHAPE_AREA SHAPE_LEN geometry
462| 11170133 서빙고동  Seobinggo-dong   461    0.000081   0.038065 POLYGON ((126.991846532 37.52515417799998, 126...
463| 11170134 주성동    Juseong-dong     462    0.000018   0.021908 POLYGON ((126.99820106 37.52447825600001, 126....
464| 11170122 문배동    Munbae-dong      442    0.000012   0.018005 POLYGON ((126.970954824 37.53771650800002, 126...
465| 11170103 용산동4가 Yongsan-dong4-ga  447    0.000081  0.036409 POLYGON ((126.988095507 37.53433249699998, 126...
466| 11170112 원효로1가 Wonhyoro 1(iI)-ga 449    0.000021  0.020099 POLYGON ((126.970521024 37.5415352, 126.970578...
'''

# 전통 시장 주소 데이터
market_file = "/current_directory/Book/contents.xlsx"
market = pd.read_excel(market_file)
'''
[열 정보]

콘텐츠 ID (필수 입력)/ 사용유무 (필수입력)/ 콘텐츠명 (필수 입력)/ 서브카테고리 명 (선택 입력)/
시군 (선택입력)/ 구명 (선택입력)/ 새주소[도로명 주소] (조건부 선택 입력)/ 지번주소
'''
```

- 데이터 전처리

```py
# 한글 주소 생성
address = "서울 특별시 " + market["지번주소"].sample(50)
address = list(address)
address[:3]
## > ['서울 특별시 동작구 상도동 324-1', '서울 특별시 영등포구 신길동 255-9', '서울 특별시 동작구 사당동 318-8']

# geocode를 사용해서 geometry와 address(영어) 불러오기
## api_key는 받아야 한다.
location = gpd.tools.geocode(address, provider='googlev3', api_key="your_api_key")
location.tail()
''' 
   geometry                       address
45 POINT (127.0110531 37.5704381) 436-41 Changsin-dong, Jongno-gu, Seoul, South ...
46 POINT (127.0021911 37.4992379) 163 Banpo-dong, Seocho-gu, Seoul, South Korea
47 POINT (127.0806369 37.6167463) 157-34 Muk-dong, Jungnang-gu, Seoul, South Korea
48 POINT (127.0229047 37.630752)   54-5 Suyu-dong, Gangbuk-gu, Seoul, South Korea
49 POINT (126.8911206 37.5228224) 30 Yangpyeongdong 1(il)-ga, Yeongdeungpo-gu, S...
'''
```

- 데이터 시각화

```py
ax = seoul.plot(figsize=(11, 11), color="w", edgecolor="k")
ax.set_title("서울시 내 전통시장 들의 위치")
location.plot(ax=ax, color='r')
ax.set_axis_off()
plt.show()
```

![](/assets/images/Preprocessing14_1.png)