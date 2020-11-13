---
title:  "[머신러닝] 데이터 전처리 (지리정보)  - 1. 지리 정보 데이터 처리"
excerpt: "지리 정보 데이터 처리를 위한 기초(GeoPandas) - Geometry데이터 및 관계연산과 속성연산, 좌표계"

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

> 아래 포스팅은 기존 수업의 복습 차원에서 올리는 포스팅입니다. 따라서 세부적인 수학적 정리가 생략된 부분이 있습니다. 따라서 좀 더 구체적인 정보나 원하시면 [데이터 사이언스 스쿨 사이트](https://datascienceschool.net/03%20machine%20learning/03.04.01%20%EC%A7%80%EB%A6%AC%20%EC%A0%95%EB%B3%B4%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%B2%98%EB%A6%AC.html)를 참고 부탁드립니다. 특히 아래 코드를 이용한 시각화 그래프 코드와 모습을 보고 싶으시면 링크 확인부탁드립니다.  

# 지리 정보 데이터 처리

지리정보데이터, GIS(Geospatial Information System) 라고 말하는 것은 위치에 대한 정보를 광범위하게 포함하는 말로 **좌표, 주소, 도시, 우편번호 등**을 포함한다.

# 1. GeoPandas

GeoPandas는 파이썬에서 **지리정보 데이터 처리의 기하하적 연산과 시각화 등**을 돕는 패키지이다. 이름으로도 알 수 있듯이, GeoPandas는 Pandas와 비슷하다. 두 가지의 자료형 `GeoSeries`와 `GeoDataFrame`이 있다. 다루는 방법에 큰 차이가 없다. 다만 지리정보 데이터 분석에 유용한 속성과 메서드가 존재한다. 다음처럼 `gpd`라는 이름으로 임포트 하는 것이 관례이다. 

```py
import geopandas as gpd
```

GeoPandas는 간단한 지리정보데이터를 실습 할 수 있는 샘플 데이터 셋을 제공한다. 이 데이터를 사용해 GeoPandas의 기초적인 기능을 학습할 수 있다. 이 실습 데이터는 `gpd.dataset.get_path()` 명령으로 데이터의 링크를 불러와 사용 한다.

```py
countries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
countries.tail(3)
```

![](\assets\images\Preprocessing13_1.JPG){: .align-center}


> 지리 정보 위치가 Polygon으로 들어가 있다.

```py
cities = gpd.read_file(gpd.datasets.get_path('naturalearth_cities'))
cities.tail()
```

![](\assets\images\Preprocessing13_2.JPG){: .align-center}

> Polygon뿐만 아니라 도시의 경우 "점(Point)"으로 나타나 있다.


## 1) 지리 정보의 시각화
GeoSeries와 GeoDataFrame 객체의 `plot()`명령을 사용하면, GeoPandas 내부의 Geometry 데이터를 손쉽게 시각화 할 수 있다. 

Geometry 데이터는 지리정보를 표현하는 **다각형, 선, 점**을 의미하는데, GeoPandas는 내부적으로 다각형, 선, 점을 `Shapely` 패키지를 사용하여 처리한다. 각각 **다각형-Polygon, 선-LineString, 점-Point**로 정의 되어 있다. 

GeoPandas가 제공하는 데이터에는 Geometry 데이터가 이미 포함되어 있지만, 우리가 가진 데이터를 활용해 생성할 수도 있다.

### 코드 사용법
**- 카테고리 데이터** 
대부분의 경우 지리정보를 시각화 할 때는 **위치에 따른 정보의 변화를 함께 표현**한다. 이 때는 `plot()`명령의 `column` 인자에 반영하고 싶은 **데이터의 열 이름을 입력**하면 해당 열의 데이터를 색(color)으로 표현한다. 표현하려는 정보가 **카테고리 데이터** 일때는 `categorical`인자를 True로 설정한다.

```py
ax = countries.plot(column="continent", legend=True, categorical=True)
ax.set_title("세계 지도")
ax.set_axis_off()
plt.show()
```

![](/assets/images/Preprocessing13_3.png)

**- 실수형 데이터**
만약, 표현하고 싶은 컬럼이 **실수 변수**라면, 데이터를 구분하는 방법과 개수를 정의할 수 있다. 구분 방법은 `plot()`명령의 `scheme` 인자로 설정한다. 지원하는 것으로는 `“Equal_interval”`(동일한 간격으로 구분), `“Quantiles”`(4분위수를 구하여 구분), `“Fisher_Jenks”`(클래스 내 분산을 줄이고, 클래스 끼리의 분산을 최대화하는 방식으로 구분)가 있다. 구분하는 개수는 **k 인자에 원하는 숫자를 입력**하면된다. 디폴트는 5이다.

아래 코드는 국가별 GDP 추정치를 해당 국가의 추정인구로 나누어, 추정 1인당 GDP를 만들고, 이를 지도에서 색으로 표현한 예이다.

![](/assets/images/Preprocessing13_4.png)

## 2) Geometry 데이터
GeoPandas에서는 `Shapely`라는 패키지를 통해 **Geometry 데이터를 처리**한다. 여기서는 Geometry 데이터에 대해서 알아보도록 하자.

### Polygons
**한 국가의 영토** 따위 등은 여러 개의 점을 이은 **다각형**으로 나타낼 수 있다. “Countries” 데이터에서는 다음처럼 Polygon 데이터를 제공한다.

```py
countries.geom_type[:3]
'''
0         Polygon
1    MultiPolygon
2         Polygon
dtype: object
'''

## polygon 데이터는 여러 개의 점(point)들로 이루어져 있다.
```

### Points
어떤 사건이 발생한 **위치**, 한 **국가의 수도**, 두 **국가간의 중앙점** 등은 하나의 **좌표**로 나타낼 수 있다. “Cities” 데이터에서는 도시를 하나의 **점**으로 나타내었다. 이 점을 다음처럼 지도위에 표현 할 수 있다.

```py
cities.geom_type[:3]
'''
0    Point
1    Point
2    Point
dtype: object
'''
```

```py
base = countries.plot(color='white', edgecolor="k")
ax = cities.plot(ax=base, marker='o', color='red', markersize=5)
ax.set_axis_off()
ax.set_title("세계 도시 분포")
plt.show()
```

![](/assets/images/Preprocessing13_5.png)

### LineString
점과 점을 이은 것은 선이 된다. 두 도시 사이의 **길**, **강의 흐름**, **국경**의 생김새, **경계면 정**보 등을 선으로 나타낼 수 있겠다.  
다음 코드는 우리나라의 육지를 선으로 나타낸 것이다. 여기서 사용되는 `squeeze()`함수는 GeoPandas 객체에서 **Geometry 데이터 만을 추출해주는 기능**을 한다. `boundary` 속성에 대해서는 다음 단락에서 학습 하겠다.

```py
korea_border = countries[countries.name == "Korea"].geometry
# 버전 6.x에서는 "South Korea"를 사용해야 함

korea_border.boundary.squeeze()
```

![](/assets/images/Preprocessing13_6.png){: width="200" height="200"}{: .align-center}

### Geometry 데이터의 속성

Geometry 타입의 데이터는 다음과 같은 속성을 가지고 있다.

- 지리정보의 속성 ( 아래의 속성값은 `Point` 데이터에서는 모두 0이다. )
 
    - `area` : 넓이
    - `boundary` : 테두리
    - `centroid` : 중앙지점

그리고 두 Geometry 간의 **거리를 계산 해주는 함수** 또한 유용하게 사용된다.
 
  - `distance` : 두 점 사이의 거리

넓이, 거리는 우리가 흔히 사용하는 제곱미터, 마일, 킬로미터 등의 단위를 사용하는 것이 아니다. 그래서 **같은 객체 안에서의 비교만 가능**하다.

## 3) GeoPandas의 지리 데이터 간의 관계 연산

GeoPandas는 지리데이터 간의 **관계를 연산해주는 기능**을 가지고 있다. 관계를 연산한다는 말은, 두 **데이터가 교차**하는지, **하나가 다른 하나의 내부**에 있는지 등을 말한다.

좌표 데이터를 `Shapely`의 `Geometry` 자료형으로 만들면, 불러온 데이터셋을 이용해, **어떤 도시가 어느 나라 안**에 있는지, **도시끼리의 거리**는 얼마나 되는지, **어떤 도시가 두 도시 사이에 존재**하는지 등의 관계를 알 수 있다.

### 관계 연산 예제

지리적 관계에 대한 경우의 수는 매우 많기 때문에, 간단한 예제를 통해 기본적인 관계연산을 알아본다.

먼저, 동북아시아의 주요 국가(한국, 중국, 일본)와 도시(서울, 베이징, 도쿄)를 각각 **선과 점**으로 나타내고, **도시들을 이어 주었다(`line`).**

```py
from shapely.geometry import Point, Polygon, LineString

northern_asia = countries.loc[countries['name'].isin(['Korea', 'China', 'Japan'])]
base = northern_asia.plot(figsize=(15, 15), color="w", edgecolor="m")

# 도시 geometry 정보
seoul = cities.loc[cities.name == "Seoul", "geometry"].squeeze()
beijing = cities.loc[cities.name == "Beijing", "geometry"].squeeze()
tokyo = cities.loc[cities.name == "Tokyo", "geometry"].squeeze()

# 라인 정의
line = LineString([beijing, seoul, tokyo])

ax = gpd.GeoSeries([seoul, beijing, tokyo, line]).plot(ax=base, color="k", edgecolor='k', lw=1)
ax.set_title("동북아시아 지도")
ax.set_axis_off()
plt.show()
```

![](/assets/images/Preprocessing13_7.png)

#### 기본적인 관계 연산 함수
관계연산의 출력값은 boolean 값이다.

- 기본 관계 연산
    - `within` : 지리적으로 **포함되는지** 여부
    - `contains` : 지리적으로 **포함하고 있는지** 여부
  
    - `intersects` : 지리적으로 **교차하는지** 여부, 두 지리가 경계선만 닿아있어도, True를 반환
    - `crosses` : 지리적으로 교차하는지 여부, intersects와 차이점은 crosses는 **내부를 지나가야만 True**를 반환한다는 것이다.

```py
## 나라의 geometry 정보를 squeeze() 함수로 빼옴
korea = countries.loc[countries['name'] == 'Korea', 'geometry'].squeeze()
china = countries.loc[countries['name'] == 'China', 'geometry'].squeeze()
japan = countries.loc[countries['name'] == 'Japan', 'geometry'].squeeze()

# 서울은 한국 안에 있다.
seoul.within(korea)
## > True

# 한국은 서울을 포함 하고 있다.
korea.contains(seoul)
## > True

# 중국과 한국의 국경은 맞닿아 있지 않다.
china.intersects(korea)
## > False

# 홍콩, 베이징, 토쿄, 서울을 잇는 선은 한국을 지나 간다.
line.crosses(korea)
## > True
```

위와 같은 간단한 관계연산을 이용하면 **데이터 검색 또한 가능**하다.

```py
countries[countries.crosses(line)]

# line이 지나는 나라를 검색한다.
```

![](/assets/images/Preprocessing13_8.png){: .align-center}

### 속성을 이용한 연산

```py
# 서울에서 베이징까지의 거리
seoul.distance(beijing)
## > 10.871264032732043

#한국의 면적 과 중국 면적의 비율
china.area / korea.area
## > 95.89681432460839

# 한국으로 부터 서울과 베이징 간의 거리 보다 가까운 데이터 검색
countries[countries.geometry.distance(seoul) <= seoul.distance(beijing)]
```

![](/assets/images/Preprocessing13_9.PNG){: .align-center}

```py
## 한국과 중국의 지리적 중심 표시
base = northern_asia[northern_asia.name != "Japan"].plot(
    figsize=(15, 15), color="w", edgecolor="m")
ax = gpd.GeoSeries([china.boundary, korea.boundary, china.centroid,
                    korea.centroid]).plot(ax=base, color="k", edgecolor='k', lw=1)
ax.set_title("중국과 한국의 지리적 중심")
ax.set_axis_off()
plt.show()
```

![](/assets/images/Preprocessing13_10.png)

## 4) 지리정보 조작

다음 나열된 함수들은 **지리정보를 변환하는 기능**을 가진다.

- `convex_hull`
    - Polygon 데이터의 convex hull을 그린다.

- `envelope`
    - Polygon 데이터를 감싸는 가장 작은 사각형을 그린다.

- `simplify(tolerance, preserve_topology=True)`
   - Polygon 데이터에 대해 컨투어 추정을 한다.

- `buffer(distance, resolution=16)`
    - Point, LineString 데이터에 실행하면 주어진 거리 내의 모든 점을 이어 Polygon 데이터를 만들고, Polygon에 적용하면 주어진 거리만큼 확장한다.

- `unary_union`
    - 여러 개의 geometry 데이터의 합집합을 구한다.
    - Polygon 내에 빈 곳이 있는 경우, unary_union가 실행 되지 않는다. 이 때는 buffer() 명령으로 Polygon의 빈 곳을 채워준 후 사용한다.

  다음 그림은 동작구의 기초 구역도를 convex_hull, envelope, unary_union을 이용해 변환한 것이다. (예제 데이터와 코드는 생략한다.)

  ![](/assets/images/Preprocessing13_11.png)

GeoDataFrame 또한 Pandas DataFrame의 `groupby` 명령과 같은 **그룹핑 기능을 제공**한다.

- `Dissolve`
    - GeoDataFrame 내의 geometry를 그룹 별로 `unary_union`를 이용해 geometry 데이터를 합친다.

  ![](/assets/images/Preprocessing13_12.png)


# 2. GeoPandas의 좌표계

**좌표계(CRS)**는 지구를 2차원 데이터(평면)로 표현하는 방법론을 의미한다. GeoPandas 데이터라면, `crs`속성값으로 확인 할 수 있다. 좌표계가 다른 데이터의 경우, 좌표간의 거리나 위치 등이 다르게 표현되기 때문에 반드시 통일 시켜주어야 한다. 주로 사용되는 좌표계는 다음과 같다.

- **WGS84(EPSG:4326)**: **GPS**가 사용하는 좌표계(경도와 위도)

- **Bessel 1841(EPSG:4004)**: 한국과 일본에 잘 맞는 **지역타원체를 사용**한 좌표계

- **GRS80 UTM-K(EPSG:5179)**: 한반도 전체를 하나의 좌표계로 나타낼 때 많이 사용하는 좌표계. **네이버 지도**

- **GRS80 중부원점(EPSG:5181)**: 과거 지리원 좌표계에서 타원체 문제를 수정한 좌표계. **다음 카카오 지도**

- **Web mercator projection(Pseudo-Mercator, EPSG:3857)** : **구글지도**/빙지도/야후지도/OSM 등 에서 사용중인 좌표계

- **Albers projection(EPSG:9822)** : 미국 지질 조사국에서 사용하는 좌표계

GeoPandas는 좌표계 변환 기능을 `to_crs()`라는 함수로 제공한다. 다음 코드는 “epsg:4326” 에서 “epsg:3857”로 변환한다.  
다음 코드의 시각화된 결과를 보면, 남극이 찢어져 있는 것을 볼 수 있다. “epsg:3857”, **Web mercator projection**은 내부적으로 계산이 간단하고 빠르다는 장점이 있지만, 북위 85도, 남위 85도 이상의 지역인 **극지방에 대해서는 정보 왜곡이 심하다**. 

```py

countries.crs
{'init': 'epsg:4326'}

# 좌표 변환
countries_mercator = countries.to_crs(epsg=3857)

ax = countries_mercator.plot(
    figsize=(15, 15), column='continent', cmap="tab20b", legend=True, categorical=True)
ax.set_title("세계 지도")
ax.set_axis_off()
plt.show()
```

![](/assets/images/Preprocessing13_13.png){: width="550" height="550"}{: .align-center}
