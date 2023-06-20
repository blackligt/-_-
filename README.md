## 프로젝트 소개

2023년-1학기 전북대학교 산학실전캡스톤 과정에서 진행한 프로젝트 입니다. <br>
프로젝트 주제는 머신러닝을 이용한 소방력 재배치 입니다. <br><br>


## 소방력 재배치
1. k-means를 사용해 응급데이터의 위치데이터를 클러스터링 하여 새로운 소방서 배치를 제안합니다.
2. xgboost로 지역별 소방위험도를 예측하여 소방위험도를 바탕으로 소방서 별 인력을 재배치 합니다.
<br>

## 개발 환경 설정
```
!pip install pandas
!pip install numpy
!pip install sklearn
!pip install tqdm
!pip install matplotlib
!pip install xgboost 
```

## Files
1. EDA : EDA를 활용하여 데이터를 분석한 파일입니다.
2. code_data : 2011-2020년도의 인구, 노령화지수, 날씨 등의 데이터를 csv로 변환하는 파일입니다.
3. emg_with_firestation : 응급데이터에 집계구코드의 columns을 합치고 전처리하는 파일입니다.
4. firedata : 화재데이터 분석하고 년도별 화재수 집계하는 파일입니다..
5. firestation_pred_kmeans : 응급데이터를 k-means로 군집화 시켜서 소방서를 재배치 하는 파일입니다.
6. Learning : xgboost로 화재,응급 데이터에 대한 모델 학습하는 파일입니다.
<br>
<img width="491" alt="그림1" src="https://github.com/blackligt/Machine_Learning_Fire/assets/107538112/8c3dbeca-c005-4cb1-a9b7-af0dd94defed"> <br>

