# %%
from sklearn.cluster import KMeans
from tqdm import tqdm     

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# %%
<<<<<<< HEAD
df_2= pd.read_csv('./dataSet/df_2_real_last.csv',encoding='EUC-KR')
=======
df_2= pd.read_csv('./df_2_real_last.csv',encoding='EUC-KR')
>>>>>>> 54cddba04422efcb93e81bbe9af9732a1d7078e9
dictionary_2 = pd.read_csv('./dictionary_2.csv', encoding = 'EUC-KR')
# %%
df = df_2.drop(labels = ['Unnamed: 0'], axis = 1)
# %% 위도 측정치 잘못 기입으로 인해 값 변경
df = df.drop('name', axis = 1)
for i in df.index:
    if df.loc[i,'long'] == 126561621.0:
        df.loc[i,'long'] = 126.561621
max(df.loc[:,'long'])
# %%
from sklearn.model_selection import train_test_split
X, test_df= train_test_split(df, test_size = 0.1, random_state = 2222)
train_df, val_df= train_test_split(X, test_size = 0.3, random_state = 2222)
# %% K-means 모델 생성 및 학습
features = ['loc_info_lat', 'loc_info_long']
X = df[features]

kmeans = KMeans(n_clusters=47, random_state=42, init = 'k-means++', max_iter = 300)
kmeans.fit(X)

cluster_labels = kmeans.labels_
#%% # 소방서 위치 추출
fire_station_locations = kmeans.cluster_centers_
#%% 기존 소방서 위치 데이터 갯수 맞춰줌 55111 -> 46
A = np.unique(np.array(df.loc[:,'lat']))
B = np.unique(np.array(df.loc[:,'long']))
#%%

# %%
# 시각화
plt.figure(figsize=(8, 6))

# 클러스터링 결과 시각화
plt.scatter(X['loc_info_lat'], X['loc_info_long'], c=cluster_labels, cmap='viridis', alpha=0.5)

# 소방서 위치 시각화
plt.scatter(np.array(df.loc[:,'lat']),np.array(df.loc[:,'long']), marker = 'p', s=100, color='blue', label='Original Fire Station')
plt.scatter(fire_station_locations[:, 0], fire_station_locations[:, 1], marker='X', s=100, color='red', label='Pred Fire Station')

plt.xlabel('loc_info_lat')
plt.ylabel('loc_info_long')
plt.title('Fire Station Locations')

plt.legend()
plt.show()
#%% 위경도 데이터 -> 직선거리 계산 함수(라디안 단위 변환 후 계산) 
def calculate_distance(lat1, lon1, lat2, lon2):

    # 위도와 경도를 라디안 단위로 변환
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    # 직선 거리 계산
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    radius = 6371  # 지구의 반지름 (단위: km)
    distance = c * radius

    return distance
# %%
loc_info_long, loc_info_lat = df.loc[:,'loc_info_long'], df.loc[:,'loc_info_lat']
#%% 진짜 찐막
distance_dict_2 = {}
count_dict_2 = {}
for i in range(len(kmeans.cluster_centers_)):
    distance_dict_2['distance_'+str(i)] = 0
    count_dict_2['count_'+str(i)] = 0
distance_dict_3 = {}
count_dict_3 = {}
for i in range(len(kmeans.cluster_centers_)):
    distance_dict_3['distance_'+str(i)] = 0
    count_dict_3['count_'+str(i)] = 0
#%%
count =0
length = []
for j in tqdm(range(len(kmeans.labels_))):
    for i in range(len(kmeans.cluster_centers_)):
        if i == kmeans.labels_[j]:
<<<<<<< HEAD
            count_dict_2['count_'+str(i)] += 1              /
=======
            count_dict_2['count_'+str(i)] += 1
>>>>>>> 54cddba04422efcb93e81bbe9af9732a1d7078e9
            cal = calculate_distance(loc_info_lat[i], loc_info_long[i], fire_station_locations[i, 0], fire_station_locations[i, 1])
            distance_dict_2['distance_'+str(i)] += cal
            count_dict_3['count_'+str(i)] += 1
            cal_2 = calculate_distance(loc_info_lat[i], loc_info_long[i], A[i], B[i])
            distance_dict_3['distance_'+str(i)] += cal_2
            if cal > cal_2:
                count +=1
            length.append(cal - cal_2)
#%% 군집화를 통한 생성한 위치와 사고발생 위치 거리가 약 5.6% 감소함
sum(length)/len(length)
# %%
<<<<<<< HEAD
=======
A,B
# %%
kmeans.cluster_centers_
# %%
>>>>>>> 54cddba04422efcb93e81bbe9af9732a1d7078e9
