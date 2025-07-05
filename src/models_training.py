import sys
import io
import os
import csv
# Change default encodings
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
from sklearn.utils import resample

import numpy as np
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt

import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler # Data normalization
from scipy.special import inv_boxcox

from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
import warnings

from joblib import dump

# ---------------------------
# 1. Used functions
# ---------------------------
# Path to Functions folder
function_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../Functions'))
sys.path.append(function_path)
from functions import *

# Danh sách các trạm
stations = [
    {
        "station_id": 1,
        "Tentram": "LaiChau" #,
        # "filename": data_LaiChau_path,
        # "updated_file": LaiChau_updated_path,
        # "daily_input_file": LaiChau_daily_input_path
    },
    {
        "station_id": 2,
        "Tentram": "BanChat" #,
        # "filename": data_BanChat_path,
        # "updated_file": BanChat_updated_path,
        # "daily_input_file": BanChat_daily_input_path
    }
    # Bạn có thể thêm các trạm khác tại đây
]

# Danh sách các mô hình
models = [
    {
        "model_id": 1,
        "Tenmohinh": "LGBM"
    },
    {
        "model_id": 2,
        "Tenmohinh": "RF"
    },
    {
        "model_id": 3,
        "Tenmohinh": "LSTM"
    }
    # Bạn có thể thêm các model khác tại đây
]

# ---------------------------
# 2. Input data from FlowObservations table of streamflow.db Database
# ---------------------------
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/streamflow.db'))
conn = get_db_connection(db_path)
# Doc danh sach cac tram
stations_df = pd.read_sql("SELECT station_id, station_name FROM Stations", conn)
station_dict = pd.Series(stations_df.station_id.values, index=stations_df.station_name).to_dict()
# print(stations_df)

# for _, station in stations_df.iterrows():
for _, station in stations_df.head(-1).iterrows(): # Tam thoi bo qua tram Nam Giang vi chua co so lieu realtime
    # print(station["station_id"])
    df = get_station_data(conn,station["station_id"])
    # print(df.head())
    # last_row_df = [df.index[-1].date()] + df.iloc[-1].tolist()
    #print(last_row_df)
    data = np.array(df['Flow']) # data la du lieu goc
    # ---------------------------
    # 3. Data preprocessing
    # ---------------------------
    ## 3.1. Data transformations: 
    # Box-Cox transformation: --> df_BoxCox
    df_copy = df.copy()
    df_BoxCox = df_copy.copy()
    df_BoxCox['Flow'], param_1 = stats.boxcox(df_copy['Flow'])
    # print('Optimal lamda1:', param_1)
    np.random.seed(seed=1500)

    ## 3.2. Handling outliers
    outlier_idxs_Q = detect_outliers(df_BoxCox["Flow"])
    #print("Outlier values: ", df_BoxCox["value"][outlier_idxs_Q])
    #data_box_cox_outliers = df_BoxCox.loc[outlier_idxs_Q, ['date', 'Flow']]

    # Replace Outliers by margin values
    Q1 = df_BoxCox['Flow'].quantile(0.25)
    Q3 = df_BoxCox['Flow'].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR

    # Remove outliers from the series
    df_BoxCox_removeOutliers = df_BoxCox.copy()

    # Thay cac outliers bang cac bien tren va bien duoi tuong ung
    df_BoxCox_removeOutliers['Flow'] = df_BoxCox['Flow'].clip(lower=lower_bound,upper=upper_bound) 
    # Interpolate outlier values using linear interpolation
    # df_BoxCox_removeOutliers['Flow'].loc[outlier_idxs_Q] = df_BoxCox_removeOutliers['Flow'].loc[outlier_idxs_Q].interpolate(method='linear')

    # # Display the new dataframe in which outliers were replaced by margin values
    # df_copy_remove_outliers.to_csv('LGBM_LC_1day_data_frame_copy_remove_outliers.csv')

    ## 3.3. Data normalization: Min-max scale
    df_BoxCox_removeOutliers_scaled=df_BoxCox_removeOutliers.copy()
    scaler = MinMaxScaler()
    df_BoxCox_removeOutliers_scaled['Flow']=scaler.fit_transform(df_BoxCox_removeOutliers[['Flow']])

    # 3.4. Denoising data: Methods: WT, FFT, TSR_WT, TSR_FFT, DAE
    df_BoxCox_removeOutliers_scaled_denoised=df_BoxCox_removeOutliers_scaled.copy()
    # Wavelt Transform Denoise: denoised_method = WT
    denoised_method = 'WT'
    if denoised_method=='WT':
        df_BoxCox_removeOutliers_scaled_denoised['Flow']=wavelet_denoise(df_BoxCox_removeOutliers_scaled['Flow'],wavelet='db4', level=3)
        # print(df_BoxCox_removeOutliers_scaled_denoised)
    if denoised_method=='FFT':
        pass
    if denoised_method=='TSR_WT':
        pass
    if denoised_method=='TSR_FFT':
        pass
    if denoised_method=='DAE':
        pass
    
    # 3.5. Creat lag features
    if station["station_id"] == 1:
        lags_lst = [1,2,3]
    elif station['station_id'] == 2:
        lags_lst = [1,2,3,4]
    elif station['station_id'] == 3:
        lags_lst = [1,2,3,5]
    for lag in lags_lst:
        df_BoxCox_removeOutliers_scaled_denoised[f'lag_{lag}']=df_BoxCox_removeOutliers_scaled_denoised['Flow'].shift(lag)
    #print(df_BoxCox_removeOutliers_scaled_denoised.columns)

    # 3.6. Thêm features thời gian (phân loại) và encode dữ liệu này
    df_BoxCox_removeOutliers_scaled_denoised['month']=df_BoxCox_removeOutliers_scaled_denoised.index.month
    #print(df_BoxCox_removeOutliers_scaled_denoised['month'])
    # One-hot encoding cho tháng
    df_BoxCox_removeOutliers_scaled_denoised=pd.get_dummies(df_BoxCox_removeOutliers_scaled_denoised,columns=['month'],prefix='month')
    df_BoxCox_removeOutliers_scaled_denoised_copy = df_BoxCox_removeOutliers_scaled_denoised.copy()    
    # print(df_BoxCox_removeOutliers_scaled_denoised.columns)
    
    # 3.7. Tạo nhãn dự báo multi-output (n_steps_ahead bước tiếp theo)
    n_steps_ahead = 10 # Số bước cần dự báo trước
    for i in range(1,n_steps_ahead+1):
        df_BoxCox_removeOutliers_scaled_denoised_copy[f'Flow_t+{i}']=df_BoxCox_removeOutliers_scaled_denoised_copy['Flow'].shift(-i)

    # Xóa NaN do shift()
    df_BoxCox_removeOutliers_scaled_denoised_copy.dropna(inplace=True)
    
    # 4. Xây dựng mô hình
    # 4.1. Chia tập huấn luyện và kiểm tra
    #split_index = int(0.8 * len(df_BoxCox_removeOutliers_scaled_denoised))
    split_index = len(df_BoxCox_removeOutliers_scaled_denoised_copy) - max(lags_lst)

    train = df_BoxCox_removeOutliers_scaled_denoised_copy[:split_index]
    #print("train length: ",len(train))
    test = df_BoxCox_removeOutliers_scaled_denoised[-1:]
    #print("test length: ",len(test))

    # Dùng 1 trong 2 cách 1 hoặc 2 sau:
    # 1.
    X_train = train.drop(columns= [f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)])
    y_train = train[[f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)]]
    y_train.to_csv('LC_y_train.csv')

    # X_test = test.drop(columns= [f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)])
    X_test = test
    # y_test = test[[f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)]]
    # y_test.to_csv('LC_y_test.csv')

    # 2.
    # X_train = train.drop(columns=[f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)])
    # y_train = train[['Flow'] + [f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)]]
    # y_train.to_csv('LC_y_train.csv')

    # X_test = test.drop(columns=[f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)])
    # y_test = test[['Flow'] + [f'Flow_t+{i}' for i in range(1, n_steps_ahead + 1)]]
    # y_test.to_csv('LC_y_test.csv')
    
    # Train mo hinh LGBM cho cac tram
    # 4.2. Hyperparameters tuning --> Read hyperparameters from file, vd: hyperparams_LGBM_LaiChau.xlsx
    if station["station_id"] == 1:
        params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/hyperparameters/hyperparams_LGBM_LaiChau.xlsx'))
    elif station['station_id'] == 2:
        params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/hyperparameters/hyperparams_LGBM_BanChat.xlsx'))
    elif station['station_id'] == 3:
        params_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/hyperparameters/hyperparams_LGBM_NamGiang.xlsx'))
        
    df_params = pd.read_excel(params_path,header=None, skiprows=1, parse_dates=[0], names=['Parameter', 'Value'])
    # print(df_params)

    # Change df_params to dict of best_params for trainning model
    best_params = {}
    for i in range(len(df_params)):
        best_params[df_params['Parameter'][i]]=df_params['Value'][i]
    # print(best_params['num_leaves'])
    
    ## 4.3. Huấn luyện mô hình cuối cùng với hyperparameters tốt nhất
    model_wavelet = MultiOutputRegressor(lgb.LGBMRegressor(num_leaves=int(best_params['num_leaves']),
                                                            max_depth=int(best_params['max_depth']),
                                                            learning_rate=best_params['learning_rate'],
                                                            n_estimators=int(best_params['n_estimators']),
                                                            random_state=42))
    model_wavelet.fit(X_train, y_train)
    
    # Luu mo hinh da huan luyen de sau nay dung cho du bao
    if station["station_id"] == 1:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/trained_models/model_LGBM_LaiChau_daily.joblib'))
    elif station['station_id'] == 2:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/trained_models/model_LGBM_BanChat_daily.joblib'))
    elif station['station_id'] == 3:
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models/trained_models/model_LGBM_NamGiang_daily.joblib'))
    dump(model_wavelet,model_path)    
