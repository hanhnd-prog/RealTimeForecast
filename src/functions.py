import os

from math import sqrt
import numpy as np
import pandas as pd
import sqlite3

import matplotlib.pyplot as plt
import pywt
import random
from scipy.stats import randint

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from sklearn.model_selection import KFold

from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
from math import sqrt
from numpy import array
from scipy.stats import randint
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
from sklearn import tree
from lightgbm import LGBMRegressor
from pmdarima import auto_arima # type: ignore
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import resample

import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

# ---------------------------
# 1. Các hàm hỗ trợ
# ---------------------------

def get_db_connection(db_path):
    conn = sqlite3.connect(db_path,check_same_thread=False)
    return conn
def get_station_data(conn, station_id):
    query = f"SELECT record_date, flow_value FROM FlowObservations WHERE station_id = {station_id} ORDER BY record_date"
    #df = pd.read_sql(query, conn, parse_dates=['record_date'])
    #df = df.set_index('record_date')
    
    df = pd.read_sql(query,conn)
    #df['record_date']=pd.to_datetime(df['record_date'],format='ISO8601')
    #df['record_date']=pd.to_datetime(df['record_date'],format='%Y-%m-%d').dt.date
    df['record_date']=pd.to_datetime(df['record_date'],format='%Y-%m-%d')
    df=df.rename(columns={'record_date':'date','flow_value':'Flow'})
    df = df.set_index('date')    
    return df

def theils_u_metric(y_true, y_pred):
    """Calculate Theil's U statistics using observed and predicted vectors."""
    SS_res =  np.mean(np.square(y_true - y_pred))
    SS_true = np.mean(np.square(y_true))
    SS_pred = np.mean(np.square(y_pred))
    SS_NSE = np.mean(np.sum(y_true))
    SS_NSE1 = np.mean(np.sum(y_pred))
    return np.sqrt(SS_res / (SS_true * SS_pred))
def generate_data(n):
    """Tạo dữ liệu chuỗi thời gian (raw data)"""
    x = np.arange(n)
    y = np.sin(0.1 * x) + np.random.normal(0, 0.1, n)
    return y

def detect_outliers(series):
    # Xác định outliers

    # series: 1-D numpy array input
    Q1 = np.quantile(series, 0.25)
    Q3 = np.quantile(series, 0.75)
    IQR = Q3-Q1
    lower_bound = Q1-1.5*IQR
    upper_bound = Q3+1.5*IQR
    lower_compare = series <= lower_bound
    upper_compare = series >= upper_bound
    outlier_idxs = np.where(lower_compare | upper_compare)[0]
    return outlier_idxs

def wavelet_denoise(data, wavelet='db4', level=3):
    """Làm mịn dữ liệu sử dụng phân tích wavelet"""
    coeffs = pywt.wavedec(data, wavelet, level=level)
    # Giữ lại hệ số xấp xỉ, đặt 0 cho các hệ số chi tiết
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]
    data_denoised = pywt.waverec(coeffs, wavelet)
    return data_denoised[:len(data)]

def create_dataset_multi(series, window_size, n_steps_ahead):
    """
    Chuyển chuỗi thành dataset đa bước với cửa sổ lịch sử.
    - X có kích thước (samples, window_size) phù hợp với mô hình học máy.
    - Y có kích thước (samples, n_steps_ahead).
    """
    X, Y = [], []
    for i in range(len(series) - window_size - n_steps_ahead + 1):
        X.append(series[i:i+window_size])
        Y.append(series[i+window_size:i+window_size+n_steps_ahead])
    return np.array(X), np.array(Y)

def create_dataset_multi_2(series, lst_lag,n_steps_ahead):
    X, Y = [], []
    m = max(lst_lag)
    for i in range(m,len(series) - n_steps_ahead + 1):
        indices = [i - x for x in lst_lag]
        tam = series[indices]
        X.append(tam)
        Y.append(series[i:i+n_steps_ahead])
    return np.array(X), np.array(Y)
def compute_mse_per_lead(true, pred, lead_times):
    """Tính MSE cho từng lead time (ví dụ: lead 3 ứng với chỉ số 2, v.v.)"""
    mse_dict = {}
    for lead in lead_times:
        idx = lead - 1
        mse = np.mean((true[:, idx] - pred[:, idx]) ** 2)
        mse_dict[lead] = mse
    return mse_dict

def calMAE_per_lead(true, pred, lead_times):
    MAE_dict={}
    for lead in lead_times:
        idx = lead - 1
        mae = np.mean(np.abs(true[:, idx] - pred[:, idx]))
        MAE_dict[lead] = mae
    return MAE_dict
def calRMSE_per_lead(true, pred,lead_times):
    RMSE_dict = {}
    for lead in lead_times:
        idx = lead - 1
        rmse = sqrt(mean_squared_error(true[:, idx], pred[:, idx]))
        RMSE_dict[lead] = rmse
    return RMSE_dict
def calSIGMA_per_lead(true,pred,lead_times):
    SIGMA_dict = {}
    for lead in lead_times:
        idx = lead - 1
        sigma = np.std(true[:, idx] - pred[:, idx])
        SIGMA_dict[lead] = sigma
    return SIGMA_dict
def calPBIAS_per_lead(true,pred,lead_times):
    PBIAS_dict = {}
    for lead in lead_times:
        idx = lead - 1
        pbias = np.mean(true[:, idx] - pred[:, idx]) - np.mean(true[:, idx])
        PBIAS_dict[lead] = pbias
    return PBIAS_dict
def calKGE_per_lead(true,pred,lead_times):
    KGE_dict = {}
    for lead in lead_times:
        idx=lead - 1
        Tu = np.mean(true[:,idx]*pred[:,idx])-np.mean(true[:,idx])*np.mean(pred[:,idx])
        Mau = sqrt((np.mean(true[:,idx]*true[:,idx])-(np.mean(true[:,idx]))**2)*(np.mean(pred[:,idx]*pred[:,idx])-(np.mean(pred[:,idx]))**2))
        Alpha = Tu/Mau
        Beta = np.std(pred[:,idx])/np.std(true[:,idx])
        Gamma = np.mean(pred[:,idx])/np.mean(true[:,idx])
        kge = 1-sqrt((Alpha-1)**2+(Beta-1)**2+(Gamma-1)**2)        
        KGE_dict[lead] = kge
    return KGE_dict
def calNSE_per_lead(true,pred,lead_times):
    NSE_dict = {}
    for lead in lead_times:
        idx = lead - 1
        Tu = np.sum((true[:,idx] - pred[:,idx])*(true[:,idx]-pred[:,idx]))
        Mau = np.sum((true[:,idx]-np.mean(true[:,idx]))**2)
        NSE = 1 - Tu/Mau
        NSE_dict[lead] = NSE
    return NSE_dict


def random_search_training_lgbm(X_train, Y_train, n_iter=100):
    """
    Random search để tìm hyperparameters tốt nhất cho LightGBM dựa trên validation loss.
    Sử dụng MultiOutputRegressor để mở rộng LightGBM cho bài toán dự báo đa đầu ra.
    """
    best_val_loss = float('inf')
    best_params = None
    # Chia tập huấn luyện thành train con và validation (80%/20%)
    X_train_sub, X_val, Y_train_sub, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    
    for i in range(n_iter):
        num_leaves = random.choice([15, 31, 63])
        max_depth = random.choice([-1, 5, 10, 20])
        learning_rate = random.choice([0.01, 0.05, 0.1, 0.2])
        n_estimators = random.choice([50, 100, 200])
        
        model = MultiOutputRegressor(lgb.LGBMRegressor(num_leaves=num_leaves,
                                                        max_depth=max_depth,
                                                        learning_rate=learning_rate,
                                                        n_estimators=n_estimators,
                                                        random_state=42))
        model.fit(X_train_sub, Y_train_sub)
        Y_val_pred = model.predict(X_val)
        mse = mean_squared_error(Y_val, Y_val_pred)
        print(f"Random search lặp {i+1}: num_leaves={num_leaves}, max_depth={max_depth}, learning_rate={learning_rate}, n_estimators={n_estimators}, val_loss={mse:.4f}")
        if mse < best_val_loss:
            best_val_loss = mse
            best_params = {'num_leaves': num_leaves, 'max_depth': max_depth, 
                           'learning_rate': learning_rate, 'n_estimators': n_estimators}
    return best_params, best_val_loss

def random_search_cv_training_lgbm(X_train, Y_train, n_iter=100, n_splits=5):
    """
    Random search để tìm hyperparameters tốt nhất cho LightGBM dựa trên validation loss.
    Sử dụng MultiOutputRegressor để mở rộng LightGBM cho bài toán dự báo đa đầu ra.
    Áp dụng K-Fold Cross Validation để đánh giá model
    """
    best_val_loss = float('inf')
    best_params = None
    
    kf = KFold(n_splits=n_splits,shuffle=True,random_state=42)
    for i in range(n_iter):
        # Sinh các giá trị hyperparameters ngẫu nhiên
        num_leaves = random.choice([15, 31, 63])
        max_depth = random.choice([-1, 5, 10, 20])
        learning_rate = random.choice([0.01, 0.05, 0.1, 0.2])
        n_estimators = random.choice([50, 100, 200])
        
        fold_losses=[]
        # Thực hiện K-Fold CV
        for train_index, val_index in kf.split(X_train):
            X_train_sub, X_val = X_train[train_index], X_train[val_index]
            Y_train_sub, Y_val = Y_train[train_index], Y_train[val_index]
            model = MultiOutputRegressor(lgb.LGBMRegressor(num_leaves=num_leaves,
                                                        max_depth=max_depth,
                                                        learning_rate=learning_rate,
                                                        n_estimators=n_estimators,
                                                        random_state=42))
            model.fit(X_train_sub, Y_train_sub)
            Y_val_pred = model.predict(X_val)
            mse = mean_squared_error(Y_val, Y_val_pred)
            fold_losses.append(mse)
        avg_loss = np.mean(fold_losses)
        print(f"Random search lặp {i+1}: num_leaves={num_leaves}, max_depth={max_depth}, learning_rate={learning_rate}, n_estimators={n_estimators}, avg_val_loss={avg_loss:.4f}")

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_params = {'num_leaves': num_leaves, 'max_depth': max_depth, 
                           'learning_rate': learning_rate, 'n_estimators': n_estimators}
    
    return best_params, best_val_loss
from scipy.special import inv_boxcox

#####################################################################################3
def bootstrap_LGBM_multi_output(x_train, y_train, X_val, y_val, num_leaves, max_depth, n_estimators, learning_rate, scaler, n_models=100): ### xóa param1
    ###################### xóa param 1###############################
    # Model-based bootstrapping
    model = MultiOutputRegressor(lgb.LGBMRegressor(num_leaves=num_leaves,
                                                   max_depth=max_depth,
                                                   learning_rate=learning_rate,
                                                   n_estimators=n_estimators,
                                                   random_state=42))
    fitted_model = model.fit(x_train, y_train)
    
    # Tính fitted_values và residuals
    fitted_values = fitted_model.predict(x_train)
    residuals = fitted_values - y_train
    
    predictions = []
    obs = []

    for i in range(n_models):
        print(f"Bootstrap iteration: {i+1}/{n_models}")

        # Lấy mẫu bootstrap residuals
        bootstrap_residuals = resample(residuals, replace=True)

        # Dữ liệu bootstrap mới
        y_train_bootstrap = fitted_values + bootstrap_residuals
        
        # Huấn luyện lại model với bootstrap data
        model = MultiOutputRegressor(lgb.LGBMRegressor(num_leaves=num_leaves,
                                                       max_depth=max_depth,
                                                       learning_rate=learning_rate,
                                                       n_estimators=n_estimators,
                                                       random_state=42))
        model.fit(x_train, y_train_bootstrap)

        # Dự báo trên tập validate
        pred_scaled = model.predict(X_val)
                
        # 🔹 Inverse Min-Max Scaler
        pred_original = scaler.inverse_transform(pred_scaled)
        obs_original = scaler.inverse_transform(y_val) #unscaled đổi thành original

        # 🔹 Inverse Box-Cox Transform (Nếu dữ liệu ban đầu đã được Box-Cox)
        # pred_original = inv_boxcox(pred_original, param_1)
        # obs_original = inv_boxcox(obs_original, param_1)  # Đảm bảo obs không bị thay đổi cấu trúc

        # Lưu kết quả
        predictions.append(pred_original)
        obs.append(obs_original)

    predictions = np.array(predictions)
    observations = np.array(obs)

    return predictions, observations

#############################
# RF
def random_search_training_rf(X_train, Y_train, n_iter=10):
    """
    Random search để tìm hyperparameters tốt nhất cho Random Forest (RF).
    Sử dụng MultiOutputRegressor để mở rộng RF cho bài toán dự báo đa đầu ra.
    """
    best_val_loss = float('inf')
    best_params = None

    X_train_sub, X_val, Y_train_sub, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    for i in range(n_iter):
        n_estimators = random.choice([50, 100, 200])
        max_depth = random.choice([5, 10, 20, None])
        min_samples_split = random.choice([2, 5, 10])
        min_samples_leaf = random.choice([1, 2, 4])

        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators,
                                                           max_depth=max_depth,
                                                           min_samples_split=min_samples_split,
                                                           min_samples_leaf=min_samples_leaf,
                                                           random_state=42))
        model.fit(X_train_sub, Y_train_sub)
        Y_val_pred = model.predict(X_val)
        mse = mean_squared_error(Y_val, Y_val_pred)

        print(f"Random search lặp {i+1}: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, val_loss={mse:.4f}")

        if mse < best_val_loss:
            best_val_loss = mse
            best_params = {'n_estimators': n_estimators, 'max_depth': max_depth,
                           'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

    return best_params, best_val_loss

def random_search_cv_training_rf(X_train, Y_train, n_iter=100, n_splits=5):
    """
    Random search để tìm hyperparameters tốt nhất cho Random Forest (RF) dựa trên Cross-Validation.
    """
    best_val_loss = float('inf')
    best_params = None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i in range(n_iter):
        n_estimators = random.choice([50, 100, 200])
        max_depth = random.choice([5, 10, 20, None])
        min_samples_split = random.choice([2, 5, 10])
        min_samples_leaf = random.choice([1, 2, 4])

        fold_losses = []
        for train_index, val_index in kf.split(X_train):
            X_train_sub, X_val = X_train[train_index], X_train[val_index]
            Y_train_sub, Y_val = Y_train[train_index], Y_train[val_index]

            model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators,
                                                               max_depth=max_depth,
                                                               min_samples_split=min_samples_split,
                                                               min_samples_leaf=min_samples_leaf,
                                                               random_state=42))
            model.fit(X_train_sub, Y_train_sub)
            Y_val_pred = model.predict(X_val)
            mse = mean_squared_error(Y_val, Y_val_pred)
            fold_losses.append(mse)

        avg_loss = np.mean(fold_losses)
        print(f"Random search lặp {i+1}: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}, avg_val_loss={avg_loss:.4f}")

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_params = {'n_estimators': n_estimators, 'max_depth': max_depth,
                           'min_samples_split': min_samples_split, 'min_samples_leaf': min_samples_leaf}

    return best_params, best_val_loss   

def bootstrap_rf_multi_output(x_train, y_train, X_val, y_val, n_estimators, max_depth, min_samples_split, min_samples_leaf, scaler, n_models=100):
    """
    Bootstrapping cho Random Forest (RF) với dự báo đa đầu ra.
    """
    model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators,
                                                       max_depth=max_depth,
                                                       min_samples_split=min_samples_split,
                                                       min_samples_leaf=min_samples_leaf,
                                                       random_state=42))
    fitted_model = model.fit(x_train, y_train)

    fitted_values = fitted_model.predict(x_train)
    residuals = fitted_values - y_train

    predictions = []
    obs = []

    for i in range(n_models):
        print(f"Bootstrap iteration: {i+1}/{n_models}")

        bootstrap_residuals = resample(residuals, replace=True)
        y_train_bootstrap = fitted_values + bootstrap_residuals

        model = MultiOutputRegressor(RandomForestRegressor(n_estimators=n_estimators,
                                                           max_depth=max_depth,
                                                           min_samples_split=min_samples_split,
                                                           min_samples_leaf=min_samples_leaf,
                                                           random_state=42))
        model.fit(x_train, y_train_bootstrap)

        pred_scaled = model.predict(X_val)

        pred_unscaled = scaler.inverse_transform(pred_scaled)
        obs_unscaled = scaler.inverse_transform(y_val)

        # pred_original = inv_boxcox(pred_unscaled, param_1)
        # obs_original = inv_boxcox(obs_unscaled, param_1)
        pred_original = pred_unscaled
        obs_original = obs_unscaled

        predictions.append(pred_original)
        obs.append(obs_original)

    predictions = np.array(predictions)
    observations = np.array(obs)

    return predictions, observations 

#####################################SVR#####################
def random_search_training_svr(X_train, Y_train, n_iter=100):
    best_val_loss = float('inf')
    best_params = None

    X_train_sub, X_val, Y_train_sub, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
    
    for i in range(n_iter):
        C = random.choice([0.1, 1, 10, 100])
        epsilon = random.choice([0.01, 0.1, 0.2, 0.5])
        kernel = random.choice(["linear", "rbf", "poly"])
        
        model = MultiOutputRegressor(SVR(C=C, epsilon=epsilon, kernel=kernel))
        model.fit(X_train_sub, Y_train_sub)
        
        Y_val_pred = model.predict(X_val)
        mse = mean_squared_error(Y_val, Y_val_pred)
        
        print(f"Random search lặp {i+1}: C={C}, epsilon={epsilon}, kernel={kernel}, val_loss={mse:.4f}")

        if mse < best_val_loss:
            best_val_loss = mse
            best_params = {'C': C, 'epsilon': epsilon, 'kernel': kernel}
    
    return best_params, best_val_loss

def random_search_cv_training_svr(X_train, Y_train, n_iter=100, n_splits=5):
    best_val_loss = float('inf')
    best_params = None
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    for i in range(n_iter):
        C = random.choice([0.1, 1, 10, 100])
        epsilon = random.choice([0.01, 0.1, 0.2, 0.5])
        kernel = random.choice(["linear", "rbf", "poly"])
        
        fold_losses = []

        for train_index, val_index in kf.split(X_train):
            X_train_sub, X_val = X_train[train_index], X_train[val_index]
            Y_train_sub, Y_val = Y_train[train_index], Y_train[val_index]
            
            model = MultiOutputRegressor(SVR(C=C, epsilon=epsilon, kernel=kernel))
            model.fit(X_train_sub, Y_train_sub)
            Y_val_pred = model.predict(X_val)
            mse = mean_squared_error(Y_val, Y_val_pred)
            fold_losses.append(mse)

        avg_loss = np.mean(fold_losses)
        
        print(f"Random search lặp {i+1}: C={C}, epsilon={epsilon}, kernel={kernel}, avg_val_loss={avg_loss:.4f}")

        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_params = {'C': C, 'epsilon': epsilon, 'kernel': kernel}
    
    return best_params, best_val_loss

def bootstrap_svr_multi_output(x_train, y_train, X_val, y_val, C, epsilon, kernel, scaler, param_1, n_models=100):
    model = MultiOutputRegressor(SVR(C=C, epsilon=epsilon, kernel=kernel))
    fitted_model = model.fit(x_train, y_train)
    
    fitted_values = fitted_model.predict(x_train)
    residuals = fitted_values - y_train

    predictions = []
    obs = []

    for i in range(n_models):
        print(f"Bootstrap iteration: {i+1}/{n_models}")

        bootstrap_residuals = resample(residuals, replace=True)
        y_train_bootstrap = fitted_values + bootstrap_residuals
        
        model = MultiOutputRegressor(SVR(C=C, epsilon=epsilon, kernel=kernel))
        model.fit(x_train, y_train_bootstrap)
        
        pred_scaled = model.predict(X_val)
        pred_unscaled = scaler.inverse_transform(pred_scaled)
        obs_unscaled = scaler.inverse_transform(y_val)

        pred_original = inv_boxcox(pred_unscaled, param_1)
        obs_original = inv_boxcox(obs_unscaled, param_1)

        predictions.append(pred_original)
        obs.append(obs_original)

    predictions = np.array(predictions)
    observations = np.array(obs)

    return predictions, observations

####################LSTM####################
from tensorflow.keras.models import Sequential # type: ignore
from sklearn.multioutput import MultiOutputRegressor
from scikeras.wrappers import KerasRegressor
import random
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization # type: ignore

# Define your model creation function
# def create_model(neurons=50, dropout_rate=0.2, input_shape=None):
#     model = Sequential()
#     model.add(LSTM(units=neurons, activation='relu', input_shape=input_shape))
#     model.add(Dropout(dropout_rate))
#     model.add(Dense(units=1))
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model

def create_model(model_neurons=64, model_dropout_rate=0.2, output_dim=1, input_shape=(10, 1), **kwargs):
    model = Sequential([
        LSTM(model_neurons, activation='relu', return_sequences=False, input_shape=input_shape),
        Dropout(model_dropout_rate),
        Dense(output_dim)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model
# from tensorflow.keras.optimizers import Adam # type: ignore
# def create_model(model_neurons=128, model_dropout_rate=0.2, input_shape=(10, 1)):
#     model = Sequential([
#         LSTM(model_neurons, activation='tanh', return_sequences=True, input_shape=input_shape),
#         BatchNormalization(),
#         Dropout(model_dropout_rate),
#         LSTM(model_neurons // 2, activation='tanh'),
#         BatchNormalization(),
#         Dropout(model_dropout_rate),
#         Dense(1)
#     ])
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
#     return model

def build_model(window_size, n_steps_ahead):
    """Xây dựng mô hình stacked LSTM với 2 lớp"""
    model = Sequential()
    # Lớp LSTM thứ nhất với return_sequences=True
    model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(window_size, 1)))
    # Lớp LSTM thứ hai
    model.add(LSTM(50, activation='relu'))
    # Lớp Dense dự báo trực tiếp n_steps_ahead bước
    model.add(Dense(n_steps_ahead))
    model.compile(optimizer='adam', loss='mse')
    return model

def random_search_training_lstm(X_train, Y_train, window_size, n_steps_ahead, n_iter=100):
    """Random search to find the best hyperparameters for LSTM"""
    best_val_loss = float('inf')
    best_params = None
    
    for i in range(n_iter):
        epochs = random.choice([20, 50, 100])
        batch_size = random.choice([16, 32, 64])
        
        model = build_model(window_size,n_steps_ahead)
        early_stop = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
        history = model.fit(X_train, Y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.2,
                            callbacks=[early_stop],
                            verbose=0)
        current_val_loss = min(history.history['val_loss'])
        print(f"Random search lặp {i+1}: epochs={epochs}, batch_size={batch_size}, val_loss={current_val_loss:.4f}")
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_params = {'epochs': epochs, 'batch_size': batch_size}
    
    return best_params, best_val_loss
    
def random_search_cv_training_lstm(X_train, Y_train, window_size, n_steps_ahead, n_iter=100, n_splits=5):
    """Random search with Cross-Validation to find the best hyperparameters for LSTM"""
    best_val_loss = float('inf')
    best_params = None
    kf = KFold(n_splits=n_splits, shuffle=True)

    for i in range(n_iter):
        # Randomly select hyperparameters
        #model_neurons = random.choice([32, 64, 128])
        #model_dropout_rate = random.choice([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
        epochs = random.choice([20, 50, 100])
        batch_size = random.choice([16, 32, 64])
        
        model = build_model(window_size,n_steps_ahead)
        early_stop = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
        
        fold_losses = []
        for train_index, val_index in kf.split(X_train):
            # Split data into training and validation sets
            X_train_sub, X_val = X_train[train_index], X_train[val_index]
            Y_train_sub, Y_val = Y_train[train_index], Y_train[val_index]

            # Create and compile the LSTM model
            history = model.fit(X_train_sub, Y_train_sub,
                                epochs=epochs,
                                batch_size=batch_size,
                                validation_data=(X_val, Y_val),
                                callbacks=[early_stop],
                                verbose=0)
            current_val_loss = min(history.history['val_loss'])
            fold_losses.append(current_val_loss)

        # Calculate the average loss across all folds
        avg_loss = np.mean(fold_losses)
        print(f"Random search lặp {i+1}: epochs={epochs}, batch_size={batch_size}, val_loss={avg_loss:.4f}")
        # Update the best parameters if the current model has a lower average validation loss
        if avg_loss < best_val_loss:
            best_val_loss = avg_loss
            best_params = {'epochs': epochs, 'batch_size': batch_size}
    return best_params, best_val_loss
####
def bootstrap_lstm_multi_output(x_train, y_train, x_val, y_val, epochs, batch_size, scaler, window_size, n_steps_ahead, n_models=100):
    predictions = []
    obs = []
    
    # Huấn luyện mô hình ban đầu để lấy giá trị dự báo fitted
    base_model = build_model(window_size, n_steps_ahead)
    base_model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Tính residuals
    fitted_values = base_model.predict(x_train)
    residuals = y_train - fitted_values  
    
    for i in range(n_models):
        print(f"Bootstrap iteration: {i+1}/{n_models}")
        
        # Lấy mẫu bootstrap residuals
        bootstrap_residuals = resample(residuals, replace=True)
        y_train_bootstrap = fitted_values + bootstrap_residuals
        
        # Huấn luyện lại model với bootstrap data
        model = MultiOutputRegressor(KerasRegressor(
            build_fn=create_model,
            # model_neurons=model_neurons,
            # model_dropout_rate=model_dropout_rate,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        ))
        model.fit(x_train, y_train_bootstrap)
        
        # Dự báo trên tập validation
        pred_scaled = model.predict(x_val)
        
        # 🔹 Inverse Min-Max Scaler
        pred_unscaled = scaler.inverse_transform(pred_scaled)
        obs_unscaled = scaler.inverse_transform(y_val)

        # # 🔹 Inverse Box-Cox Transform (Nếu dữ liệu ban đầu đã được Box-Cox)
        # pred_original = inv_boxcox(pred_original, param_1)
        # obs_original = inv_boxcox(obs_original, param_1)  # Đảm bảo obs không bị thay đổi cấu trúc

        pred_original = pred_unscaled
        obs_original = obs_unscaled
        
        # Lưu kết quả
        predictions.append(pred_original)
        obs.append(obs_original)
    
    predictions = np.array(predictions)
    observations = np.array(obs)
    
    return predictions, observations

#####################################################3
def fastfurier_denoise(data,cutoff_frequency=0.5):
    fft_result = np.fft.fft(data)
    
    t = np.linspace(0,len(data),len(data))
    frequencies = np.fft.fftfreq(len(t),t[1]-t[0])
    #magnitude = np.abs(fft_result)
    
    # Lọc tần số cao (VD: giữ tần số < 0.5 Hz)
    cutoff_frequency = cutoff_frequency
    filtered_fft = fft_result.copy()
    filtered_fft[np.abs(frequencies)>cutoff_frequency]=0
    
    # Chuyển ngược về miền thời gian
    data_denoised = np.fft.ifft(filtered_fft).real
    return data_denoised
#######################################################3
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.ndimage import uniform_filter1d # dùng cho trung bình trượt MA
from numpy.fft import fft, ifft
############################################################################################################
# def TSR_denoise(data,method = 'ma'): # TSR - Trend, Seasonal Remove
#     data_decomposed = seasonal_decompose(data, model = 'additive', period = 365)
    
#     # Phân tách trend, seasonal, residuals
#     trend_component = data_decomposed.trend
#     seasonal_component = data_decomposed.seasonal
#     residual = data_decomposed.resid
    
#     # Khử nhiễu của residuals
#     if method=='ma': # Khử nhiễu bằng moving average
#         window_size = 3 # có thể thay đổi windowsize cho phù hợp hơn
#         residual_denoised = uniform_filter1d(residual.dropna(),size=window_size)
        
#          # Tái tạo dữ liệu
#         valid_idx = residual.dropna().index
#         data_denoised = trend_component.loc[valid_idx] + seasonal_component.loc[valid_idx] + residual_denoised
#     elif method == 'fft': # Khử nhiễu bằng fast furier transform
#         resid_values = residual.values
#         freq_domain = fft(resid_values)
#         freqs = np.fft.fftfreq(len(resid_values), d=1)  # d=1 vì dữ liệu ngày
#         cutoff_freq = 0.275  # Ngưỡng tần số (chu kỳ < 20 ngày bị loại)
#         freq_domain_filtered = freq_domain.copy()
#         freq_domain_filtered[np.abs(freqs) > cutoff_freq] = 0
#         residual_denoised = np.real(ifft(freq_domain_filtered))
        
#         # Tái tạo dữ liệu
#         valid_idx = residual.index
#         data_denoised = trend_component.loc[valid_idx] + seasonal_component.loc[valid_idx] + residual_denoised
#     elif method == 'wt': # Khử nhiễu bằng wavelet transform
#         wavelet = 'db4'  # Hàm sóng nhỏ Daubechies 4
#         level = 3  # Mức phân tích
#         coeffs = pywt.wavedec(residual.values, wavelet, level=level)  # Phân tách
#         sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Ước lượng độ lệch chuẩn nhiễu
#         threshold = sigma * np.sqrt(2 * np.log(len(residual)))  # Ngưỡng phổ quát
#         coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]  # Lọc nhiễu
#         residual_denoised = pywt.waverec(coeffs_denoised, wavelet)  # Tái tạo

#         # Tái tạo dữ liệu
#         valid_idx = residual.index
#         data_denoised = trend_component.loc[valid_idx] + seasonal_component.loc[valid_idx] + residual_denoised[:len(valid_idx)] 
#     return data_denoised

##################################################################################################################################

def TSR_denoise(data, method='ma'): # không chạy ma, chỉ chạy fft và wt
    """
    Khử nhiễu tín hiệu thời gian bằng các phương pháp khác nhau: moving average, FFT, Wavelet.
    
    Tham số:
    - data (numpy.array): Dữ liệu đầu vào dạng mảng numpy.
    - method (str): Phương pháp khử nhiễu ('ma', 'fft', 'wt').

    Trả về:
    - data_denoised (numpy.array): Dữ liệu sau khi được khử nhiễu.
    """
    data_decomposed = seasonal_decompose(data, model='additive', period=365, extrapolate_trend=True)
    
    # Chuyển tất cả thành numpy array
    trend_component = data_decomposed.trend
    seasonal_component = data_decomposed.seasonal
    residual = data_decomposed.resid

    # Xử lý NaN
    trend_component = np.nan_to_num(trend_component)
    seasonal_component = np.nan_to_num(seasonal_component)
    residual = np.nan_to_num(residual)

    # Khử nhiễu residual
    if method == 'ma':  # Moving Average ( KHÔNG CHẠY)
        window_size = 3
        residual_denoised = uniform_filter1d(residual, size=window_size)

    elif method == 'fft':  # Fast Fourier Transform
        freq_domain = fft(residual)
        freqs = np.fft.fftfreq(len(residual), d=1)
        cutoff_freq = 0.275  
        freq_domain[np.abs(freqs) > cutoff_freq] = 0
        residual_denoised = np.real(ifft(freq_domain))

    elif method == 'wt':  # Wavelet Transform
        wavelet = 'db4'
        level = 3
        coeffs = pywt.wavedec(residual, wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(residual)))
        coeffs_denoised = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]
        residual_denoised = pywt.waverec(coeffs_denoised, wavelet)
        residual_denoised = residual_denoised[:len(residual)]  # Cắt để khớp kích thước

    else:
        raise ValueError("Phương pháp không hợp lệ. Chọn 'ma', 'fft' hoặc 'wt'.")

    # Tái tạo dữ liệu đã khử nhiễu
    data_denoised = trend_component + seasonal_component + residual_denoised

    return data_denoised