import streamlit as st
import pandas as pd
import sqlite3
import joblib
import plotly.graph_objects as go
from datetime import datetime, timedelta
from model_utils import *

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Dự báo Dòng chảy",
    page_icon="🌊",
    layout="wide"
)

# --- KẾT NỐI CSDL (Cache để tăng tốc) ---
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect('data/streamflow.db',check_same_thread=False)
    return conn

# # --- TẢI MÔ HÌNH (Cache để không phải tải lại) ---
# @st.cache_resource
# def load_models(station_id):
#     # Đường dẫn có thể thay đổi tùy theo cách bạn lưu tên file
#     lgbm_model = joblib.load(f'models/lgbm_station_{station_id}.pkl')
#     rf_model = joblib.load(f'models/rf_station_{station_id}.pkl')
#     lstm_model = load_model(f'models/lstm_station_{station_id}.h5')
#     return {'LGBM': lgbm_model, 'RF': rf_model, 'LSTM': lstm_model}

# --- GIAO DIỆN CHÍNH ---
st.title("🌊 Dự báo dòng chảy thượng lưu sông Đà")
st.markdown("Phần mềm này là sản phẩm của đề tài ĐTĐL.CN.06.23")


# --- SIDEBAR: LỰA CHỌN CỦA NGƯỜI DÙNG ---
conn = get_db_connection()
stations_df = pd.read_sql("SELECT station_id, station_name FROM Stations", conn)
station_dict = pd.Series(stations_df.station_id.values, index=stations_df.station_name).to_dict()

st.sidebar.header("Tùy chọn vị trí dự báo")
selected_station_name = st.sidebar.selectbox("Chọn trạm quan trắc:", options=list(station_dict.keys()))
selected_station_id = station_dict[selected_station_name]

# --- TẢI DỮ LIỆU VÀ MÔ HÌNH TƯƠNG ỨNG ---
df_hist = get_station_data(conn, selected_station_id)
# models = load_models(selected_station_id)


st.header(f"Vị trí dự báo: {selected_station_name}")
# st.header(f"Mã trạm: {selected_station_id}")
# st.dataframe(df_hist.head())

# --- HIỂN THỊ DỮ LIỆU LỊCH SỬ ---
st.subheader("Dữ liệu dòng chảy quan trắc")
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_hist.index,
    #x=df_hist['record_date'], 
    y=df_hist['flow_value'], 
    mode='lines',
    line=dict(color = 'skyblue'),
    name='Dòng chảy thực đo'))
fig.update_layout(title={'text':'Biểu đồ dòng chảy theo thời gian','x':0.5,'xanchor':'center'}, xaxis_title='Thời gian', yaxis_title='Lưu lượng (m³/s)')
st.plotly_chart(fig, use_container_width=True)

st.sidebar.subheader("Phân tích quá khứ:")
today = datetime.today()
date_range = st.sidebar.date_input(
    "Chọn khoảng thời gian xem xét:",
    value = (today - timedelta(days=10),today),
    max_value=today
)
start_date,end_date = date_range
# st.header(str(start_date)+','+str(end_date))
start_date = str(start_date)
end_date = str(end_date)
# 3. Chọn lead time dự báo
# st.sidebar.subheader("Tùy chọn dự báo")
# Giả sử bạn dự báo trước từ 1 đến 10 ngày
selected_lead_time = st.sidebar.selectbox(
    "Chọn thời gian dự kiến (số bước):",
    options=list(range(1, 11)), # Thay đổi range này nếu bạn có thời gian dự kiến khác
    index=0 # Mặc định là dự báo 1 ngày
)

# --- TẢI DỮ LIỆU DỰA TRÊN LỰA CHỌN ---
df_actual = get_actual_data(conn, selected_station_id, str(start_date), str(end_date))
df_pred = get_prediction_data(conn, selected_station_id, start_date, end_date, selected_lead_time)
# st.subheader(f"df_pred từ ngày: {start_date} tới ngày {end_date}")
# st.dataframe(df_pred.head())
# st.subheader(f"df_actual từ ngày: {start_date} tới ngày: {end_date}")
# st.dataframe(df_actual.head())

# Gộp dữ liệu thực đo và dự báo vào cùng một DataFrame để so sánh
df_comparison = df_actual.join(df_pred, how='left')
df_comparison.rename(columns={'flow_value': 'Thực đo'}, inplace=True)

# --- HIỂN THỊ KẾT QUẢ ---
st.subheader(f"So sánh kết quả dự báo trong quá khứ cho trạm: {selected_station_name}")
st.info(f"Đang hiển thị so sánh cho các dự báo trong quá khứ có thời gian dự kiến **{selected_lead_time} ngày**.")
# st.dataframe(df_comparison.head())
# 1. BIỂU ĐỒ SO SÁNH
fig = go.Figure()

# Vẽ đường thực đo
fig.add_trace(go.Scatter(
    x=df_comparison.index,
    y=df_comparison['Thực đo'],
    mode='lines',
    name='Thực đo',
    line=dict(color='royalblue', width=3)
))

# Vẽ các đường dự báo (nếu có dữ liệu)
if not df_pred.empty:
    model_colors = {'LGBM': 'orange', 'RF': 'green', 'LSTM': 'firebrick'}
    for model_name in df_pred.columns:
        fig.add_trace(go.Scatter(
            x=df_comparison.index,
            y=df_comparison[model_name],
            mode='lines+markers',
            name=f'Dự báo {model_name}',
            line=dict(color=model_colors.get(model_name, 'gray'), dash='dash'),
            marker=dict(size=8)
        ))

fig.update_layout(
    title=f'Biểu đồ so sánh Dòng chảy Thực đo và Dự báo trong quá khứ (Thời gian dự kiến: {selected_lead_time} ngày)',
    xaxis_title='Ngày',
    yaxis_title='Lưu lượng (m³/s)',
    legend_title="Chú giải"
)
st.plotly_chart(fig, use_container_width=True)

# 2. BẢNG ĐÁNH GIÁ DỰ BÁO TRONG QUÁ KHỨ
st.subheader(f"Đánh giá hiệu suất mô hình (Thời gian dự kiến: {selected_lead_time} ngày)")
#st.markdown(f"Tính toán sai số trong khoảng thời gian từ **{start_date.strftime('%d/%m/%Y')}** đến **{end_date.strftime('%d/%m/%Y')}**.")
st.markdown(f"Đánh giá hiệu suất các mô hình trong khoảng thời gian từ **{start_date}** đến **{end_date}**.")

# Bỏ các hàng không có cả giá trị thực đo và dự báo để đánh giá
df_eval = df_comparison.dropna()

if not df_eval.empty:
    metrics = {}
    # Chỉ tính toán cho các cột mô hình có trong df_eval
    model_columns = [col for col in df_pred.columns if col in df_eval.columns]
    
    for model_name in model_columns:
        # Tính toán RMSE (Root Mean Squared Error)
        rmse = ((df_eval[model_name] - df_eval['Thực đo']) ** 2).mean() ** 0.5
        # Tính toán MAE (Mean Absolute Error)
        mae = (df_eval[model_name] - df_eval['Thực đo']).abs().mean()
        metrics[model_name] = {'RMSE': f"{rmse:.2f}", 'MAE': f"{mae:.2f}"}

    st.dataframe(pd.DataFrame(metrics).T.rename_axis('Mô hình'), use_container_width=True)
else:
    st.warning("Không có đủ dữ liệu để đánh giá trong khoảng thời gian và thời gian dự kiến đã chọn.")

# --- THỰC HIỆN DỰ BÁO "REAL-TIME" ---
st.sidebar.subheader("Tùy chọn mô hình dự báo thời gian thực:")
option_LGBM = st.sidebar.checkbox("LGBM")
option_RF   = st.sidebar.checkbox("RF")
option_LSTM = st.sidebar.checkbox("LSTM")

st.header(f"Kết quả dự báo thời gian thực cho trạm: {selected_station_name}")

# Lấy dữ liệu dự báo cho hiện tại từ bảng FlowPredictions
df_realtime_pred = get_realtime_prediction_data(conn,selected_station_id)

st.subheader("Bảng kết quả dự báo thời gian thực:")
st.dataframe(df_realtime_pred)

st.subheader("Đồ thị kết quả dự báo thời gian thực:")
# Lấy dữ liệu quan trắc trong 10 ngày gần nhất
ngay_cuoi_quantrac = datetime.today() - timedelta(days=1)
ngay_cuoi_quantrac = ngay_cuoi_quantrac.strftime('%Y-%m-%d')
# st.header(ngay_cuoi_quantrac)
ngay_dau_quantrac = datetime.today() - timedelta(days=11)
ngay_dau_quantrac = ngay_dau_quantrac.strftime('%Y-%m-%d')

df_obs = get_actual_data(conn, selected_station_id, ngay_dau_quantrac, ngay_cuoi_quantrac)
# st.dataframe(df_obs)

# Chuan bi du lieu de ve phan du lieu du bao
#add_row = pd.DataFrame({'LGBM':[df_obs['flow_value'][-1]]},index=pd.to_datetime(ngay_cuoi_quantrac))
add_row = pd.DataFrame({'LGBM':[df_obs['flow_value'][-1]]},index=[ngay_cuoi_quantrac])
df_add_row_to_realtime_pred = pd.concat([add_row,df_realtime_pred])
# st.dataframe(df_add_row_to_realtime_pred) 

# Ve hinh ket qua du bao
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_obs.index,
    y=df_obs['flow_value'],
    mode='lines',
    name='Quan trắc gần đây',
    line=dict(color='skyblue')
))
fig.add_trace(go.Scatter(
    x=df_add_row_to_realtime_pred.index,
    y=df_add_row_to_realtime_pred['LGBM'],
    mode='lines+markers',
    name='Dự báo bằng mô hình LGBM',
    line=dict(dash='dash',color='darkorange'),
    marker=dict(symbol='diamond',size=8)
))

st.plotly_chart(fig)


