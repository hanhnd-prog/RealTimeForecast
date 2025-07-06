import pandas as pd
from datetime import date, timedelta
# --- HÀM LẤY DỮ LIỆU ---
def get_station_data(conn, station_id):
    query = f"SELECT record_date, flow_value FROM FlowObservations WHERE station_id = {station_id} ORDER BY record_date"
    #df = pd.read_sql(query, conn, parse_dates=['record_date'])
    #df = df.set_index('record_date')
    
    df = pd.read_sql(query,conn)
    #df['record_date']=pd.to_datetime(df['record_date'],format='ISO8601')
    df['record_date']=pd.to_datetime(df['record_date'],format='%Y-%m-%d').dt.date
    df = df.set_index('record_date')    
    return df

def get_actual_data(conn, station_id, start_date, end_date):
    """Lấy dữ liệu dòng chảy thực đo trong một khoảng thời gian."""
    query = f"""
        SELECT record_date, flow_value
        FROM FlowObservations
        WHERE station_id = ? 
        AND record_date BETWEEN ? AND ?
        ORDER BY record_date
    """
    df = pd.read_sql(query, conn, params=(station_id, start_date, end_date), parse_dates=['record_date'])
    df = df.set_index('record_date')
    return df

# --- HÀM LẤY DỮ LIỆU DỰ BÁO ĐÃ CẬP NHẬT ---
def get_prediction_data(conn, station_id, start_date, end_date, lead_time):
    """
    Lấy dữ liệu dự báo trong một khoảng thời gian VÀ cho một lead time cụ thể.
    """
    query = f"""
        SELECT prediction_for_date, model_name, predicted_value
        FROM FlowPredictions
        WHERE station_id = ?
          AND lead_time = ?
          AND prediction_for_date BETWEEN ? AND ?
    """
    df = pd.read_sql(query, conn, params=(station_id, lead_time, start_date, end_date), parse_dates=['prediction_for_date'])
    
    # Chuyển đổi dữ liệu từ dạng "dài" sang "rộng" để mỗi mô hình là một cột
    if not df.empty:
        df_pivot = df.pivot_table(index='prediction_for_date', columns='model_name', values='predicted_value')
        return df_pivot
    return pd.DataFrame()

def get_realtime_prediction_data(conn,station_id):
    """Lấy dữ liệu dự báo thời gian thực từ bảng FlowPredictions của CSDL

    Args:
        conn (_type_): _description_
        station_id (_type_): _description_
    """
    yesterday = date.today() - timedelta(days=1) # khi cho chạy realtime thì days = 1
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    # value = today - timedelta(days=1)
    # value  = str(value)
    query = f"""
        SELECT prediction_for_date, model_name, predicted_value
        FROM FlowPredictions
        WHERE station_id = ?
          AND prediction_made_on_date = ?
    """
    df = pd.read_sql(query, conn, params=(station_id,yesterday_str))
    df.rename(columns={'prediction_for_date': 'Dự báo cho ngày'}, inplace=True)
    df['predicted_value'] = df['predicted_value'].round(1)
    if not df.empty:
        df_pivot = df.pivot_table(index='Dự báo cho ngày', columns='model_name', values='predicted_value')
        return df_pivot
    return pd.DataFrame()
