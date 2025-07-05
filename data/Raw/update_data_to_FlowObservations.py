import requests
import json
import sys
import os
import pandas as pd
import sqlite3

from datetime import datetime,timedelta

# URL API
url = "https://thuydienvietnam.vn/jaxrs/QuanTracHoChua/getDataQuanTracTB"

# Danh sách các trạm với Tentram, filename (ten file rawdata), updated_file (ten file data moi duoc update)
data_LaiChau_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'LaiChau_Q.txt'))
data_BanChat_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'BanChat_Q.txt'))

LaiChau_updated_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'LaiChau_updated_Q.txt'))
BanChat_updated_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'BanChat_updated_Q.txt'))

LaiChau_daily_input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Inputs/LaiChau.csv'))
BanChat_daily_input_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Inputs/BanChat.csv'))

# Danh sách các trạm
stations = [
    {
        "hc_uuid": "e9d6597d-66a1-4a0f-b6f4-1da275bf3db3",
        "tents": "Q đến hồ",
        "mats": "QDEN",
        "station_id": 1,
        "Tentram": "LaiChau",
        "filename": data_LaiChau_path,
        "updated_file": LaiChau_updated_path,
        "daily_input_file": LaiChau_daily_input_path
    },
    {
        "hc_uuid": "93f34e29-16db-4ce5-81ed-688f2e45b7b4",
        "tents": "Q đến hồ",
        "mats": "QDEN",
        "station_id": 2,
        "Tentram": "BanChat",
        "filename": data_BanChat_path,
        "updated_file": BanChat_updated_path,
        "daily_input_file": BanChat_daily_input_path
    }
    # Bạn có thể thêm các trạm khác tại đây
]

# -- Get update data to file Tentram_updated_Q.txt --
# Các thông tin chung
# tungay = "2025-05-02 00:00:00"
# tungay = "2015-01-01 00:00:00"
with open(LaiChau_updated_path,"r",encoding="utf-8") as f:
    lines = f.readlines()
    last_line = lines[-1].strip()
    tungay = last_line.split('\t')[0].strip('"')
    tungay=tungay[:-1]+"1" # tang thoi gian len 1s, tranh bi lap du lieu cap nhat

# denngay = "2025-05-17 23:00:00"
yesterday = datetime.now() - timedelta(days=1)
yesterday_235959 = datetime(
    year=yesterday.year,
    month=yesterday.month,
    day=yesterday.day,
    hour=23,
    minute=59,
    second=59
)

# now = str(datetime.now())
# denngay = now
denngay = str(yesterday_235959)

#namdulieu = "2025"
namdulieu = str(yesterday.year)
# namht = 2025
namht = datetime.now().year

# Lặp qua từng trạm và gửi yêu cầu
for station in stations:
    payload = {
        "data": {
            "hc_uuid": station["hc_uuid"],
            "tents": station["tents"],
            "mats": station["mats"],
            "tungay": tungay,
            "denngay": denngay,
            "namdulieu": namdulieu,
            "namht": namht,
            "cua": ""
        }
    }

    print(f"🔄 Đang lấy dữ liệu cho: {station['tents']}")

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        try:
            data = response.json()
            records = data.get("dtDataTable", [])

            with open(station["filename"], "a", encoding="utf-8") as f:
                # Tiêu đề phụ thuộc vào loại dữ liệu (mats)
                #f.write("thoigian\tgiatri\n")
                for row in records:
                    thoigian = row.get("data_thoigian", "")
                    giatri = row.get(station["mats"].lower(), "")
                    f.write(f"\"{thoigian}\"\t{giatri}\n")

            print(f"✅ Đã ghi dữ liệu ra file: {station['filename']}")

            with open(station["updated_file"], "w", encoding="utf-8") as f2:
                # Tiêu đề phụ thuộc vào loại dữ liệu (mats)
                f2.write("thoigian\tgiatri\n")
                for row in records:
                    thoigian = row.get("data_thoigian", "")
                    giatri = row.get(station["mats"].lower(), "")
                    f2.write(f"\"{thoigian}\"\t{giatri}\n")
                    
        except Exception as e:
            print(f"❌ Lỗi khi xử lý JSON cho {station['tents']}: {e}")
    else:
        print(f"❌ Yêu cầu thất bại ({station['tents']}). Mã trạng thái: {response.status_code}")
        
# -- Update data to FlowObservations table of streamflow.db database --
# duong dan den CSDL: streamflow.db
db_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../streamflow.db'))
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

for station in stations:
    # 1. Doc file du lieu duoc cap nhat raw: VD: LaiChau_updated_Q.txt
    station_id = station["station_id"]
      
    df = pd.read_csv(station["updated_file"], delimiter='\t', parse_dates=['thoigian'])
    
    # 2. Tinh toan, xu ly du lieu daily
    df['date']=df['thoigian'].dt.date
    df_avg = df.groupby('date')['giatri'].mean().reset_index()
    df_avg.columns=['date','value']
        
    # 3. Update vao bang FlowObservations cua database streamflow.db
    for row in df_avg.itertuples(index=False):
        record_date = str(row.date)
        flow_value = float(row.value)
        cursor.execute("""
            INSERT INTO FlowObservations (station_id, record_date, flow_value)
            SELECT ?, ?, ?
            WHERE NOT EXISTS (
                SELECT 1 FROM FlowObservations
                WHERE station_id = ? AND record_date = ?
            )
        """, (station_id, record_date, flow_value, station_id, record_date))
    
    # # 3. Update vao file input: ../Inputs/LaiChau.csv
    # df_old = pd.read_csv(station['daily_input_file'],parse_dates=['date'])
    
    # df_avg['date']=pd.to_datetime(df_avg['date'])
    
    # df_combined = pd.concat([df_old, df_avg]).drop_duplicates(subset='date').sort_values(by='date')
    
    # df_combined.to_csv(station['daily_input_file'], index=False)      

conn.commit()
conn.close()