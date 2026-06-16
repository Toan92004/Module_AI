import os
import json
import h5py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import numpy as np
from collections import deque
import time
import uvicorn

app = FastAPI(title="AI Fire Warning - Dynamic N-Nodes")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL_PATH = "forecast_model.h5"
model = None

# -------------------------------------------------------------
# BÁC SĨ PHẪU THUẬT V2: QUÉT ĐỆ QUY & SỬA LỖI DTYPEPOLICY
# -------------------------------------------------------------
def clean_frankenstein_model(filepath):
    import json
    import h5py
    try:
        with h5py.File(filepath, 'r+') as f:
            if 'model_config' in f.attrs:
                config_str = f.attrs['model_config']
                if isinstance(config_str, bytes):
                    config_str = config_str.decode('utf-8')
                config_json = json.loads(config_str)
                
                # Hàm đệ quy quét sâu vào mọi tầng cấu trúc
                def fix_config(d):
                    if isinstance(d, dict):
                        # 1. Trị dứt điểm khối u DTypePolicy
                        if d.get('class_name') == 'DTypePolicy' and 'config' in d:
                            return d['config'].get('name', 'float32')
                            
                        new_d = {}
                        for k, v in d.items():
                            # 2. Vứt bỏ các thông số đời mới gây dị ứng
                            if k in ['optional', 'quantization_config', 'registered_name', 'build_config', 'compile_config']:
                                continue
                            
                            # 3. Chữa tên gọi batch_shape
                            new_key = 'batch_input_shape' if k == 'batch_shape' else k
                            new_d[new_key] = fix_config(v)
                        return new_d
                    elif isinstance(d, list):
                        return [fix_config(i) for i in d]
                    else:
                        return d
                        
                # Áp dụng nội soi toàn bộ
                config_json = fix_config(config_json)
                
                # Ghi đè lại file
                new_config_str = json.dumps(config_json).encode('utf-8')
                f.attrs['model_config'] = new_config_str
                print("-> [HỆ THỐNG] Đã phẫu thuật DTypePolicy & dọn rác .h5 thành công!")
    except Exception as e:
        print("-> [HỆ THỐNG] Bỏ qua phẫu thuật do:", e)
# -------------------------------------------------------------

# Các biến môi trường khác của bạn
MIN_VALUES = np.array([0.0, 0.0, 0.0, 0.0])
MAX_VALUES = np.array([70.0, 100.0, 1000.0, 1000.0])
TIME_STEPS = 12
DELTA_AI_UPDATE_SEC = 900  

node_buffers: Dict[str, deque] = {}
last_30s_data: Dict[str, np.ndarray] = {}
last_ai_update_time = time.time()

# KHỞI ĐỘNG HỆ THỐNG THÔNG MINH
@app.on_event("startup")
async def load_ai_model():
    global model
    if os.path.exists(MODEL_PATH):
        print("-> [AI ENGINE] Đang nạp thư viện và bộ não AI, vui lòng đợi...")
        
        # 1. Gọi bác sĩ phẫu thuật dọn file trước
        clean_frankenstein_model(MODEL_PATH)
        
        # 2. Bắt đầu lôi TensorFlow ra đọc (Lúc này file đã sạch sẽ hoàn toàn)
        import tensorflow as tf 
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("-> [AI ENGINE] Bộ não 4 biến (0-70°C) đã sẵn sàng!")
    else:
        print("-> [AI ENGINE] CẢNH BÁO: Không tìm thấy file forecast_model.h5!")

# ĐÃ SỬA LỖI: Loại bỏ hoàn toàn b_sen để đồng bộ với Node.js và phần cứng mạch Uno
class NodeData(BaseModel):
    t: float; h: float; p2: float; p10: float; b_uno: float

def predict_for_node(buffer_data):
    if len(buffer_data) == TIME_STEPS and model is not None:
        scaled_input = (np.array(buffer_data) - MIN_VALUES) / (MAX_VALUES - MIN_VALUES)
        scaled_input = np.expand_dims(scaled_input, axis=0)
        scaled_pred = model.predict(scaled_input, verbose=0)[0]
        return {
            "temp": round(float(scaled_pred[0] * (MAX_VALUES[0] - MIN_VALUES[0]) + MIN_VALUES[0]), 1),
            "hum": round(float(scaled_pred[1] * (MAX_VALUES[1] - MIN_VALUES[1]) + MIN_VALUES[1]), 1),
            "pm25": round(float(scaled_pred[2] * (MAX_VALUES[2] - MIN_VALUES[2]) + MIN_VALUES[2]), 1),
            "pm10": round(float(scaled_pred[3] * (MAX_VALUES[3] - MIN_VALUES[3]) + MIN_VALUES[3]), 1)
        }
    return None

def calculate_risk(forecast):
    risk_temp = max(0.0, min((forecast["temp"] - 45.0) / (70.0 - 45.0), 1.0)) * 100
    risk_pm25 = max(0.0, min((forecast["pm25"] - 50.0) / (300.0 - 50.0), 1.0)) * 100
    risk_pm10 = max(0.0, min((forecast["pm10"] - 80.0) / (400.0 - 80.0), 1.0)) * 100
    return round(max(risk_temp, risk_pm25, risk_pm10), 1)

@app.post("/api/predict")
async def predict_environment(data: Dict[str, NodeData]):
    global node_buffers, last_30s_data, last_ai_update_time
    current_time = time.time()
    
    missed_intervals = int((current_time - last_ai_update_time) // DELTA_AI_UPDATE_SEC)
    is_time_to_update = missed_intervals >= 1
    
    results = {}
    max_risk = 0.0
    critical_node = "Không xác định"
    low_battery_nodes = []

    for node_id, sensor in data.items():
        if node_id not in node_buffers:
            node_buffers[node_id] = deque(maxlen=TIME_STEPS)
            last_30s_data[node_id] = None
            print(f"[*] Đã tự động cấp phát bộ nhớ AI cho trạm mới: {node_id}")

        curr_array = np.array([sensor.t, sensor.h, sensor.p2, sensor.p10])
        should_run_ai_for_this_node = False

        spike = False
        if last_30s_data[node_id] is not None:
            if abs(curr_array[0] - last_30s_data[node_id][0]) >= 2.0 or abs(curr_array[2] - last_30s_data[node_id][2]) >= 40.0:
                spike = True
        last_30s_data[node_id] = curr_array

        if is_time_to_update:
            if len(node_buffers[node_id]) == 0 or missed_intervals > 8:
                node_buffers[node_id].clear()
                for _ in range(TIME_STEPS): node_buffers[node_id].append(curr_array)
            else:
                for _ in range(missed_intervals): node_buffers[node_id].append(curr_array)
            should_run_ai_for_this_node = True
            
        if spike and len(node_buffers[node_id]) > 0:
            node_buffers[node_id][-1] = curr_array
            should_run_ai_for_this_node = True

        if should_run_ai_for_this_node:
            fc = predict_for_node(node_buffers[node_id])
            if fc:
                r = calculate_risk(fc)
                results[node_id] = {"forecast": fc, "risk": r}
                if r > max_risk:
                    max_risk = r
                    critical_node = node_id

        # ĐÃ SỬA LỖI: Chỉ kiểm tra b_uno, bỏ b_sen
        if sensor.b_uno < 2.6:
            low_battery_nodes.append(node_id)

    if is_time_to_update:
        last_ai_update_time += missed_intervals * DELTA_AI_UPDATE_SEC

    risk_level = "safe"
    message = f"Hệ thống ổn định ({len(data)} trạm hoạt động)."

    if max_risk >= 70.0:
        risk_level = "critical"
        message = f"🚨 CHÁY TẠI {critical_node.upper()}! Rủi ro: {max_risk}%"
    elif max_risk >= 40.0:
        risk_level = "warning"
        message = f"⚠️ Nhiệt/bụi tăng nhanh tại {critical_node} ({max_risk}%)"
    elif len(low_battery_nodes) > 0 and risk_level == "safe":
        risk_level = "warning"
        message = f"⚠️ Trạm {', '.join(low_battery_nodes)} sắp hết pin!"

    return {
        "status": "success",
        "active_nodes_count": len(data),
        "node_results": results, 
        "global_risk": {"score": max_risk, "level": risk_level, "message": message}
    }

if __name__ == "__main__":
    uvicorn.run("ai_service:app", host="0.0.0.0", port=8000)