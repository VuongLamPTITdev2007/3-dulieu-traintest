import os
import time
import psutil
import tracemalloc
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

# Đường dẫn đến các tập Test (đã được chia từ bước trước)
BASE_DIR = r"D:\Combined_Datasets"
KDD_TEST_FILE = os.path.join(BASE_DIR, "KDD", "KDDTest+.csv")
TON_TEST_FILE = os.path.join(BASE_DIR, "TON", "TONTest.csv")
EDGE_TEST_FILE = os.path.join(BASE_DIR, "Edge", "EdgeTest.csv")

def profile_inference(model, X_test, y_test, dataset_name):
    """
    Hàm này thực hiện đo lường RAM, CPU, Latency trong quá trình dự đoán (Inference).
    - X_test: numpy array hoặc pandas DataFrame chứa features
    - y_test: nhãn thực tế
    - dataset_name: Tên dataset để lưu file log
    """
    print(f"\n[{dataset_name}] Bắt đầu chạy Inference Profiling...")
    logs = []
    
    # Bật theo dõi bộ nhớ (RAM allocation)
    tracemalloc.start()
    
    # Để giả lập môi trường Edge, ta sẽ dự đoán từng dòng (batch_size = 1)
    # Lưu ý: Chạy từng dòng sẽ chậm hơn dự đoán cả cục (batch), nhưng phản ánh đúng thực tế streaming trên Edge.
    # Để tránh chạy quá lâu (vd 30k dòng), ta chỉ test trên 1000 dòng đầu tiên.
    max_samples = min(1000, len(X_test))
    
    # Lấy đối tượng process hiện tại để đo RAM/CPU
    process = psutil.Process(os.getpid())
    
    for i in range(max_samples):
        # Lấy dữ liệu 1 dòng
        if isinstance(X_test, pd.DataFrame):
            x_input = X_test.iloc[i:i+1]
        else:
            x_input = X_test[i:i+1]
            
        y_true_val = y_test.iloc[i] if hasattr(y_test, "iloc") else y_test[i]
        
        # --- BẮT ĐẦU ĐO LƯỜNG ---
        cpu_before = process.cpu_percent(interval=None)
        mem_before = process.memory_info().rss / (1024 * 1024) # Đổi sang MB
        
        start_time = time.perf_counter()
        
        # Thực hiện dự đoán (Inference)
        y_pred_val = model.predict(x_input)[0]
        
        # Đo xác suất nếu mô hình hỗ trợ (dùng cho ROC-AUC)
        y_prob_val = model.predict_proba(x_input)[0][1] if hasattr(model, "predict_proba") else y_pred_val
        
        latency_ms = (time.perf_counter() - start_time) * 1000 # Đổi sang milli-giây
        
        # --- KẾT THÚC ĐO LƯỜNG ---
        cpu_after = process.cpu_percent(interval=None)
        mem_after = process.memory_info().rss / (1024 * 1024)
        
        logs.append({
            "sample_id": i,
            "true_label": y_true_val,
            "pred_label": y_pred_val,
            "pred_prob": y_prob_val,
            "latency_ms": latency_ms,
            "cpu_usage_percent": cpu_after, 
            "ram_usage_mb": mem_after
        })

    tracemalloc.stop()
    
    # 1. Lưu logs ra file CSV
    df_log = pd.DataFrame(logs)
    log_file = os.path.join(BASE_DIR, f"{dataset_name}_Inference_Log.csv")
    df_log.to_csv(log_file, index=False)
    print(f"[{dataset_name}] Đã lưu file log tại: {log_file}")
    
    # 2. Đối chiếu chỉ số thuật toán
    y_true_list = df_log["true_label"]
    y_pred_list = df_log["pred_label"]
    y_prob_list = df_log["pred_prob"]
    
    # Có thể xảy ra trường hợp nhãn chuỗi (String), ta cần quy đổi sang số 0/1 để tính ROC-AUC
    # Giả định ở đây là nhãn nhị phân đã được encode, nếu chưa thì Accuracy vẫn tính được.
    acc = accuracy_score(y_true_list, y_pred_list)
    prec = precision_score(y_true_list, y_pred_list, average='weighted', zero_division=0)
    rec = recall_score(y_true_list, y_pred_list, average='weighted', zero_division=0)
    f1 = f1_score(y_true_list, y_pred_list, average='weighted', zero_division=0)
    
    try:
        roc = roc_auc_score(y_true_list, y_prob_list)
    except:
        roc = "N/A (Cần nhãn 0/1 để tính ROC)"
        
    print(f"\n--- ĐỐI CHIẾU CHỈ SỐ THUẬT TOÁN ({dataset_name}) ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc}")
    
    # 3. Chỉ số hệ thống
    print(f"\n--- CHỈ SỐ HỆ THỐNG / EDGE PROFILING ({dataset_name}) ---")
    print(f"Độ trễ TB (Latency): {df_log['latency_ms'].mean():.4f} ms / mẫu")
    print(f"CPU Usage TB:        {df_log['cpu_usage_percent'].mean():.2f} %")
    print(f"Peak RAM Usage:      {df_log['ram_usage_mb'].max():.2f} MB\n")

if __name__ == "__main__":
    # Ví dụ mẫu: Để chạy profiling, ta cần một mô hình và dữ liệu Test
    # Ở đây tôi tạo một kịch bản giả lập (Mock) với Random Forest trên 1 trong 3 tập dữ liệu.
    
    print("Mã nguồn Profiling RAM, CPU và Log đối chiếu thuật toán.")
    print("Mô hình sẽ sinh ra tập dữ liệu ngẫu nhiên giả lập nếu file chưa sẵn sàng.")
    
    # --- GIẢ LẬP DỮ LIỆU ĐỂ TEST TOOL ---
    # Thay vì load file lớn và train mất thời gian, ta sẽ tạo dữ liệu ngẫu nhiên 
    # tượng trưng cho KDD, TON, Edge để mô phỏng cách hoạt động.
    
    datasets = ["KDD", "TON", "Edge"]
    
    for ds in datasets:
        # Giả lập 1000 mẫu test, 41 features
        X_mock = pd.DataFrame(np.random.rand(1000, 41))
        y_mock = pd.Series(np.random.randint(0, 2, 1000))
        
        # Huấn luyện mô hình siêu nhanh để có cái predict
        model = RandomForestClassifier(n_estimators=10, max_depth=5)
        model.fit(X_mock, y_mock)
        
        # Gọi hàm Profiling
        profile_inference(model, X_mock, y_mock, ds)

