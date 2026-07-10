import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = r"D:\Combined_Datasets"
TON_FILE = os.path.join(BASE_DIR, "TON_Train_Test_Windows_7.csv")
EDGE_DIR = os.path.join(BASE_DIR, "Edge", "Selected dataset for ML and DL")
EDGE_ML_FILE = os.path.join(EDGE_DIR, "ML-EdgeIIoT-dataset.csv")

TON_OUT_DIR = os.path.join(BASE_DIR, "TON")
EDGE_OUT_DIR = os.path.join(BASE_DIR, "Edge")
os.makedirs(TON_OUT_DIR, exist_ok=True)

def split_and_save(file_path, train_out, test_out, dataset_name):
    print(f"Đang xử lý dataset {dataset_name}...")
    if not os.path.exists(file_path):
        print(f"  -> Không tìm thấy tệp {file_path}")
        return
        
    try:
        # Load data
        print(f"  -> Đọc file: {file_path}")
        # Dùng low_memory=False để tránh warning với file lớn
        df = pd.read_csv(file_path, low_memory=False)
        
        print(f"  -> Tổng số bản ghi: {len(df):,}")
        
        # Tách Train/Test (80% Train, 20% Test)
        # Bật shuffle để trộn dữ liệu cho khách quan
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
        
        print(f"  -> Chia xong: Train ({len(train_df):,}) - Test ({len(test_df):,})")
        
        # Ghi ra file
        print(f"  -> Đang lưu file Train...")
        train_df.to_csv(train_out, index=False)
        print(f"  -> Đang lưu file Test...")
        test_df.to_csv(test_out, index=False)
        
        print(f"  -> Hoàn thành dataset {dataset_name}.\n")
    except Exception as e:
        print(f"  -> Lỗi khi xử lý {dataset_name}: {e}\n")

if __name__ == "__main__":
    print("Bắt đầu phân chia dữ liệu...\n")
    
    # TON Dataset
    split_and_save(
        TON_FILE,
        os.path.join(TON_OUT_DIR, "TONTrain.csv"),
        os.path.join(TON_OUT_DIR, "TONTest.csv"),
        "TON"
    )
    
    # Edge Dataset (ML version)
    split_and_save(
        EDGE_ML_FILE,
        os.path.join(EDGE_OUT_DIR, "EdgeTrain.csv"),
        os.path.join(EDGE_OUT_DIR, "EdgeTest.csv"),
        "Edge-IIoTset"
    )
    
    # Có thể tự xoá file TON gốc nếu muốn tiết kiệm dung lượng
    # os.remove(TON_FILE)
    
    print("HOÀN TẤT TẤT CẢ!")
