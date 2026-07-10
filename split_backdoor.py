import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os

input_file = r"D:\Combined_Datasets\Edge\Attack traffic\Backdoor_attack.csv"
if not os.path.exists(input_file):
    input_file = r"D:\Edgedataset\Edge-IIoTset dataset\Attack traffic\Backdoor_attack.csv"

print(f"Loading {input_file}...")
df = pd.read_csv(input_file, low_memory=False)

# Giữ lại nhãn
target_col = 'Attack_label'
if target_col not in df.columns:
    target_col = 'Attack_type'

y = df[target_col]

# Loại bỏ các cột object/string để mô hình có thể dùng được ngay
df_numeric = df.select_dtypes(include=[np.number])
if target_col not in df_numeric.columns:
    df_numeric[target_col] = y

# Chia train/test (80/20)
train_df, test_df = train_test_split(df_numeric, test_size=0.2, random_state=42)

out_train = r"D:\Combined_Datasets\Edge\EdgeTrain.csv"
out_test = r"D:\Combined_Datasets\Edge\EdgeTest.csv"

# Tạo thư mục nếu chưa có
os.makedirs(os.path.dirname(out_train), exist_ok=True)

train_df.to_csv(out_train, index=False)
test_df.to_csv(out_test, index=False)

print(f"Đã tạo {out_train} với {len(train_df)} dòng.")
print(f"Đã tạo {out_test} với {len(test_df)} dòng.")
print(f"Số lượng nhãn trong Train:\n{train_df[target_col].value_counts()}")
