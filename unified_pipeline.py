import os
import time
import gc
import psutil
import tracemalloc
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from stable_baselines3 import PPO, DQN
import gymnasium as gym
from gymnasium import spaces

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

BASE_DIR = r"D:\Combined_Datasets"
RESULTS_FILE = os.path.join(BASE_DIR, "evaluation_results_all.csv")

DATASETS = {
    "KDD": {
        "train": os.path.join(BASE_DIR, "KDD", "KDDTrain+.csv"),
        "test": os.path.join(BASE_DIR, "KDD", "KDDTest+.csv"),
        "target": "label",
        "normal_val": "normal"
    },
    "TON": {
        "train": os.path.join(BASE_DIR, "TON", "TONTrain.csv"),
        "test": os.path.join(BASE_DIR, "TON", "TONTest.csv"),
        "target": "type",
        "normal_val": "normal"
    },
    "Edge": {
        "train": os.path.join(BASE_DIR, "Edge", "EdgeTrain.csv"),
        "test": os.path.join(BASE_DIR, "Edge", "EdgeTest.csv"),
        "target": "Attack_label",
        "normal_val": 0
    }
}

# ==========================================
# 1. PREPROCESSING
# ==========================================
def load_and_preprocess(ds_info, name):
    print(f"\n--- Loading {name} Dataset ---")
    
    # Hàm đọc KDD (file không có header chuẩn)
    if name == "KDD":
        cols = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
                "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
                "root_shell","su_attempted","num_root","num_file_creations","num_shells",
                "num_access_files","num_outbound_cmds","is_hot_login","is_guest_login","count",
                "srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
                "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
                "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
                "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
                "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate",
                "label","difficulty_level"]
        df_train = pd.read_csv(ds_info["train"], names=cols, header=0 if pd.read_csv(ds_info["train"], nrows=1).shape[1] == 43 else None)
        df_test = pd.read_csv(ds_info["test"], names=cols, header=0 if pd.read_csv(ds_info["test"], nrows=1).shape[1] == 43 else None)
    else:
        df_train = pd.read_csv(ds_info["train"], low_memory=False)
        df_test = pd.read_csv(ds_info["test"], low_memory=False)

    # Giảm sample để chạy mô phỏng nhanh
    df_train = df_train.sample(min(15000, len(df_train)), random_state=42)
    df_test = df_test.sample(min(5000, len(df_test)), random_state=42)
    
    target_col = ds_info["target"]
    normal_val = ds_info["normal_val"]
    
    # Binary Label
    y_train = (df_train[target_col] != normal_val).astype(int).values
    y_test = (df_test[target_col] != normal_val).astype(int).values
    
    # Drop target and non-numeric
    X_train = df_train.drop(columns=[target_col], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    X_test = df_test.drop(columns=[target_col], errors='ignore').select_dtypes(include=[np.number]).fillna(0)
    
    # Căn chỉnh cột (trường hợp train/test khác biệt)
    common_cols = X_train.columns.intersection(X_test.columns)
    X_train = X_train[common_cols]
    X_test = X_test[common_cols]
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    return X_train, y_train, X_test, y_test

# ==========================================
# 2. GYM ENV FOR RL
# ==========================================
class IdsEnv(gym.Env):
    def __init__(self, X, y):
        super(IdsEnv, self).__init__()
        self.X, self.y = X, y
        self.n = len(X)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(X.shape[1],), dtype=np.float32)
        self.idx = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        return self.X[self.idx], {}

    def step(self, action):
        reward = 1.0 if action == self.y[self.idx] else -1.0
        self.idx += 1
        done = self.idx >= self.n
        obs = self.X[self.idx] if not done else np.zeros_like(self.X[0])
        return obs, reward, done, False, {}

# ==========================================
# 3. PROFILING ENGINE
# ==========================================
def profile_predict(predict_func, X_test, y_test, is_keras=False, is_rl=False, is_tflite=False):
    logs = []
    process = psutil.Process(os.getpid())
    tracemalloc.start()
    
    y_pred_list = []
    
    if is_tflite:
        interpreter = predict_func
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    for i in range(len(X_test)):
        x_in = X_test[i:i+1]
        
        cpu_b = process.cpu_percent(interval=None)
        start_t = time.perf_counter()
        
        if is_keras:
            p = predict_func(x_in, verbose=0)[0][0]
            yp = 1 if p > 0.5 else 0
        elif is_rl:
            yp, _ = predict_func(x_in[0], deterministic=True)
        elif is_tflite:
            interpreter.set_tensor(input_details[0]['index'], x_in)
            interpreter.invoke()
            p = interpreter.get_tensor(output_details[0]['index'])[0][0]
            yp = 1 if p > 0.5 else 0
        else:
            yp = predict_func(x_in)[0]
            
        latency = (time.perf_counter() - start_t) * 1000
        cpu_a = process.cpu_percent(interval=None)
        mem_a = process.memory_info().rss / (1024*1024)
        
        y_pred_list.append(yp)
        logs.append([latency, cpu_a, mem_a])

    tracemalloc.stop()
    
    logs = np.array(logs)
    acc = accuracy_score(y_test, y_pred_list)
    prec = precision_score(y_test, y_pred_list, zero_division=0)
    rec = recall_score(y_test, y_pred_list, zero_division=0)
    f1 = f1_score(y_test, y_pred_list, zero_division=0)
    
    # Tính trung bình các thông số profiling
    lat = logs[:,0].mean()
    cpu = logs[:,1].mean()
    ram = logs[:,2].max()
    
    return acc, prec, rec, f1, lat, cpu, ram

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    results = []
    
    for ds_name, info in DATASETS.items():
        X_tr, y_tr, X_te, y_te = load_and_preprocess(info, ds_name)
        n_features = X_tr.shape[1]
        
        # --- 1. ML: Random Forest & Decision Tree ---
        print(f"[{ds_name}] Training ML Models...")
        rf = RandomForestClassifier(n_estimators=20, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        acc, prec, rec, f1, lat, cpu, ram = profile_predict(rf.predict, X_te, y_te)
        results.append((ds_name, "ML", "Random Forest", acc, prec, rec, f1, lat, cpu, ram))
        
        dt = DecisionTreeClassifier(max_depth=10, random_state=42)
        dt.fit(X_tr, y_tr)
        acc, prec, rec, f1, lat, cpu, ram = profile_predict(dt.predict, X_te, y_te)
        results.append((ds_name, "ML", "Decision Tree", acc, prec, rec, f1, lat, cpu, ram))
        
        # --- 2. DL: DNN & CNN1D ---
        print(f"[{ds_name}] Training DL Models...")
        dnn = keras.Sequential([
            layers.Input(shape=(n_features,)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        dnn.fit(X_tr, y_tr, epochs=3, batch_size=256, verbose=0)
        acc, prec, rec, f1, lat, cpu, ram = profile_predict(dnn.predict, X_te, y_te, is_keras=True)
        results.append((ds_name, "DL", "DNN", acc, prec, rec, f1, lat, cpu, ram))
        
        cnn = keras.Sequential([
            layers.Input(shape=(n_features, 1)),
            layers.Conv1D(16, 3, activation='relu'),
            layers.Flatten(),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        cnn.compile(optimizer='adam', loss='binary_crossentropy')
        cnn.fit(X_tr[..., np.newaxis], y_tr, epochs=3, batch_size=256, verbose=0)
        
        def cnn_predict(x, verbose=0):
            return cnn.predict(x[..., np.newaxis], verbose=verbose)
        
        acc, prec, rec, f1, lat, cpu, ram = profile_predict(cnn_predict, X_te, y_te, is_keras=True)
        results.append((ds_name, "DL", "CNN 1D", acc, prec, rec, f1, lat, cpu, ram))

        # --- 3. RL: PPO & DQN ---
        print(f"[{ds_name}] Training RL Models...")
        env = IdsEnv(X_tr, y_tr)
        
        ppo = PPO("MlpPolicy", env, verbose=0, n_steps=2048)
        ppo.learn(total_timesteps=5000)
        acc, prec, rec, f1, lat, cpu, ram = profile_predict(ppo.predict, X_te, y_te, is_rl=True)
        results.append((ds_name, "RL", "PPO", acc, prec, rec, f1, lat, cpu, ram))
        
        dqn = DQN("MlpPolicy", env, verbose=0)
        dqn.learn(total_timesteps=5000)
        acc, prec, rec, f1, lat, cpu, ram = profile_predict(dqn.predict, X_te, y_te, is_rl=True)
        results.append((ds_name, "RL", "DQN", acc, prec, rec, f1, lat, cpu, ram))

        # --- 4. TinyML: TFLite (Float16 & INT8 Quantized) ---
        print(f"[{ds_name}] Converting to TinyML...")
        converter = tf.lite.TFLiteConverter.from_keras_model(dnn)
        
        # Float16
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_fp16 = converter.convert()
        with open("temp_fp16.tflite", "wb") as f: f.write(tflite_fp16)
        
        interp_fp16 = tf.lite.Interpreter(model_path="temp_fp16.tflite")
        interp_fp16.allocate_tensors()
        acc, prec, rec, f1, lat, cpu, ram = profile_predict(interp_fp16, X_te, y_te, is_tflite=True)
        results.append((ds_name, "TinyML", "TFLite Float16", acc, prec, rec, f1, lat, cpu, ram))
        
        # INT8
        def representative_data_gen():
            for i in range(min(100, len(X_tr))):
                yield [X_tr[i:i+1]]
                
        converter_int8 = tf.lite.TFLiteConverter.from_keras_model(dnn)
        converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]
        converter_int8.representative_dataset = representative_data_gen
        converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter_int8.inference_input_type = tf.float32
        converter_int8.inference_output_type = tf.float32
        
        tflite_int8 = converter_int8.convert()
        with open("temp_int8.tflite", "wb") as f: f.write(tflite_int8)
        
        interp_int8 = tf.lite.Interpreter(model_path="temp_int8.tflite")
        interp_int8.allocate_tensors()
        acc, prec, rec, f1, lat, cpu, ram = profile_predict(interp_int8, X_te, y_te, is_tflite=True)
        results.append((ds_name, "TinyML", "TFLite INT8", acc, prec, rec, f1, lat, cpu, ram))

        gc.collect()

    # Lưu CSV
    df_res = pd.DataFrame(results, columns=["Dataset", "Category", "Algorithm", "Accuracy", "Precision", "Recall", "F1", "Latency_ms", "CPU_pct", "RAM_MB"])
    df_res.to_csv(RESULTS_FILE, index=False)
    print(f"\n[DONE] Lưu kết quả tại {RESULTS_FILE}")

if __name__ == "__main__":
    main()
