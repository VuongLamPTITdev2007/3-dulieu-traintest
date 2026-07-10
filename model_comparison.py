"""
=============================================================================
SO SÁNH MÔ HÌNH ML / DL / RL / TinyML
Trên 3 bộ dữ liệu phát hiện xâm nhập mạng:
  - Edge-IIoT
  - NSL-KDD
  - TON_IoT
=============================================================================
Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
Resource: RAM Usage, CPU Utilization, Inference Time, Model Size
=============================================================================
"""

import os
import sys
import time
import warnings
import traceback
import tempfile
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import psutil
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
from math import pi

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

import xgboost as xgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# ─── Cấu hình toàn cục ─────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# Màu sắc cho từng category
CATEGORY_COLORS = {
    'ML': '#4FC3F7',    # Light Blue
    'DL': '#AB47BC',    # Purple
    'RL': '#FF7043',    # Deep Orange
    'TinyML': '#66BB6A' # Green
}

MODEL_COLORS = {
    'Random Forest': '#29B6F6',
    'XGBoost':       '#0288D1',
    'SVM':           '#01579B',
    'Logistic Reg.': '#4DD0E1',
    'DNN':           '#CE93D8',
    'CNN-1D':        '#AB47BC',
    'LSTM':          '#7B1FA2',
    'DQN (RL)':      '#FF7043',
    'TinyML (Q-DNN)':'#66BB6A',
}

MODEL_CATEGORY = {
    'Random Forest': 'ML', 'XGBoost': 'ML', 'SVM': 'ML', 'Logistic Reg.': 'ML',
    'DNN': 'DL', 'CNN-1D': 'DL', 'LSTM': 'DL',
    'DQN (RL)': 'RL',
    'TinyML (Q-DNN)': 'TinyML',
}


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN 1: LOAD & TIỀN XỬ LÝ DỮ LIỆU
# ═══════════════════════════════════════════════════════════════════════════

def load_edge_dataset():
    """Load Edge-IIoT dataset (full ML dataset with both classes)."""
    print("\n📂 Loading Edge-IIoT dataset...")
    data_path = os.path.join(BASE_DIR, "Edge", "Selected dataset for ML and DL",
                             "ML-EdgeIIoT-dataset.csv")

    df = pd.read_csv(data_path, low_memory=False)

    # Drop non-feature columns
    drop_cols = ['frame.time', 'ip.src_host', 'ip.dst_host', 'arp.dst.proto_ipv4',
                 'arp.src.proto_ipv4', 'Attack_type',
                 'http.file_data', 'http.request.uri.query', 'http.request.method',
                 'http.referer', 'http.request.full_uri', 'http.request.version',
                 'dns.qry.name', 'mqtt.protoname', 'mqtt.topic',
                 'tcp.options', 'tcp.payload', 'tcp.flags',
                 'http.tls_port', 'mqtt.msg']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    label_col = 'Attack_label'

    # Convert all columns to numeric, coerce errors to NaN
    for col in df.columns:
        if col != label_col:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df.drop(columns=[label_col])
    y = df[label_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Labels — Train: {dict(y_train.value_counts())}, Test: {dict(y_test.value_counts())}")
    return X_train, X_test, y_train, y_test


def load_kdd_dataset():
    """Load NSL-KDD dataset (multi-class → binary: normal=0, attack=1)."""
    print("\n📂 Loading NSL-KDD dataset...")
    train_path = os.path.join(BASE_DIR, "KDD", "KDDTrain+.csv")
    test_path  = os.path.join(BASE_DIR, "KDD", "KDDTest+.csv")

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    # Drop difficulty_level
    df_train = df_train.drop(columns=['difficulty_level'], errors='ignore')
    df_test  = df_test.drop(columns=['difficulty_level'], errors='ignore')

    # Binary label: normal=0, tất cả attack=1
    df_train['label'] = (df_train['label'] != 'normal').astype(int)
    df_test['label']  = (df_test['label'] != 'normal').astype(int)

    # Encode categorical columns
    cat_cols = ['protocol_type', 'service', 'flag']
    le_dict = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([df_train[col], df_test[col]], axis=0)
        le.fit(combined)
        df_train[col] = le.transform(df_train[col])
        df_test[col]  = le.transform(df_test[col])
        le_dict[col] = le

    label_col = 'label'
    X_train = df_train.drop(columns=[label_col])
    y_train = df_train[label_col]
    X_test  = df_test.drop(columns=[label_col])
    y_test  = df_test[label_col]

    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Labels — Train: {dict(y_train.value_counts())}, Test: {dict(y_test.value_counts())}")
    return X_train, X_test, y_train, y_test


def load_ton_dataset():
    """Load TON_IoT dataset (binary: label column)."""
    print("\n📂 Loading TON_IoT dataset...")
    train_path = os.path.join(BASE_DIR, "TON", "TONTrain.csv")
    test_path  = os.path.join(BASE_DIR, "TON", "TONTest.csv")

    df_train = pd.read_csv(train_path)
    df_test  = pd.read_csv(test_path)

    # Drop 'type' column (multi-class text) — chỉ dùng 'label' binary
    df_train = df_train.drop(columns=['type'], errors='ignore')
    df_test  = df_test.drop(columns=['type'], errors='ignore')

    label_col = 'label'
    X_train = df_train.drop(columns=[label_col])
    
    # Convert all training columns to numeric
    for col in X_train.columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        
    y_train = df_train[label_col].astype(int)
    
    X_test  = df_test.drop(columns=[label_col])
    
    # Convert all testing columns to numeric
    for col in X_test.columns:
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        
    y_test  = df_test[label_col].astype(int)

    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"   Labels — Train: {dict(y_train.value_counts())}, Test: {dict(y_test.value_counts())}")
    return X_train, X_test, y_train, y_test


def preprocess(X_train, X_test, y_train, y_test, max_features=50):
    """Chuẩn hoá & xử lý NaN/Inf, giới hạn features."""
    # Thay thế inf
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test  = X_test.replace([np.inf, -np.inf], np.nan)

    # Fill NaN bằng median
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_test  = X_test.fillna(medians)

    # Giới hạn features (lấy top variance)
    if X_train.shape[1] > max_features:
        variances = X_train.var().sort_values(ascending=False)
        top_cols = variances.head(max_features).index.tolist()
        X_train = X_train[top_cols]
        X_test  = X_test[top_cols]
        print(f"   ✂️  Reduced to top {max_features} features by variance")

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN 2: ĐO LƯỜNG TÀI NGUYÊN
# ═══════════════════════════════════════════════════════════════════════════

class ResourceMonitor:
    """Đo RAM và CPU trong quá trình inference."""

    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def measure_inference(self, predict_fn, X_test, n_samples=500):
        """Đo inference time, RAM, CPU trên n_samples."""
        n_samples = min(n_samples, len(X_test))
        X_sub = X_test[:n_samples]

        # Warm up
        try:
            predict_fn(X_sub[:5])
        except:
            pass

        # Đo RAM trước
        ram_before = self.process.memory_info().rss / (1024 * 1024)  # MB
        cpu_before = psutil.cpu_percent(interval=None)

        # Inference
        start_time = time.perf_counter()
        _ = predict_fn(X_sub)
        end_time = time.perf_counter()

        # Đo RAM/CPU sau
        ram_after = self.process.memory_info().rss / (1024 * 1024)
        cpu_after = psutil.cpu_percent(interval=0.1)

        total_time_ms = (end_time - start_time) * 1000
        avg_latency = total_time_ms / n_samples

        return {
            'inference_time_ms': avg_latency,
            'ram_usage_mb': max(ram_after, ram_before),
            'ram_delta_mb': ram_after - ram_before,
            'cpu_percent': cpu_after,
        }

    def get_model_size_kb(self, model, model_type='sklearn'):
        """Lấy kích thước model trên disk (KB)."""
        with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as f:
            tmp_path = f.name

        try:
            if model_type == 'sklearn':
                with open(tmp_path, 'wb') as f:
                    pickle.dump(model, f)
            elif model_type == 'keras':
                model.save(tmp_path + '.keras')
                tmp_path = tmp_path + '.keras'
            elif model_type == 'tflite':
                with open(tmp_path, 'wb') as f:
                    f.write(model)  # model là tflite bytes
            size_kb = os.path.getsize(tmp_path) / 1024
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return size_kb


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN 3: HUẤN LUYỆN CÁC MÔ HÌNH
# ═══════════════════════════════════════════════════════════════════════════

def train_ml_models(X_train, y_train, X_test, y_test, monitor):
    """Train 4 ML models: RF, XGBoost, SVM, Logistic Regression."""
    results = {}
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100, max_depth=20, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='logloss', verbosity=0
        ),
        'SVM': SVC(
            kernel='rbf', C=1.0, probability=True, random_state=RANDOM_STATE,
            max_iter=5000
        ),
        'Logistic Reg.': LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    for name, model in models.items():
        print(f"   🔧 Training {name}...")
        start = time.time()

        # Giới hạn SVM samples nếu dataset lớn
        if name == 'SVM' and len(X_train) > 20000:
            idx = np.random.choice(len(X_train), 20000, replace=False)
            model.fit(X_train[idx], y_train[idx])
        else:
            model.fit(X_train, y_train)

        train_time = time.time() - start

        # Predictions
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_test)
            y_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
        else:
            y_prob = y_pred.astype(float)

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)

        # Resource
        resource = monitor.measure_inference(model.predict, X_test)
        resource['model_size_kb'] = monitor.get_model_size_kb(model, 'sklearn')
        resource['train_time_s'] = train_time

        results[name] = {
            'model': model,
            'metrics': metrics,
            'resource': resource,
            'y_pred': y_pred,
            'y_prob': y_prob,
        }
        print(f"      ✅ {name} — Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f} ({train_time:.1f}s)")

    return results


def build_dnn(input_dim):
    """Build a Deep Neural Network."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_cnn1d(input_dim):
    """Build a 1D CNN for tabular data."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_lstm(input_dim):
    """Build LSTM model for sequential pattern detection."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim, 1)),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_dl_models(X_train, y_train, X_test, y_test, monitor):
    """Train 3 DL models: DNN, CNN-1D, LSTM."""
    results = {}
    input_dim = X_train.shape[1]

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
    )

    dl_models = {
        'DNN': (build_dnn(input_dim), X_train, X_test),
        'CNN-1D': (build_cnn1d(input_dim),
                   X_train.reshape(-1, input_dim, 1),
                   X_test.reshape(-1, input_dim, 1)),
        'LSTM': (build_lstm(input_dim),
                 X_train.reshape(-1, input_dim, 1),
                 X_test.reshape(-1, input_dim, 1)),
    }

    for name, (model, X_tr, X_te) in dl_models.items():
        print(f"   🧠 Training {name}...")
        start = time.time()

        model.fit(
            X_tr, y_train,
            epochs=30, batch_size=256,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0
        )
        train_time = time.time() - start

        # Predictions
        y_prob = model.predict(X_te, verbose=0).flatten()
        y_pred = (y_prob >= 0.5).astype(int)

        # Metrics
        metrics = compute_metrics(y_test, y_pred, y_prob)

        # Resource
        def predict_fn(x):
            return model.predict(x, verbose=0).flatten()

        resource = monitor.measure_inference(predict_fn, X_te)
        resource['model_size_kb'] = monitor.get_model_size_kb(model, 'keras')
        resource['train_time_s'] = train_time

        results[name] = {
            'model': model,
            'metrics': metrics,
            'resource': resource,
            'y_pred': y_pred,
            'y_prob': y_prob,
        }
        print(f"      ✅ {name} — Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f} ({train_time:.1f}s)")

    return results


# ─── DQN-based RL Classifier ───────────────────────────────────────────────

class DQNClassifier:
    """
    Deep Q-Network adapted for binary classification.
    State = feature vector, Action = 0 or 1 (class prediction)
    Reward = +1 nếu đúng, -1 nếu sai.
    """

    def __init__(self, state_dim, n_actions=2, lr=0.001):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.gamma = 0.0  # No future rewards in classification
        self.memory = []
        self.batch_size = 128
        self.model = self._build_model(lr)

    def _build_model(self, lr):
        model = keras.Sequential([
            layers.Input(shape=(self.state_dim,)),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.n_actions, activation='linear')
        ])
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.n_actions)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])

    def remember(self, state, action, reward):
        self.memory.append((state, action, reward))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]

        states = np.array([b[0] for b in batch])
        actions = np.array([b[1] for b in batch])
        rewards = np.array([b[2] for b in batch])

        targets = self.model.predict(states, verbose=0)
        for i in range(len(batch)):
            targets[i][actions[i]] = rewards[i]

        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def predict(self, X):
        q_values = self.model.predict(X, verbose=0)
        return np.argmax(q_values, axis=1)

    def predict_proba(self, X):
        q_values = self.model.predict(X, verbose=0)
        # Softmax to get probabilities
        exp_q = np.exp(q_values - np.max(q_values, axis=1, keepdims=True))
        probs = exp_q / np.sum(exp_q, axis=1, keepdims=True)
        return probs[:, 1]


def train_rl_model(X_train, y_train, X_test, y_test, monitor):
    """Train DQN-based RL classifier."""
    results = {}
    print(f"   🎮 Training DQN (RL) Classifier...")
    start = time.time()

    agent = DQNClassifier(state_dim=X_train.shape[1])

    # Training episodes - giới hạn samples cho tốc độ
    n_train = min(len(X_train), 10000)
    n_episodes = 3  # Số epoch

    for ep in range(n_episodes):
        indices = np.random.permutation(n_train)
        correct = 0
        for i in indices:
            state = X_train[i]
            action = agent.act(state)
            reward = 1.0 if action == y_train[i] else -1.0
            agent.remember(state, action, reward)
            if action == y_train[i]:
                correct += 1

            if len(agent.memory) >= agent.batch_size and i % 100 == 0:
                agent.replay()

        acc = correct / n_train
        print(f"      Episode {ep+1}/{n_episodes} — Train Acc: {acc:.4f}, ε: {agent.epsilon:.3f}")

    train_time = time.time() - start

    # Predictions
    y_pred = agent.predict(X_test)
    y_prob = agent.predict_proba(X_test)

    # Metrics
    metrics = compute_metrics(y_test, y_pred, y_prob)

    # Resource
    resource = monitor.measure_inference(agent.predict, X_test)
    resource['model_size_kb'] = monitor.get_model_size_kb(agent.model, 'keras')
    resource['train_time_s'] = train_time

    results['DQN (RL)'] = {
        'model': agent,
        'metrics': metrics,
        'resource': resource,
        'y_pred': y_pred,
        'y_prob': y_prob,
    }
    print(f"      ✅ DQN (RL) — Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f} ({train_time:.1f}s)")
    return results


# ─── TinyML (TFLite Quantized) ─────────────────────────────────────────────

def train_tinyml_model(X_train, y_train, X_test, y_test, monitor):
    """Train a quantized TinyML model via TFLite INT8."""
    results = {}
    print(f"   📱 Training TinyML (Quantized DNN)...")
    start = time.time()

    # Build a small DNN
    input_dim = X_train.shape[1]
    base_model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    base_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
    )
    base_model.fit(
        X_train, y_train,
        epochs=30, batch_size=256,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=0
    )

    # Convert to TFLite with INT8 quantization
    def representative_dataset():
        for i in range(min(200, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    try:
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    except:
        pass  # Fallback nếu INT8 full không hỗ trợ

    tflite_model = converter.convert()
    train_time = time.time() - start

    # TFLite Inference
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype  = input_details[0]['dtype']
    output_dtype = output_details[0]['dtype']

    # Quantization params
    input_scale, input_zero_point = 1.0, 0
    output_scale, output_zero_point = 1.0, 0
    if input_details[0].get('quantization_parameters'):
        qp = input_details[0]['quantization_parameters']
        if len(qp.get('scales', [])) > 0:
            input_scale = qp['scales'][0]
            input_zero_point = qp['zero_points'][0]
    if output_details[0].get('quantization_parameters'):
        qp = output_details[0]['quantization_parameters']
        if len(qp.get('scales', [])) > 0:
            output_scale = qp['scales'][0]
            output_zero_point = qp['zero_points'][0]

    def tflite_predict(X):
        preds = []
        for i in range(len(X)):
            input_data = X[i:i+1].astype(np.float32)
            if input_dtype == np.uint8:
                input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
            elif input_dtype == np.int8:
                input_data = (input_data / input_scale + input_zero_point).astype(np.int8)

            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output = interpreter.get_tensor(output_details[0]['index'])

            if output_dtype in [np.uint8, np.int8]:
                output = (output.astype(np.float32) - output_zero_point) * output_scale

            preds.append(output.flatten()[0])
        return np.array(preds)

    # Predictions (giới hạn cho tốc độ)
    n_test = min(len(X_test), 2000)
    X_test_sub = X_test[:n_test]
    y_test_sub = y_test[:n_test]

    y_prob = tflite_predict(X_test_sub)
    y_prob = np.clip(y_prob, 0, 1)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = compute_metrics(y_test_sub, y_pred, y_prob)

    # Resource
    def predict_fn(x):
        return tflite_predict(x)

    resource = monitor.measure_inference(predict_fn, X_test_sub[:200])
    resource['model_size_kb'] = len(tflite_model) / 1024
    resource['train_time_s'] = train_time

    results['TinyML (Q-DNN)'] = {
        'model': tflite_model,
        'metrics': metrics,
        'resource': resource,
        'y_pred': y_pred,
        'y_prob': y_prob,
        'y_test_actual': y_test_sub,  # Lưu y_test thực tế vì dùng subset
    }
    print(f"      ✅ TinyML (Q-DNN) — Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    print(f"         Model size: {resource['model_size_kb']:.1f} KB ({train_time:.1f}s)")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN 4: TÍNH METRICS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob):
    """Tính toàn bộ classification metrics."""
    return {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
        'roc_auc':   roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN 5: TRỰC QUAN HÓA
# ═══════════════════════════════════════════════════════════════════════════

def setup_plot_style():
    """Thiết lập style chung cho tất cả biểu đồ."""
    plt.rcParams.update({
        'figure.facecolor': '#0D1117',
        'axes.facecolor': '#161B22',
        'axes.edgecolor': '#30363D',
        'axes.labelcolor': '#C9D1D9',
        'text.color': '#C9D1D9',
        'xtick.color': '#8B949E',
        'ytick.color': '#8B949E',
        'grid.color': '#21262D',
        'grid.alpha': 0.6,
        'font.family': 'sans-serif',
        'font.size': 10,
        'axes.titlesize': 14,
        'axes.labelsize': 11,
        'legend.fontsize': 9,
        'legend.facecolor': '#161B22',
        'legend.edgecolor': '#30363D',
    })


def plot_grouped_bar_metrics(all_results, dataset_name):
    """Biểu đồ 1: Grouped Bar Chart — Accuracy, Precision, Recall, F1."""
    fig, ax = plt.subplots(figsize=(16, 8))

    metric_names = ['accuracy', 'precision', 'recall', 'f1']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    model_names = list(all_results.keys())
    n_models = len(model_names)
    n_metrics = len(metric_names)

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    for i, model_name in enumerate(model_names):
        values = [all_results[model_name]['metrics'][m] for m in metric_names]
        color = MODEL_COLORS.get(model_name, '#888888')
        bars = ax.bar(x + i * width - (n_models - 1) * width / 2, values,
                      width, label=model_name, color=color, alpha=0.9,
                      edgecolor='white', linewidth=0.3)
        # Giá trị trên bar
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.008,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6.5,
                    color='#C9D1D9', fontweight='bold')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(f'📊 Classification Metrics Comparison — {dataset_name}',
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper left', ncol=3, fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'01_metrics_bar_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Saved: {path}")


def plot_roc_curves(all_results, y_test, dataset_name):
    """Biểu đồ 2: ROC Curves overlay."""
    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, res in all_results.items():
        y_prob = res['y_prob']
        # Nếu TinyML dùng subset
        yt = res.get('y_test_actual', y_test)
        if len(y_prob) != len(yt):
            yt = yt[:len(y_prob)]

        if len(np.unique(yt)) < 2:
            continue

        fpr, tpr, _ = roc_curve(yt, y_prob)
        auc = roc_auc_score(yt, y_prob)
        color = MODEL_COLORS.get(model_name, '#888888')
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (AUC={auc:.4f})')

    ax.plot([0, 1], [0, 1], 'w--', alpha=0.3, lw=1)
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'📈 ROC Curves — {dataset_name}', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'02_roc_curves_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Saved: {path}")


def plot_heatmap(all_results, dataset_name):
    """Biểu đồ 3: Heatmap — Ma trận metrics × models."""
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    model_names = list(all_results.keys())

    data = []
    for model_name in model_names:
        row = [all_results[model_name]['metrics'][m] for m in metric_names]
        data.append(row)

    df = pd.DataFrame(data, index=model_names, columns=metric_labels)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(df, annot=True, fmt='.4f', cmap='YlOrRd',
                linewidths=1, linecolor='#30363D',
                cbar_kws={'label': 'Score', 'shrink': 0.8},
                ax=ax, vmin=0, vmax=1,
                annot_kws={'size': 11, 'fontweight': 'bold'})
    ax.set_title(f'🔥 Metrics Heatmap — {dataset_name}',
                 fontsize=16, fontweight='bold', pad=15, color='#C9D1D9')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'03_heatmap_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Saved: {path}")


def plot_ram_usage(all_results, dataset_name):
    """Biểu đồ 4: RAM Usage comparison."""
    fig, ax = plt.subplots(figsize=(14, 7))
    model_names = list(all_results.keys())
    ram_values = [all_results[m]['resource']['ram_usage_mb'] for m in model_names]
    colors = [MODEL_COLORS.get(m, '#888888') for m in model_names]

    bars = ax.barh(model_names, ram_values, color=colors, alpha=0.9,
                   edgecolor='white', linewidth=0.5, height=0.6)
    for bar, val in zip(bars, ram_values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{val:.1f} MB', ha='left', va='center',
                fontsize=10, fontweight='bold', color='#C9D1D9')

    ax.set_xlabel('RAM Usage (MB)', fontsize=12)
    ax.set_title(f'💾 RAM Usage During Inference — {dataset_name}',
                 fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'04_ram_usage_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Saved: {path}")


def plot_cpu_utilization(all_results, dataset_name):
    """Biểu đồ 5: CPU Utilization comparison."""
    fig, ax = plt.subplots(figsize=(14, 7))
    model_names = list(all_results.keys())
    cpu_values = [all_results[m]['resource']['cpu_percent'] for m in model_names]
    colors = [MODEL_COLORS.get(m, '#888888') for m in model_names]

    bars = ax.barh(model_names, cpu_values, color=colors, alpha=0.9,
                   edgecolor='white', linewidth=0.5, height=0.6)
    for bar, val in zip(bars, cpu_values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', ha='left', va='center',
                fontsize=10, fontweight='bold', color='#C9D1D9')

    ax.set_xlabel('CPU Utilization (%)', fontsize=12)
    ax.set_title(f'⚡ CPU Utilization During Inference — {dataset_name}',
                 fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'05_cpu_util_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Saved: {path}")


def plot_inference_time(all_results, dataset_name):
    """Biểu đồ 6: Inference Time comparison."""
    fig, ax = plt.subplots(figsize=(14, 7))
    model_names = list(all_results.keys())
    time_values = [all_results[m]['resource']['inference_time_ms'] for m in model_names]
    colors = [MODEL_COLORS.get(m, '#888888') for m in model_names]

    bars = ax.bar(model_names, time_values, color=colors, alpha=0.9,
                  edgecolor='white', linewidth=0.5, width=0.6)
    for bar, val in zip(bars, time_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f} ms', ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#C9D1D9')

    ax.set_ylabel('Avg Inference Time per Sample (ms)', fontsize=11)
    ax.set_title(f'⏱️  Inference Time — {dataset_name}',
                 fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'06_inference_time_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Saved: {path}")


def plot_model_size(all_results, dataset_name):
    """Biểu đồ 7: Model Size comparison (KB)."""
    fig, ax = plt.subplots(figsize=(14, 7))
    model_names = list(all_results.keys())
    size_values = [all_results[m]['resource']['model_size_kb'] for m in model_names]
    colors = [MODEL_COLORS.get(m, '#888888') for m in model_names]

    bars = ax.bar(model_names, size_values, color=colors, alpha=0.9,
                  edgecolor='white', linewidth=0.5, width=0.6)
    for bar, val in zip(bars, size_values):
        label = f'{val:.0f} KB' if val < 1024 else f'{val/1024:.1f} MB'
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                label, ha='center', va='bottom',
                fontsize=9, fontweight='bold', color='#C9D1D9')

    ax.set_ylabel('Model Size (KB)', fontsize=11)
    ax.set_title(f'📦 Model Size Comparison — {dataset_name}',
                 fontsize=16, fontweight='bold', pad=15)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'07_model_size_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Saved: {path}")


def plot_radar_chart(all_results, dataset_name):
    """Biểu đồ 8: Radar/Spider chart tổng hợp."""
    categories = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_facecolor('#161B22')
    fig.set_facecolor('#0D1117')

    for model_name, res in all_results.items():
        values = [res['metrics'][m] for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']]
        values += values[:1]
        color = MODEL_COLORS.get(model_name, '#888888')
        ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, color='#C9D1D9')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8, color='#8B949E')
    ax.set_title(f'🎯 Radar Chart — {dataset_name}',
                 fontsize=16, fontweight='bold', pad=25, color='#C9D1D9')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(color='#30363D', alpha=0.5)
    ax.spines['polar'].set_color('#30363D')

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'08_radar_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Saved: {path}")


def plot_confusion_matrices(all_results, y_test, dataset_name):
    """Biểu đồ 9: Confusion Matrices cho tất cả models."""
    model_names = list(all_results.keys())
    n = len(model_names)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
    fig.suptitle(f'🔢 Confusion Matrices — {dataset_name}',
                 fontsize=18, fontweight='bold', y=1.02, color='#C9D1D9')

    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes_flat = np.array(axes).flatten()

    for i, model_name in enumerate(model_names):
        ax = axes_flat[i]
        y_pred = all_results[model_name]['y_pred']
        yt = all_results[model_name].get('y_test_actual', y_test)
        if len(y_pred) != len(yt):
            yt = yt[:len(y_pred)]

        cm = confusion_matrix(yt, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    ax=ax, cbar=False,
                    linewidths=1, linecolor='#30363D',
                    annot_kws={'size': 14, 'fontweight': 'bold'})
        ax.set_title(model_name, fontsize=12, fontweight='bold', color='#C9D1D9')
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('Actual', fontsize=9)
        color = MODEL_COLORS.get(model_name, '#888888')
        for spine in ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)

    # Ẩn axes thừa
    for j in range(i+1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f'09_confusion_matrices_{dataset_name}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Saved: {path}")


def plot_cross_dataset_comparison(all_dataset_results):
    """Biểu đồ tổng hợp: So sánh mô hình GIỮA các datasets."""
    metric_names = ['accuracy', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'F1-Score', 'ROC-AUC']

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.suptitle('🌐 Cross-Dataset Model Performance Comparison',
                 fontsize=20, fontweight='bold', y=1.02, color='#C9D1D9')

    dataset_names = list(all_dataset_results.keys())

    for idx, (metric, mlabel) in enumerate(zip(metric_names, metric_labels)):
        ax = axes[idx]
        model_names = None

        for ds_name in dataset_names:
            ds_results = all_dataset_results[ds_name]
            if model_names is None:
                model_names = list(ds_results.keys())
            values = [ds_results[m]['metrics'][metric] for m in model_names]

            x = np.arange(len(model_names))
            width = 0.25
            ds_idx = dataset_names.index(ds_name)
            ds_colors = ['#FF6B6B', '#4ECDC4', '#FFD93D']
            ax.bar(x + ds_idx * width - width, values, width,
                   label=ds_name, color=ds_colors[ds_idx], alpha=0.85,
                   edgecolor='white', linewidth=0.3)

        ax.set_xticks(np.arange(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_title(mlabel, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, '10_cross_dataset_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   💾 Saved: {path}")


def plot_resource_comparison_all(all_dataset_results):
    """Biểu đồ tổng hợp: RAM + CPU + Inference Time + Model Size cho tất cả datasets."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.suptitle('🖥️  Resource Usage Comparison Across All Datasets',
                 fontsize=20, fontweight='bold', y=1.02, color='#C9D1D9')

    resource_metrics = [
        ('ram_usage_mb', 'RAM Usage (MB)', '💾'),
        ('cpu_percent', 'CPU Utilization (%)', '⚡'),
        ('inference_time_ms', 'Avg Inference Time (ms)', '⏱️'),
        ('model_size_kb', 'Model Size (KB)', '📦'),
    ]

    dataset_names = list(all_dataset_results.keys())
    ds_colors = ['#FF6B6B', '#4ECDC4', '#FFD93D']

    for idx, (rmetric, rlabel, emoji) in enumerate(resource_metrics):
        ax = axes[idx // 2][idx % 2]
        model_names = None

        for ds_idx, ds_name in enumerate(dataset_names):
            ds_results = all_dataset_results[ds_name]
            if model_names is None:
                model_names = list(ds_results.keys())
            values = [ds_results[m]['resource'][rmetric] for m in model_names]

            x = np.arange(len(model_names))
            width = 0.25
            ax.bar(x + ds_idx * width - width, values, width,
                   label=ds_name, color=ds_colors[ds_idx], alpha=0.85,
                   edgecolor='white', linewidth=0.3)

        ax.set_xticks(np.arange(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{emoji} {rlabel}', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, '11_resource_comparison_all.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   💾 Saved: {path}")


def save_results_csv(all_dataset_results):
    """Lưu toàn bộ kết quả vào CSV."""
    rows = []
    for ds_name, ds_results in all_dataset_results.items():
        for model_name, res in ds_results.items():
            row = {
                'Dataset': ds_name,
                'Model': model_name,
                'Category': MODEL_CATEGORY.get(model_name, 'Unknown'),
                **res['metrics'],
                **{f'resource_{k}': v for k, v in res['resource'].items()},
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    path = os.path.join(RESULTS_DIR, 'all_results.csv')
    df.to_csv(path, index=False)
    print(f"\n   📄 Results CSV saved: {path}")

    # Hiển thị bảng tóm tắt
    print("\n" + "=" * 100)
    print("📊 BẢNG TỔNG HỢP KẾT QUẢ")
    print("=" * 100)
    summary_cols = ['Dataset', 'Model', 'Category', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc',
                    'resource_ram_usage_mb', 'resource_cpu_percent', 'resource_inference_time_ms',
                    'resource_model_size_kb']
    print(df[summary_cols].to_string(index=False))
    return df


# ═══════════════════════════════════════════════════════════════════════════
# PHẦN 6: MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline_for_dataset(dataset_name, load_fn, monitor):
    """Chạy toàn bộ pipeline cho 1 dataset."""
    print(f"\n{'='*80}")
    print(f"🚀 PROCESSING DATASET: {dataset_name}")
    print(f"{'='*80}")

    # Load data
    X_train, X_test, y_train, y_test = load_fn()

    # Preprocess
    print(f"\n⚙️  Preprocessing {dataset_name}...")
    X_train_s, X_test_s, y_train, y_test, scaler = preprocess(
        X_train, X_test, y_train, y_test, max_features=50
    )
    print(f"   Final shape — Train: {X_train_s.shape}, Test: {X_test_s.shape}")

    # Train all models
    all_results = {}

    print(f"\n🔷 Training ML Models...")
    ml_results = train_ml_models(X_train_s, y_train, X_test_s, y_test, monitor)
    all_results.update(ml_results)

    print(f"\n🔮 Training DL Models...")
    dl_results = train_dl_models(X_train_s, y_train, X_test_s, y_test, monitor)
    all_results.update(dl_results)

    print(f"\n🎮 Training RL Model...")
    rl_results = train_rl_model(X_train_s, y_train, X_test_s, y_test, monitor)
    all_results.update(rl_results)

    print(f"\n📱 Training TinyML Model...")
    tinyml_results = train_tinyml_model(X_train_s, y_train, X_test_s, y_test, monitor)
    all_results.update(tinyml_results)

    # Visualize
    print(f"\n🎨 Generating visualizations for {dataset_name}...")
    plot_grouped_bar_metrics(all_results, dataset_name)
    plot_roc_curves(all_results, y_test, dataset_name)
    plot_heatmap(all_results, dataset_name)
    plot_ram_usage(all_results, dataset_name)
    plot_cpu_utilization(all_results, dataset_name)
    plot_inference_time(all_results, dataset_name)
    plot_model_size(all_results, dataset_name)
    plot_radar_chart(all_results, dataset_name)
    plot_confusion_matrices(all_results, y_test, dataset_name)

    return all_results


def main():
    """Main entry point."""
    print("=" * 80)
    print("🔬 SO SÁNH MÔ HÌNH ML / DL / RL / TinyML")
    print("   Trên 3 bộ dữ liệu phát hiện xâm nhập mạng")
    print("=" * 80)

    setup_plot_style()
    monitor = ResourceMonitor()

    # Danh sách datasets
    datasets = {
        'Edge-IIoT':  load_edge_dataset,
        'NSL-KDD':    load_kdd_dataset,
        'TON_IoT':    load_ton_dataset,
    }

    all_dataset_results = {}
    for ds_name, load_fn in datasets.items():
        try:
            results = run_pipeline_for_dataset(ds_name, load_fn, monitor)
            all_dataset_results[ds_name] = results
        except Exception as e:
            print(f"\n❌ Error processing {ds_name}: {e}")
            traceback.print_exc()

    # Cross-dataset comparison charts
    if len(all_dataset_results) > 1:
        print(f"\n🌐 Generating cross-dataset comparison charts...")
        plot_cross_dataset_comparison(all_dataset_results)
        plot_resource_comparison_all(all_dataset_results)

    # Save all results to CSV
    save_results_csv(all_dataset_results)

    print(f"\n{'='*80}")
    print(f"✅ HOÀN TẤT! Tất cả kết quả được lưu tại: {RESULTS_DIR}")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
