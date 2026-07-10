# 🔐 So Sánh Mô Hình ML/DL/RL/TinyML cho Phát Hiện Xâm Nhập Mạng

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.x-189A2C?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Đánh giá toàn diện 9 mô hình AI trên 3 bộ dữ liệu phát hiện xâm nhập mạng (IDS)**  
*Metrics: Accuracy · Precision · Recall · F1 · ROC-AUC · RAM · CPU · Inference Time · Model Size*

</div>

---

## 📋 Tổng Quan

Dự án này thực hiện so sánh hệ thống các mô hình **Machine Learning**, **Deep Learning**, **Reinforcement Learning** và **TinyML** trong bài toán phát hiện xâm nhập mạng (Network Intrusion Detection System — NIDS). Mỗi mô hình được đánh giá không chỉ về độ chính xác mà còn về **tài nguyên hệ thống** (RAM, CPU, thời gian inference, kích thước mô hình) — phù hợp cho môi trường **edge computing**.

---

## 🗂️ Cấu Trúc Thư Mục

```
📁 Combined_Datasets/
│
├── 📄 model_comparison.py          # Pipeline chính: train & so sánh 9 mô hình
├── 📄 unified_pipeline.py          # Pipeline tổng hợp
├── 📄 visualize.py                 # Module trực quan hóa
├── 📄 split_datasets.py            # Tách dataset train/test
├── 📄 profiler_pipeline.py         # Đo RAM, CPU, inference time
│
├── 📊 Edge_Inference_Log.csv       # Log inference thực tế — Edge-IIoT
├── 📊 KDD_Inference_Log.csv        # Log inference thực tế — NSL-KDD
├── 📊 TON_Inference_Log.csv        # Log inference thực tế — TON_IoT
│
├── 📁 Edge/                        # Dataset Edge-IIoTset
│   ├── EdgeTrain.csv               # Train set
│   ├── EdgeTest.csv                # Test set
│   └── 📁 Selected dataset for ML and DL/
│       └── ML-EdgeIIoT-dataset.csv # Dataset đầy đủ (157,800 rows) ← dùng trong script
│
├── 📁 KDD/                         # Dataset NSL-KDD
│   ├── KDDTrain+.csv               # Train set (125,973 rows)
│   ├── KDDTest+.csv                # Test set  (22,544 rows)
│   └── convert_to_csv.py           # Convert .txt → .csv
│
├── 📁 TON/                         # Dataset TON_IoT
│   ├── TONTrain.csv                # Train set (12,785 rows)
│   └── TONTest.csv                 # Test set  (3,196 rows)
│
└── 📁 results/                     # Output: 29 biểu đồ + bảng kết quả
    ├── all_results.csv             # Bảng tổng hợp toàn bộ metrics
    ├── 01_metrics_bar_*.png        # Grouped bar: Acc/Prec/Recall/F1
    ├── 02_roc_curves_*.png         # ROC Curves overlay
    ├── 03_heatmap_*.png            # Heatmap metrics × models
    ├── 04_ram_usage_*.png          # RAM Usage per model
    ├── 05_cpu_util_*.png           # CPU Utilization per model
    ├── 06_inference_time_*.png     # Inference Time per sample
    ├── 07_model_size_*.png         # Model Size (KB)
    ├── 08_radar_*.png              # Radar/Spider chart
    ├── 09_confusion_matrices_*.png # Confusion Matrices
    ├── 10_cross_dataset_comparison.png  # So sánh tổng hợp 3 datasets
    └── 11_resource_comparison_all.png   # Resource tổng hợp 3 datasets
```

---

## 🤖 Mô Hình So Sánh

| Category | Mô hình | Thư viện |
|----------|---------|---------|
| **ML** | Random Forest | scikit-learn |
| **ML** | XGBoost | xgboost |
| **ML** | SVM (RBF Kernel) | scikit-learn |
| **ML** | Logistic Regression | scikit-learn |
| **DL** | DNN (MLP 3 lớp) | TensorFlow/Keras |
| **DL** | CNN-1D | TensorFlow/Keras |
| **DL** | LSTM | TensorFlow/Keras |
| **RL** | DQN Classifier | TensorFlow/Keras (custom) |
| **TinyML** | Quantized DNN (INT8) | TFLite |

> **RL approach:** DQN agent học phân loại qua reward signal (+1 đúng / -1 sai), phù hợp cho môi trường tự thích nghi.  
> **TinyML:** DNN được lượng tử hóa INT8 qua TFLite — kích thước chỉ ~12 KB, tối ưu cho thiết bị edge.

---

## 📊 Bộ Dữ Liệu

| Dataset | Loại | Train | Test | Features | Nhãn |
|---------|------|-------|------|---------|------|
| **Edge-IIoTset** | IoT/IIoT network traffic | 126,240 | 31,560 | 62 | Binary (0/1) |
| **NSL-KDD** | Network intrusion (classic) | 125,973 | 22,544 | 41 | Binary (normal/attack) |
| **TON_IoT** | System host metrics | 12,785 | 3,196 | 132 | Binary (0/1) |

### Tiền Xử Lý
- **NSL-KDD:** Label encode categorical features (`protocol_type`, `service`, `flag`); multi-class → binary
- **TON_IoT / Edge-IIoT:** Ép kiểu numeric (`pd.to_numeric`), thay thế `inf`/`NaN` bằng median
- **Tất cả:** `StandardScaler`, giới hạn top-50 features theo variance

---

## 📈 Kết Quả Nổi Bật

### Edge-IIoT Dataset

| Model | Accuracy | F1 | ROC-AUC | RAM (MB) | Inference (ms) | Size (KB) |
|-------|----------|-----|---------|---------|----------------|-----------|
| Random Forest | **99.94%** | **99.96%** | **99.99%** | 481 | 0.062 | 1,810 |
| XGBoost | 99.94% | 99.97% | 99.99% | 793 | **0.002** | 165 |
| DNN | 99.72% | 99.84% | 99.96% | 882 | 0.131 | 240 |
| DQN (RL) | 91.06% | 94.93% | 94.20% | 1,100 | 0.138 | 216 |
| **TinyML** | 91.80% | 95.31% | 86.88% | 1,143 | 0.010 | **12** |

### NSL-KDD Dataset

| Model | Accuracy | F1 | ROC-AUC | Inference (ms) | Size (KB) |
|-------|----------|-----|---------|----------------|-----------|
| XGBoost | **80.97%** | **80.45%** | 96.79% | **0.002** | 289 |
| CNN-1D | 80.12% | 80.37% | 92.13% | 0.149 | 152 |
| Random Forest | 77.31% | 75.75% | **97.29%** | 0.093 | 7,511 |
| **TinyML** | 78.75% | 76.84% | 86.73% | 0.012 | **12** |

### TON_IoT Dataset

| Model | Accuracy | F1 | ROC-AUC | Inference (ms) | Size (KB) |
|-------|----------|-----|---------|----------------|-----------|
| Random Forest | **100%** | **100%** | **100%** | 0.060 | 336 |
| XGBoost | 100% | 100% | 100% | **0.001** | 114 |
| DQN (RL) | 99.59% | 99.45% | 99.95% | 0.121 | 228 |
| **TinyML** | 99.95% | 99.93% | 100% | 0.012 | **12** |

---

## 🖼️ Biểu Đồ Kết Quả

<details>
<summary><b>Xem biểu đồ mẫu</b></summary>

Tất cả biểu đồ được lưu trong thư mục `results/`. 9 loại biểu đồ cho mỗi dataset:

1. **Grouped Bar Chart** — So sánh Accuracy, Precision, Recall, F1
2. **ROC Curves** — Overlay toàn bộ mô hình
3. **Heatmap** — Ma trận metrics × models
4. **RAM Usage** — Bộ nhớ sử dụng khi inference
5. **CPU Utilization** — CPU trong quá trình inference
6. **Inference Time** — Thời gian dự đoán trung bình/sample
7. **Model Size** — Kích thước mô hình trên disk
8. **Radar Chart** — Spider chart tổng hợp đa chiều
9. **Confusion Matrices** — Ma trận nhầm lẫn từng mô hình

</details>

---

## ⚙️ Cài Đặt & Chạy

### Yêu cầu hệ thống
- Python 3.10+
- RAM ≥ 8 GB (khuyến nghị 16 GB)
- GPU (tùy chọn, tăng tốc DL/RL)

### Cài thư viện

```bash
pip install numpy pandas scikit-learn xgboost tensorflow psutil matplotlib seaborn
```

### Chạy so sánh đầy đủ

```bash
# Windows
set PYTHONIOENCODING=utf-8
python model_comparison.py

# Linux/Mac
python model_comparison.py
```

Output sẽ được lưu tự động vào thư mục `results/`.

### Lưu ý
> ⚠️ File `DNN-EdgeIIoT-dataset.csv` (~1.2 GB) và `ML-EdgeIIoT-dataset.csv` (~78 MB) không được đưa lên GitHub do giới hạn kích thước. Tải về từ [Edge-IIoTset Dataset](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot) và đặt vào `Edge/Selected dataset for ML and DL/`.

---

## 🔑 Kết Luận Chính

| Tiêu chí | Mô hình tốt nhất |
|----------|-----------------|
| Accuracy cao nhất | **XGBoost / Random Forest** |
| Tốc độ inference nhanh nhất | **XGBoost** (~0.001–0.002 ms/sample) |
| Kích thước nhỏ nhất | **TinyML (Q-DNN)** (~12 KB) |
| Phù hợp edge device | **TinyML** (nhỏ, nhanh, chấp nhận được) |
| Tổng hợp tốt nhất | **XGBoost** (nhanh + chính xác + nhỏ gọn) |

---

## 📚 Tài Liệu Tham Khảo

- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl.html) — University of New Brunswick
- [Edge-IIoTset Dataset](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)
- [TON_IoT Dataset](https://research.unsw.edu.au/projects/toniot-datasets) — UNSW Sydney
- [TensorFlow Lite Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)

---

## 👤 Tác Giả

**Nguyễn Thọ Việt Dũng**  
📧 vietdung29092007@gmail.com  
🔗 [github.com/VuongLamPTITdev2007](https://github.com/VuongLamPTITdev2007)

---

<div align="center">

*⭐ Nếu dự án này hữu ích, hãy để lại một Star!*

</div>
