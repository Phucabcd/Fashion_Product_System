# 👕 Fashion Product Classification & Recommendation System

Hệ thống phân loại sản phẩm thời trang (Fashion-MNIST: 10 lớp nhãn, ảnh grayscale 28x28) được xây dựng bằng Python, TensorFlow/Keras và Scikit-Learn, tích hợp giao diện demo tương tác trên Streamlit.

---

## 📌 Tổng quan Dự án

Dự án triển khai nhiều phương pháp phân loại ảnh thời trang khác nhau từ cơ bản đến nâng cao, nhằm so sánh hiệu năng và ứng dụng thực tế:

1. **Machine Learning Truyền thống (Baseline)**: Mô hình Logistic Regression xử lý dữ liệu ảnh dạng vector phẳng (`784` features).
2. **Deep Learning (Custom CNN)**: Mạng Nơ-ron Cuộn (CNN) thiết kế gồm 2 khối Convolutional (32 & 64 filters), kết hợp Batch Normalization, Max Pooling và Dropout nhằm chống overfit.
3. **Transfer Learning (MobileNetV2)**: Sử dụng kiến trúc MobileNetV2 pre-trained trên ImageNet, qua xử lý resize (96x96) và chuyển đổi từ Grayscale sang RGB, huấn luyện Fine-Tuning 2 giai đoạn.
4. **Đánh giá & Trực quan hóa**:
   - Đánh giá toàn diện các chỉ số: Accuracy, Precision, Recall, Macro-F1 Score, và Confusion Matrix trên tập test riêng biệt.
   - Vẽ đồ thị theo dõi quá trình huấn luyện (Loss & Accuracy qua từng Epoch).
5. **Giao diện Web Tương tác (Streamlit)**: Cho phép người dùng tải ảnh lên, lựa chọn mô hình dự đoán (CNN / Baseline), hiển thị kết quả phân loại, độ tin cậy và biểu đồ phân bố xác suất trực quan.
6. **Mở rộng Gợi ý Sản phẩm (Feature Embedding)**: Thử nghiệm trích xuất vector đặc trưng bằng ResNet50 và so sánh độ tương đồng Cosine (Cosine Similarity) để tìm sản phẩm tương tự.

---

## 📂 Cấu trúc Thư mục

```
Fashion_Recom_Model/
├── app.py                     # Giao diện Web App tương tác (Streamlit)
├── train_model.py             # Huấn luyện mô hình Baseline (Logistic Regression) & Custom CNN
├── train_transfer_model.py    # Huấn luyện mô hình Transfer Learning (MobileNetV2)
├── predict.py                 # Module dự đoán ảnh (Load mô hình & Preprocess)
├── evaluate.py                # Đánh giá hiệu năng mô hình (Accuracy, F1, Precision, Recall, Confusion Matrix)
├── review_history_trained.py  # Trực quan hóa đồ thị Loss / Accuracy quá trình huấn luyện CNN
├── export_sample.py           # Xuất ảnh mẫu ngẫu nhiên từ dataset ra thư mục uploads/
├── build_embedding.py         # (Thử nghiệm) Trích xuất Feature Embedding với ResNet50
├── recommend.py               # (Thử nghiệm) Tìm sản phẩm tương tự dựa trên Cosine Similarity
├── requirements.txt           # Danh sách các thư viện phụ thuộc
├── data/
│   └── archive/               # Thư mục chứa dataset (fashion-mnist_train.csv, fashion-mnist_test.csv)
├── models/                    # Lưu trữ các mô hình đã train (baseline_model.pkl, cnn_model.keras, cnn_history.joblib)
├── models_transfer/           # Lưu trữ mô hình Transfer Learning (mobilenetv2_transfer.keras)
└── uploads/                   # Thư mục lưu trữ ảnh mẫu xuất ra hoặc ảnh tải lên
```

---

## 🔄 Luồng Dữ liệu & Kiến trúc Pipeline

Sơ đồ dưới đây mô tả toàn bộ luồng xử lý dữ liệu và huấn luyện mô hình trong project:

```
DATA (fashion-mnist_train.csv / fashion-mnist_test.csv)
│
│  load_data(): tách label/pixel, reshape (N,28,28), normalize [0,1]
│
┌────────┴────────┐
↓                 ↓
Baseline          Custom CNN
Logistic          2 block Conv2D (32→64)
Regression        + BatchNorm
(flatten          + MaxPooling
784 features)     + Dropout
                  + Dense(256)
                  + Dense(10, softmax)
│                 │
│                 │  train/val split (90/10, stratified)
│                 │  EarlyStopping + ReduceLROnPlateau
│                 │
↓                 ↓
Test Accuracy     Test Accuracy
│                 │
└────────┬────────┘
         ↓
      SO SÁNH
  (evaluate.py)
```

> **Lưu ý**: MobileNetV2 (Transfer Learning) chạy trên pipeline riêng biệt qua `train_transfer_model.py`, với bước preprocess resize 28×28 → 96×96 và chuyển Grayscale → RGB trước khi đưa vào Base Model.

---

## 📊 Dataset & Các Lớp Nhãn

- **Tập dữ liệu**: Fashion-MNIST (Kaggle / Zalando Research)
- **Quy mô**: 70.000 ảnh grayscale (60.000 ảnh train, 10.000 ảnh test), kích thước 28x28 pixel.
- **10 Lớp nhãn (Class Names)**:
  0. `T-shirt/top` (Áo thun/Áo phông)
  1. `Trouser` (Quần dài)
  2. `Pullover` (Áo len chui đầu)
  3. `Dress` (Váy/Đầm)
  4. `Coat` (Áo khoác dáng dài)
  5. `Sandal` (Giày sandal)
  6. `Shirt` (Áo sơ mi)
  7. `Sneaker` (Giày thể thao)
  8. `Bag` (Túi xách)
  9. `Ankle boot` (Ủng/Giày cổ ngắn)

---

## 🛠️ Cài đặt & Hướng dẫn Sử dụng

### 1. Cài đặt Môi trường
Cài đặt các thư viện phụ thuộc bằng `pip`:

```bash
pip install -r requirements.txt
```

*Các thư viện chính bao gồm: `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `streamlit`, `joblib`, `matplotlib`, `pillow`.*

---

### 2. Chuẩn bị Dữ liệu
Đặt 2 file CSV của tập dữ liệu vào đúng đường dẫn:
- `data/archive/fashion-mnist_train.csv`
- `data/archive/fashion-mnist_test.csv`

---

### 3. Huấn luyện Mô hình

#### A. Train Baseline (Logistic Regression) & Custom CNN
Huấn luyện song song mô hình Logistic Regression và mô hình Custom CNN:

```bash
python train_model.py
```
- Các mô hình và lịch sử huấn luyện sẽ được lưu tự động vào thư mục `models/`:
  - `baseline_model.pkl`
  - `cnn_model.keras`
  - `cnn_history.joblib`

#### B. Train Transfer Learning (MobileNetV2)
Huấn luyện mô hình Transfer Learning 2 giai đoạn (Freeze Base -> Fine-Tune):

```bash
python train_transfer_model.py
```
- Mô hình lưu tại `models_transfer/mobilenetv2_transfer.keras`.

---

### 4. Đánh giá Mô hình & Trực quan hóa

- **Tính toán chỉ số chi tiết (Accuracy, F1, Precision, Recall, Confusion Matrix)**:
  ```bash
  python evaluate.py
  ```

- **Vẽ đồ thị Loss / Accuracy qua các Epoch**:
  ```bash
  python review_history_trained.py
  ```

- **Xuất ảnh mẫu để test giao diện Web**:
  ```bash
  python export_sample.py
  ```

---

### 5. Khởi chạy Giao diện Web (Streamlit Demo)

Khởi chạy ứng dụng web tương tác:

```bash
streamlit run app.py
```

Khi ứng dụng mở trên trình duyệt:
1. Lựa chọn loại mô hình dự đoán (`CNN` hoặc `Baseline`).
2. Tải lên file ảnh trang phục (PNG, JPG, JPEG).
3. Xem nhãn dự đoán, phần trăm độ tin cậy và biểu đồ phân bố xác suất từng lớp.

---

## ⚙️ Chi tiết Kiến trúc Mô hình

### 1. Baseline - Logistic Regression
- **Đầu vào**: Vector phẳng 784 phần tử (28x28).
- **Cấu hình**: `max_iter=500`.

### 2. Custom CNN
- **Cấu trúc**:
  - `Input(28, 28, 1)`
  - `Conv2D(32, 3x3)` + `BatchNorm` + `Conv2D(32, 3x3)` + `BatchNorm` + `MaxPooling2D(2x2)` + `Dropout(0.25)`
  - `Conv2D(64, 3x3)` + `BatchNorm` + `Conv2D(64, 3x3)` + `BatchNorm` + `MaxPooling2D(2x2)` + `Dropout(0.25)`
  - `Flatten` -> `Dense(256)` + `BatchNorm` + `Dropout(0.5)` -> `Dense(10, softmax)`
- **Kỹ thuật**: Stratified Train/Val split (90/10), `EarlyStopping` (patience=5), `ReduceLROnPlateau` (patience=3).

### 3. Transfer Learning (MobileNetV2)
- **Preprocess**: Resize 28x28 -> 96x96, biến đổi Grayscale -> RGB (3 channels), chuẩn hóa theo chuẩn MobileNetV2.
- **Quy trình 2 Giai đoạn**:
  - *Giai đoạn 1*: Freeze toàn bộ Base Model MobileNetV2, chỉ huấn luyện classifier đầu ra (10 Epochs, Adam).
  - *Giai đoạn 2*: Unfreeze 20 layers cuối cùng của MobileNetV2 để Fine-Tuning với Learning Rate nhỏ (`1e-5`).
