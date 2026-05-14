# Heart Disease Classification — Random Forest Pipeline

Dự án xây dựng mô hình phân loại bệnh tim mạch sử dụng **Random Forest** với đầy đủ quy trình từ tiền xử lý đến đánh giá mô hình.

---

## Cấu trúc thư mục

```
├── heart_disease_uci.csv          # Dữ liệu gốc
├── heart_train_cleaned.csv        # Dữ liệu train sau tiền xử lý
├── heart_test_cleaned.csv         # Dữ liệu test sau tiền xử lý
├── eda_summary_statistics.csv     # Thống kê mô tả
├── preprocessor.pkl               # Pipeline tiền xử lý (đã lưu)
├── selector.pkl                   # Bộ chọn feature (đã lưu)
├── rf_model.pkl                   # Mô hình Random Forest tốt nhất (đã lưu)
├── heart_disease_pipeline.py      # Code pipeline chính
└── eda_plots/
    ├── 1_target_distribution.png
    ├── 2_numeric_boxplots.png
    ├── 3_numeric_histograms.png
    ├── 4_correlation_heatmap.png
    ├── 5_target_before_after_smote.png
    └── 6_pca_space_comparison.png
```

## Workflow

```
Load Data → EDA → Data Cleaning → Train/Test Split
    → Preprocessing Pipeline → SMOTE → Feature Selection
        → Hyperparameter Tuning → Predict & Evaluate
```

### 1. Load Data
- Đọc file `heart_disease_uci.csv`
- Nhị phân hóa nhãn `num`: `0` = không bệnh, `1` = có bệnh
- Loại bỏ cột `id`, `dataset`

### 2. EDA
- Thống kê mô tả → xuất ra `eda_summary_statistics.csv`
- Boxplot kiểm tra outlier
- Histogram kiểm tra phân phối (skewness)
- Phân bố biến mục tiêu

### 3. Data Cleaning
- Kiểm tra missing values
- Lọc các feature có tương quan cao (> 0.9) bằng correlation heatmap

### 4. Train / Test Split
- Tỉ lệ: **80% train / 20% test**
- `stratify=y` để đảm bảo phân phối nhãn đồng đều

### 5. Preprocessing Pipeline
| Bước | Numeric | Categorical |
|------|---------|-------------|
| Impute | KNN (k=5) | Most Frequent |
| Scale | RobustScaler | — |
| Encode | — | OneHotEncoder |
| Filter | VarianceThreshold (0.01) | — |

### 6. SMOTE
- Cân bằng dữ liệu mất cân bằng bằng **Synthetic Minority Over-sampling Technique**
- Áp dụng **chỉ trên tập train** để tránh data leakage

### 7. Feature Selection
- Chọn top **k=15** features tốt nhất bằng `SelectKBest` + `mutual_info_classif`

### 8. Hyperparameter Tuning
- `GridSearchCV` với `cv=10`
- Không gian tìm kiếm:

```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
```

### 9. Đánh giá mô hình
- Confusion Matrix
- ROC-AUC Curve
- Accuracy, Sensitivity (Recall), Specificity
- K-Fold Cross Validation (k=10)
 

## Kết quả đầu ra

Sau khi chạy xong, các file sau sẽ được tạo:

| File | Mô tả |
|------|-------|
| `preprocessor.pkl` | Pipeline tiền xử lý |
| `selector.pkl` | Bộ chọn feature |
| `rf_model.pkl` | Mô hình tốt nhất |
| `heart_train_cleaned.csv` | Dữ liệu train đã xử lý |
| `heart_test_cleaned.csv` | Dữ liệu test đã xử lý |
| `eda_plots/` | Toàn bộ biểu đồ EDA và đánh giá |

---

## Ghi chú

- SMOTE chỉ áp dụng trên tập **train**, không áp dụng trên **test**
- PCA 2D chỉ dùng để **visualize** so sánh trước/sau SMOTE, không dùng trong training
- Mô hình được lưu bằng `joblib` để tái sử dụng mà không cần train lại
