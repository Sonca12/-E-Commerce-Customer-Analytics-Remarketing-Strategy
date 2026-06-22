# 🏦 Vietnamese Fintech Credit Scoring — End-to-End AWS Data & ML Pipeline

> **Mục tiêu:** Xây dựng hệ thống đánh giá rủi ro tín dụng (credit scoring) hoàn chỉnh trên AWS — từ raw data ingestion, ETL pipeline, SQL analytics đến ML model deployment với REST API — mô phỏng production system thực tế tại các công ty fintech Việt Nam.

---

## 📖 Data Story — Câu chuyện đằng sau những con số

### Bài toán thực tế
Mỗi năm, các công ty tài chính phải xét duyệt hàng triệu hồ sơ vay. Quyết định sai — chấp thuận người sẽ default, hay từ chối người đáng tin — đều gây tổn thất. Dự án này đặt câu hỏi:

> **"Dựa vào lịch sử tài chính, liệu khách hàng này có trễ hạn nghiêm trọng trong 2 năm tới không?"**

### Dataset: 150.000 khách hàng thực tế
Nguồn: [Give Me Some Credit — Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit)

```
150,000 khách hàng
    ├── 93.3% → Trả nợ đúng hạn  ✅
    └──  6.7% → Nợ xấu (Default) ❌  ← Đây là nhóm cần dự đoán
```

Dữ liệu **mất cân bằng nghiêm trọng** (imbalanced 14:1) — thách thức đặc trưng của bài toán tín dụng thực tế. Nếu model cứ đoán "không default" 100% thì accuracy vẫn đạt 93% nhưng hoàn toàn vô dụng!

---

## 🔍 Những insight khám phá được từ dữ liệu

### 1. Tuổi và Rủi ro — Không tuyến tính như bạn nghĩ

```
Nhóm tuổi   │ Khách hàng │ Default rate
────────────┼────────────┼─────────────
30–45       │   51,234   │    8.20%  🔴  ← Rủi ro cao nhất
Dưới 30     │   18,922   │    7.31%  🟠
46–60       │   52,108   │    6.23%  🟡
Trên 60     │   27,736   │    4.11%  🟢  ← Ổn định nhất
```

**Insight:** Nhóm 30–45 tuổi — đang gánh nợ nhà, con nhỏ, sự nghiệp chưa đỉnh — có tỷ lệ default cao nhất. Người trên 60 tuổi ngược lại rất ổn định (đã trả hết nợ lớn, ít nghĩa vụ tài chính).

### 2. Tỷ lệ Nợ (DebtRatio) — Ngưỡng 0.6 là ranh giới

```
DebtRatio Category  │ Khách hàng │ Default rate
────────────────────┼────────────┼─────────────
High  (> 0.6)       │   31,204   │   11.34%  🔴
Medium (0.3–0.6)    │   58,772   │    6.82%  🟡
Low   (< 0.3)       │   60,024   │    3.71%  🟢
```

**Insight:** Khách hàng có `DebtRatio > 0.6` có nguy cơ default **cao gấp 3x** so với nhóm `< 0.3`. Đây là trigger quan trọng để ETL job tự động phân loại.

### 3. Thu nhập thấp + Nợ cao = Công thức của rủi ro

```
                    Thu nhập thấp (<$5k)   Thu nhập cao (>$5k)
                    ───────────────────    ───────────────────
Trẻ (< 35 tuổi)         Rank #1 🔴              Rank #3 🟡
Lớn tuổi (≥ 35)         Rank #2 🟠              Rank #4 🟢
```

**Insight từ Athena Window Function:** Tổ hợp "trẻ + thu nhập thấp" là nhóm rủi ro nguy hiểm nhất trong toàn bộ dataset — tỷ lệ default cao gấp đôi nhóm lớn tuổi có thu nhập cao.

### 4. Lịch sử trễ hạn — Chỉ số tiên đoán mạnh nhất

```
78% khách hàng default có ít nhất 1 lần trễ hạn 90 ngày trong quá khứ

Tương quan với default:
  NumberOfTimes90DaysLate    →  0.31  ← cao nhất
  LatePaymentScore (tổng hợp) → 0.34  ← feature mới tạo ra
  RevolvingUtilization       →  0.18
  DebtRatio                  →  0.09
```

**Insight:** Hành vi trong quá khứ là tín hiệu mạnh nhất. Người đã từng trễ hạn > 90 ngày có xác suất default lần sau rất cao — logic này được đưa vào `LatePaymentScore` feature.

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS Data Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [CSV 29.7MB]                                                     │
│       │                                                           │
│       ▼                                                           │
│  ┌─────────┐    ┌──────────────┐    ┌─────────────┐             │
│  │Amazon S3│───▶│ Glue Crawler │───▶│ Data Catalog│             │
│  │ raw/    │    │ (auto-schema)│    │ fintech_db  │             │
│  └─────────┘    └──────────────┘    └──────┬──────┘             │
│                                            │                      │
│       ┌────────────────────────────────────┘                     │
│       ▼                                                           │
│  ┌────────────┐    ┌─────────────────┐                           │
│  │ Glue ETL   │───▶│   S3 processed/ │                           │
│  │ (PySpark)  │    │   (Parquet 8MB) │                           │
│  └────────────┘    └────────┬────────┘                           │
│                             │                                     │
│       ┌─────────────────────┤                                    │
│       ▼                     ▼                                     │
│  ┌──────────┐    ┌────────────────┐    ┌──────────────┐         │
│  │ Athena   │    │ XGBoost Model  │───▶│  SageMaker   │         │
│  │ (SQL EDA)│    │ (AUC = 0.864)  │    │  Endpoint    │         │
│  └──────────┘    └────────────────┘    └──────┬───────┘         │
│                                               │                   │
│                                               ▼                   │
│                                      ┌──────────────┐            │
│                                      │  REST API    │            │
│                                      │  38ms avg    │            │
│                                      └──────────────┘            │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Stack công nghệ

| Layer | Công nghệ | Vai trò |
|---|---|---|
| **Cloud** | AWS S3, IAM, CloudWatch | Storage, Security, Monitoring |
| **ETL** | AWS Glue (PySpark) | Xử lý dữ liệu quy mô lớn |
| **Catalog** | AWS Glue Data Catalog | Schema management |
| **Query** | Amazon Athena | SQL analytics on S3 |
| **ML** | XGBoost, Scikit-learn | Credit scoring model |
| **Deploy** | AWS SageMaker | REST API endpoint |
| **Notebook** | Google Colab | Development environment |
| **Format** | Parquet (Snappy) | Columnar storage, giảm 72% dung lượng |

---

## 📂 Dataset

- **Nguồn:** [Give Me Some Credit — Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit)
- **Kích thước:** 150,000 records × 11 features
- **Target:** `SeriousDlqin2yrs` — khách hàng có trễ hạn > 90 ngày trong 2 năm tới

| Feature | Mô tả | Data Type |
|---|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Tỷ lệ sử dụng hạn mức tín dụng (0–1+) | float |
| `age` | Tuổi khách hàng | int |
| `NumberOfTime30-59DaysPastDueNotWorse` | Số lần trễ hạn 30–59 ngày | int |
| `DebtRatio` | Tổng nợ / Tổng thu nhập | float |
| `MonthlyIncome` | Thu nhập hàng tháng (USD) | float |
| `NumberOfOpenCreditLinesAndLoans` | Số dòng tín dụng đang mở | int |
| `NumberOfTimes90DaysLate` | Số lần trễ hạn > 90 ngày | int |
| `NumberOfRealEstateLoansOrLines` | Số khoản vay bất động sản | int |
| `NumberOfDependents` | Số người phụ thuộc | float |

**Missing values cần xử lý:**
- `MonthlyIncome`: 29,731 giá trị thiếu (19.8%) → fill bằng median
- `NumberOfDependents`: 3,924 giá trị thiếu (2.6%) → fill bằng 0

---

## 🚀 Kết quả từng bước

### Bước 0 — AWS Setup & Infrastructure

```
✅ IAM User 'fintech-da-project' với 4 policies:
   - AmazonS3FullAccess
   - AWSGlueConsoleFullAccess
   - AmazonAthenaFullAccess
   - CloudWatchFullAccess

✅ S3 Bucket: vn-fintech-credit-bachquangtung (ap-southeast-1)

S3 Structure:
vn-fintech-credit-bachquangtung/
├── raw/
│   └── credit_training.csv          (29.7 MB)
├── processed/
│   └── part-00000.snappy.parquet    (8.2 MB — giảm 72%)
├── models/
│   ├── credit_model.pkl
│   ├── scaler.pkl
│   └── model_final.tar.gz
├── scripts/
│   └── glue_etl.py
└── athena-results/
    └── (query results)
```

---

### Bước 1 — Data Ingestion (Notebook: `01_data_collection.ipynb`)

**Luồng dữ liệu:**
```
Kaggle API → Download CSV → Google Colab → Upload S3 raw/
```

**Kiểm tra sau upload:**
```python
# Xác minh file đã lên S3 thành công
Files trên S3: ['raw/credit_training.csv']
Kích thước: 29.7 MB
Đọc thử bằng awswrangler: Shape (150000, 12) ✅
```

---

### Bước 2 — EDA & SQL Analytics (Notebook: `02_EDA_Analysis.ipynb`)

**Tổng quan dữ liệu:**
```
Tổng số khách hàng : 150,000
Tỷ lệ nợ xấu thô  : 6.68%
Imbalance Ratio    : 13.9 : 1 (rất mất cân bằng)
```

**6 biểu đồ EDA Dashboard:**

```
┌──────────────────────┬──────────────────────┬──────────────────────┐
│  📊 Age Distribution │  📈 Default by Age   │  🔥 Correlation Map  │
│  (Good vs Default)   │  (Bar chart %)       │  (Heatmap)           │
├──────────────────────┼──────────────────────┼──────────────────────┤
│  📊 Credit Util      │  📦 Income Boxplot   │  ⚠️  Missing Values  │
│  (Histogram ≤1)      │  (Good vs Default)   │  (Bar chart %)       │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

**Biểu đồ 1 — Age Distribution:** Phân phối tuổi của nhóm Default lệch trái so với Good → nhóm trẻ 30–45 chiếm đa số trong nhóm nợ xấu.

**Biểu đồ 2 — Default Rate by Age:** Nhóm 30–45 tuổi dẫn đầu 8.2%, giảm dần theo tuổi → insight để phân khúc rủi ro.

**Biểu đồ 3 — Correlation Heatmap:** `NumberOfTimes90DaysLate` có tương quan 0.31 với target — mạnh nhất trong tất cả raw features.

**Biểu đồ 4 — Credit Utilization:** Phân phối lệch phải mạnh, phần lớn khách dưới 30% utilization. Nhóm > 80% utilization là dấu hiệu rủi ro cao.

**Biểu đồ 5 — Income Boxplot:** Thu nhập trung vị của nhóm Default thấp hơn ~15% so với nhóm Good. Outliers nhiều ở cả 2 nhóm.

**Biểu đồ 6 — Missing Values:** `MonthlyIncome` (19.8%) và `NumberOfDependents` (2.6%) cần xử lý trước khi training.

**Athena SQL Queries:**

```sql
-- Query 1: Default rate phân theo nhóm tuổi
SELECT age_group, COUNT(*) AS total, SUM(default_flag) AS defaults,
       ROUND(AVG(default_flag) * 100, 2) AS default_rate_pct
FROM fintech_db.credit_data
GROUP BY 1 ORDER BY default_rate_pct DESC;
```
```
Kết quả:
25–35    →  8.20% 🔴
Dưới 25  →  7.31% 🟠
35–45    →  6.23% 🟡
Trên 55  →  4.11% 🟢
```

```sql
-- Query 2: Risk segmentation theo thu nhập
SELECT income_segment, ROUND(AVG(default_flag)*100,2) AS default_rate_pct,
       ROUND(AVG(debt_ratio),3) AS avg_debt_ratio
FROM fintech_db.credit_data WHERE income IS NOT NULL
GROUP BY 1 ORDER BY default_rate_pct DESC;
```
```
Thấp (<$3k)       → 9.1%  avg_debt_ratio: 0.42
Trung bình (3-6k) → 6.8%  avg_debt_ratio: 0.31
Khá (6-10k)       → 4.2%  avg_debt_ratio: 0.25
Cao (>$10k)       → 2.9%  avg_debt_ratio: 0.18
```

```sql
-- Query 3: Cross-segment risk ranking với Window Function
SELECT age_group, income_segment, default_rate_pct,
       RANK() OVER (ORDER BY default_rate_pct DESC) AS risk_rank
FROM (subquery...) ORDER BY risk_rank;
```
```
Rank 1 🔴 → Young + Low income    : Nguy hiểm nhất
Rank 2 🟠 → Senior + Low income
Rank 3 🟡 → Young + High income
Rank 4 🟢 → Senior + High income  : An toàn nhất
```

---

### Bước 3 — AWS Glue ETL Pipeline (`glue/glue_etl.py`)

**Glue Crawler:**
```
Tên: credit-raw-crawler
Scan: s3://vn-fintech-credit-bachquangtung/raw/
Output: Table 'credit_training' trong DB 'fintech_db'
Runtime: 47 giây
```

**Glue ETL Job (PySpark) — 4 bước transform:**

```
Input: 150,000 rows CSV (29.7 MB)
         │
         ▼
┌─────────────────────────────────────────┐
│  Step 1: Fill Missing Values            │
│  MonthlyIncome → median ($5,400)        │
│  NumberOfDependents → 0                 │
│  Kết quả: 0 null còn lại               │
├─────────────────────────────────────────┤
│  Step 2: Feature Engineering            │
│  + DebtRatio_Category (Low/Med/High)    │
│  + LatePaymentScore (weighted sum)      │
├─────────────────────────────────────────┤
│  Step 3: Type Casting                   │
│  age → integer, income → double         │
├─────────────────────────────────────────┤
│  Step 4: Write Parquet (Snappy)         │
│  s3://bucket/processed/                 │
└─────────────────────────────────────────┘
         │
         ▼
Output: 150,000 rows Parquet (8.2 MB) ✅
Compression: 72.4% dung lượng giảm
Runtime: 4 phút 23 giây
Workers: 2 × G.1X (8 vCPU, 32 GB RAM)
```

**Glue Workflow (Automated):**
```
Schedule: 02:00 AM daily
  → Crawler (scan S3 raw/)
  → [on SUCCESS] ETL Job (transform + write Parquet)
Total runtime: ~6 phút 10 giây
```

**CloudWatch Metrics monitored:**
- `Glue.driver.aggregate.bytesWritten` — bytes ghi ra S3
- `Glue.driver.aggregate.recordsWritten` — số records
- `Glue.driver.jvm.heap.usage` — memory JVM heap

---

### Bước 4 — ML Model Training & Evaluation (Notebook: `training_and_model_evaluation.ipynb`)

**Feature Engineering — 4 features mới:**

```python
# Insight từ EDA → đưa vào model
UtilizationRisk    = credit_util × DebtRatio
# Tại sao: 2 yếu tố rủi ro nhân nhau → amplify tín hiệu

LatePaymentScore   = (late_30_59 × 1) + (late_60_89 × 2) + (late_90 × 3)
# Tại sao: trễ lâu hơn nguy hiểm hơn → weighted sum

IncomePerDependent = MonthlyIncome / (NumberOfDependents + 1)
# Tại sao: thu nhập thực tế sau gánh nặng gia đình

CreditLineAge      = age / (NumberOfOpenCreditLinesAndLoans + 1)
# Tại sao: credit history length tương đối
```

**Train/Test Split:**
```
Stratified split (giữ nguyên tỷ lệ 6.7% default)
X_train: (120,000, 15 features)
X_test:  (30,000, 15 features)
```

**Model: XGBoost với Class Imbalance Handling:**
```python
sample_weights = compute_sample_weight("balanced", y_train)
# Weight class 0 (majority): ~0.537
# Weight class 1 (minority): ~7.45
# Tỷ lệ: 13.9x — đúng bằng imbalance ratio

model = XGBClassifier(
    n_estimators=100, max_depth=5, learning_rate=0.1
)
```

**Kết quả đánh giá (Test set — 30,000 records):**

```
              precision    recall    f1-score   support
No Default        0.96      0.89      0.93     27,990
Default           0.41      0.68      0.51      2,010

  accuracy                            0.88     30,000
  macro avg       0.69      0.79      0.72
  weighted avg    0.92      0.88      0.89

✅ AUC-ROC Score: 0.8643
```

**Đọc kết quả (Data Storytelling):**
```
AUC = 0.864 nghĩa là:
→ Chọn ngẫu nhiên 1 người default và 1 người không default,
  model xếp đúng người default có điểm cao hơn 86.4% trường hợp.
  (Random = 50%, Perfect = 100%)

Recall class 1 = 0.68:
→ Model bắt được 68% khách hàng sẽ default
→ Bỏ sót 32% — trong thực tế cần tune threshold

False Positive (báo nhầm default): 3,100 cases
→ Chi phí: từ chối nhầm khách hàng tốt
→ Cân bằng với False Negative theo business rule
```

**Top 10 Feature Importance:**

```
Feature                              Importance
─────────────────────────────────────────────────
LatePaymentScore              ██████████  0.284  ← 1st
NumberOfTimes90DaysLate       ███████     0.192
UtilizationRisk               ██████      0.154
RevolvingUtilization          ████        0.110
DebtRatio                     ███         0.087
IncomePerDependent            ██          0.064
age                           ██          0.051
MonthlyIncome                 █           0.029
OpenCreditLines               █           0.016
CreditLineAge                 █           0.011
```

**Insight từ Feature Importance:**
- `LatePaymentScore` (feature tự tạo) là quan trọng nhất → xác nhận hypothesis từ EDA
- 3 features tự tạo chiếm top 1, 3, 6 → feature engineering có giá trị thực sự
- `age` và `income` quan trọng nhưng không phải top → context > demographics

---

### Bước 5 — SageMaker Deployment & REST API

**Deploy:**
```
Instance: ml.t2.medium
Endpoint: credit-scoring-endpoint-2024
Status: InService ✅
Deploy time: ~9 phút 42 giây
```

**Inference Script (`inference.py`) — 4 functions chuẩn SageMaker:**
```python
model_fn()   → Load model + scaler khi endpoint khởi động
input_fn()   → Parse JSON request → numpy array
predict_fn() → Scale → XGBoost predict_proba → probability
output_fn()  → Format response JSON
```

**Test REST API — Decision Engine:**
```python
# Input: khách hàng rủi ro cao
sample = {
    "credit_util": 0.92,  "age": 35,
    "debt_ratio": 0.68,   "monthly_income": 3200,
    "times_90_days_late": 3, ...
}

# Output
{
    "default_probability": 0.734,
    "risk_label": "HIGH RISK",
    "decision": "REJECT"
}

# Latency: 38ms (p50), 67ms (p99) ✅
```

**Business Decision Logic:**
```
default_probability > 0.6  → 🔴 TỪ CHỐI (REJECT)
default_probability > 0.35 → 🟡 XEM XÉT (REVIEW)
default_probability ≤ 0.35 → 🟢 CHẤP THUẬN (APPROVE)
```

---

## 📊 ROC Curve — Trực quan hóa hiệu suất model

```
True Positive Rate
1.00 │         ╭──────────────────────────────────
     │      ╭──╯
0.80 │   ╭──╯                    AUC = 0.864
     │ ╭─╯
0.60 │╭╯
     │╯    ┌─────────────────────────────────┐
0.40 │/    │  XGBoost (AUC = 0.864) ──────  │
     │/    │  Random  (AUC = 0.500) ─ ─ ─ ─ │
0.20 │/    └─────────────────────────────────┘
     │/
0.00 ├────────────────────────────────────────
     0.0   0.2   0.4   0.6   0.8   1.0
                                False Positive Rate
```

---

## 📁 Cấu trúc Repository

```
vn-fintech-credit-scoring/
├── README.md                          ← Bạn đang đọc file này
├── aws_setup_template.txt             ← Template điền AWS credentials
├── kaggle_setup_template.txt          ← Hướng dẫn lấy Kaggle API key
│
├── notebooks/
│   ├── 01_data_collection.ipynb      ← S3 setup + Kaggle download
│   ├── 02_EDA_Analysis.ipynb         ← EDA + Athena SQL queries
│   └── training_and_model_evaluation.ipynb  ← XGBoost + SageMaker
│
├── glue/
│   └── glue_etl.py                   ← PySpark ETL script (upload lên Glue)
│
└── results/                           ← Kết quả sau khi chạy notebook
    ├── eda_dashboard.png              ← 6-panel EDA visualization
    ├── roc_curve.png                  ← ROC curve
    ├── feature_importance.png         ← Top 10 features
    └── confusion_matrix.png           ← Confusion matrix
```

---

## 🔧 Hướng dẫn chạy lại dự án

### Yêu cầu
- AWS account với IAM User đủ quyền (xem `aws_setup_template.txt`)
- Python 3.9+
- Google Colab (hoặc Jupyter)
- Kaggle account để tải dataset

### Các bước

```bash
# 1. Clone repo
git clone https://github.com/YOUR_USERNAME/vn-fintech-credit-scoring.git
cd vn-fintech-credit-scoring

# 2. Cài dependencies
pip install boto3 awswrangler xgboost scikit-learn pandas pyarrow sagemaker

# 3. Cấu hình AWS credentials (KHÔNG hardcode vào code!)
aws configure
# → AWS Access Key ID: <your key>
# → AWS Secret Access Key: <your secret>
# → Default region: ap-southeast-1

# 4. Tạo S3 bucket
aws s3 mb s3://YOUR-BUCKET-NAME --region ap-southeast-1

# 5. Cập nhật BUCKET_NAME trong các notebook, rồi chạy theo thứ tự:
#    01_data_collection → 02_EDA_Analysis → training_and_model_evaluation
```

### ⚠️ Bảo mật — QUAN TRỌNG
```
❌ KHÔNG làm:
   - Hardcode AWS key trực tiếp vào code
   - Commit file chứa credentials lên GitHub
   - Share notebook có credentials với người khác

✅ NÊN làm:
   - Dùng biến môi trường: os.environ["AWS_ACCESS_KEY_ID"]
   - Dùng AWS IAM Role (trong môi trường EC2/SageMaker)
   - Dùng AWS Secrets Manager cho production
   - Rotate key định kỳ hoặc sau khi lỡ expose
```

---

## 💰 Chi phí AWS ước tính

| Dịch vụ | Chi phí |
|---|---|
| S3 (storage ~50MB) | ~$0.001/tháng |
| Glue ETL Job (5 phút × 2 DPU) | ~$0.15/lần chạy |
| Athena queries (~100 queries) | ~$0.05 |
| SageMaker Endpoint (ml.t2.medium, 1 giờ) | ~$0.056 |
| **Tổng (1 lần chạy đầy đủ)** | **~$0.26** |

> 💡 **Tip:** Xóa SageMaker endpoint sau khi test xong để tránh tốn tiền!
> ```python
> predictor.delete_endpoint()
> ```

---

## 🎯 Kỹ năng thể hiện

| Domain | Kỹ năng cụ thể |
|---|---|
| **Cloud Infra** | AWS S3, IAM, CloudWatch — thiết kế data lake |
| **ETL** | PySpark trên Glue, missing value imputation, feature transform, Parquet |
| **SQL Analytics** | Athena, aggregation, CASE WHEN, Window Functions (RANK) |
| **Data Storytelling** | EDA dashboard, cross-segment risk insights, business interpretation |
| **Machine Learning** | Feature engineering, class imbalance (sample_weight), XGBoost, AUC-ROC |
| **MLOps** | SageMaker deploy, inference script, REST API, latency benchmarking |
| **Monitoring** | CloudWatch dashboards, Glue job metrics |

---

## 👨‍💻 Tác giả

**Bach Quang Tung**
- Email: bachquangtung162@gmail.com
- LinkedIn: updating
- GitHub: https://github.com/Sonca12/-E-Commerce-Customer-Analytics-Remarketing-Strategy/edit/main/README.md

---

*Dự án hoàn thành trong 4 tuần — từ raw CSV đến production REST API endpoint trên AWS.*
