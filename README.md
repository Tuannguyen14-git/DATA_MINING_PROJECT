# Data Mining Project – Phân tích đánh giá khách sạn

## 1. Tổng quan dự án

Dự án này áp dụng các **kỹ thuật khai phá dữ liệu (Data Mining)** để phân tích các đánh giá của khách hàng về khách sạn.
Mục tiêu là khai thác thông tin hữu ích từ dữ liệu văn bản (review) và xây dựng các mô hình dự đoán mức **rating của khách sạn**.

Pipeline của dự án tuân theo quy trình khai phá dữ liệu chuẩn:

**Data → Preprocessing → Feature Engineering → Mining / Modeling → Evaluation**

Các kỹ thuật được sử dụng trong dự án:

* Phân tích dữ liệu khám phá (EDA)
* Tiền xử lý văn bản
* Trích xuất đặc trưng TF-IDF
* Khai phá luật kết hợp (Association Rules)
* Phân cụm (Clustering)
* Phân lớp (Classification)
* Học bán giám sát (Semi-supervised learning)
* Hồi quy dự đoán rating


## 2. Dataset

Dataset sử dụng: **Hotel Reviews Dataset**

Dữ liệu chứa các thông tin về đánh giá của khách hàng đối với khách sạn, bao gồm:

* Nội dung review
* Điểm rating
* Thông tin khách sạn
* Địa điểm khách sạn

Vị trí file dữ liệu:

data/raw/7282_1.csv

Các cột quan trọng được sử dụng trong dự án:

| Cột            | Ý nghĩa                     |
| -------------- | --------------------------- |
| reviews.text   | Nội dung đánh giá của khách |
| reviews.rating | Điểm rating của khách       |
| name           | Tên khách sạn               |
| city           | Thành phố                   |

Biến mục tiêu (target):

reviews.rating

Rating nằm trong khoảng **1 đến 5**.


## 3. Cấu trúc thư mục dự án

DATA_MINING_PROJECT
│
├── configs/                  
│   └── params.yaml
│
├── data/                     
│
├── notebooks/                
│   ├── 02_preprocess_feature.ipynb
│   ├── 03_mining_clustering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation_report.ipynb
│
├── src/                      
│   ├── data/                 
│   │   └── loader.py
│   │
│   ├── features/             
│   │   └── builder.py
│   │
│   ├── mining/               
│   │   ├── clustering.py
│   │   └── association.py
│   │
│   └── models/               
│
├── scripts/                  
│   ├── run_pipeline.py       
│   └── run_papermill.py      
│
├── outputs/                  
│   ├── figures/
│   │   └── cluster_visualization.png
│   │
│   ├── tables/
│   │   ├── clustering_results.csv
│   │   └── labels.csv
│   │
│   └── models/              
│       ├── kmeans_model.pkl
│       └── tfidf_vectorizer.pkl
│
├── requirements.txt          
└── README.md                 


## 4. Quy trình khai phá dữ liệu

### 4.1 Phân tích dữ liệu khám phá (EDA)

Notebook:

01_eda.ipynb

Các bước thực hiện:

* Khám phá cấu trúc dataset
* Kiểm tra dữ liệu thiếu
* Phân tích phân bố rating
* Phân tích độ dài review

Mục tiêu:

Hiểu rõ đặc điểm của dữ liệu trước khi xây dựng mô hình.


### 4.2 Tiền xử lý dữ liệu và trích xuất đặc trưng

Notebook:

02_preprocess_feature.ipynb

Các bước xử lý:

* chuyển chữ thường
* loại bỏ URL
* loại bỏ ký tự đặc biệt
* loại bỏ stopwords

Sau đó sử dụng phương pháp:

TF-IDF Vectorization

để chuyển đổi dữ liệu văn bản thành vector số phục vụ cho machine learning.


### 4.3 Khai phá tri thức và phân cụm

Notebook:

03_mining_clustering.ipynb

#### Khai phá luật kết hợp

Thuật toán sử dụng:

Apriori

Các chỉ số:

* support
* confidence
* lift

Ví dụ luật:

(clean, staff) → friendly

Ý nghĩa:

Các review nhắc đến phòng sạch thường cũng nhắc đến nhân viên thân thiện.


#### Phân cụm

Thuật toán sử dụng:

K-Means

Mục tiêu:

Nhóm các review có nội dung tương tự nhau thành các cụm.

Sau đó phân tích các **từ khóa đặc trưng** của mỗi cụm để hiểu chủ đề chính.


### 4.4 Xây dựng mô hình phân lớp

Notebook:

04_modeling.ipynb

Bài toán:

Phân loại sentiment của review.

Các mô hình sử dụng:

* Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)

Các metric đánh giá:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix


### 4.5 Học bán giám sát

Để mô phỏng tình huống **thiếu dữ liệu nhãn**, dự án thực hiện:

* chỉ giữ lại một phần dữ liệu có nhãn
* phần còn lại coi như dữ liệu chưa gán nhãn

Kỹ thuật sử dụng:

Self-training với pseudo-label

Sau đó vẽ **learning curve theo % dữ liệu có nhãn**.


### 4.6 Hồi quy dự đoán rating

Notebook:

05_evaluation_report.ipynb

Mục tiêu:

Dự đoán **rating của khách sạn dựa trên nội dung review**.

Mô hình sử dụng:
Ridge Regression


Các metric đánh giá:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)

Ví dụ kết quả:
MAE ≈ 0.76
RMSE ≈ 0.97

Ngoài ra sử dụng biểu đồ:

**Actual Rating vs Predicted Rating**

để so sánh giá trị dự đoán với giá trị thực tế.


## 5. Cài đặt thư viện

Cài đặt các thư viện cần thiết:

pip install -r requirements.txt

Các thư viện chính:

* pandas
* numpy
* scikit-learn
* matplotlib
* nltk
* mlxtend


## 6. Cách chạy project

Mở và chạy các notebook theo thứ tự:

01_eda.ipynb
02_preprocess_feature.ipynb
03_mining_clustering.ipynb
04_modeling.ipynb
05_evaluation_report.ipynb


Mỗi notebook tương ứng với một bước trong pipeline khai phá dữ liệu.



## 7. Kết quả chính

### Phân lớp

Các mô hình được so sánh:

| Model               | Accuracy |
| ------------------- | -------- |
| Naive Bayes         | ~0.76    |
| Logistic Regression | ~0.77    |
| SVM                 | ~0.76    |

Logistic Regression cho kết quả tốt nhất.



### Hồi quy

Mô hình:
Ridge Regression


Kết quả:
MAE ≈ 0.76
RMSE ≈ 0.97


Mô hình có thể dự đoán rating với sai số trung bình nhỏ hơn 1 điểm.



## 8. Insight từ dữ liệu

Một số insight rút ra từ dữ liệu review:

* phòng sạch thường đi kèm đánh giá tốt về nhân viên
* review tích cực thường nhắc đến dịch vụ và vị trí
* review tiêu cực thường liên quan đến vệ sinh hoặc cơ sở vật chất

Phân cụm review cho thấy các chủ đề chính như:

* chất lượng phòng
* dịch vụ nhân viên
* vị trí khách sạn
* mức độ thoải mái



## 9. Kết luận

Dự án đã thực hiện đầy đủ quy trình **Data Mining cho dữ liệu văn bản**, bao gồm:

* phân tích dữ liệu
* tiền xử lý
* trích xuất đặc trưng
* khai phá luật kết hợp
* phân cụm
* phân lớp
* học bán giám sát
* hồi quy dự đoán rating

Kết quả cho thấy dữ liệu review có thể cung cấp nhiều thông tin hữu ích để đánh giá chất lượng dịch vụ khách sạn.



## 10. Thông tin dự án

Môn học: Data Mining
Năm học: 2025 – 2026
