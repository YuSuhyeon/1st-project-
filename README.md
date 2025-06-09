# 🏠 Seoul Apartment Price Prediction

서울 아파트 실거래가 데이터를 활용한 가격 예측 ML 프로젝트

## 📊 프로젝트 개요
- **목적**: 2022-2024년 실거래 데이터로 2025년 아파트 가격 예측
- **데이터**: 136,069건 서울시 아파트 실거래가 데이터
- **목표**: MAPE < 15%, R² > 0.65

## 🚀 빠른 시작 순서
pip install -r requirements.txt  \

data_collection.ipynb 실행시켜서 raw 데이터 생성
 data/raw/ 폴더 생성 후 데이터 파일 배치
mkdir -p data/raw
20250604_182224_seoul_real_estate.csv 파일을 data/raw/ 폴더에 복사

python simple_preprocessing.py  \
python save_models.py  \
python quick_fix_validation.py  \
streamlit run dashboard_blue.py



### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 데이터 전처리
```bash

데이터 수집 (선택사항)
bash# 방법 A: 이미 데이터가 있다면
mkdir -p data/raw
# 20250604_182224_seoul_real_estate.csv 파일을 data/raw/ 폴더에 복사

# 방법 B: 데이터를 새로 수집한다면
# data_collection.ipynb 실행 (Jupyter Notebook에서)

프로젝트폴더/
├── data/
│   └── raw/
│       └── 20250604_182224_seoul_real_estate.csv  ← 이 파일이 필요!
├── simple_preprocessing.py
├── save_models.py
└── 기타 파일들...

python simple_preprocessing.py 실행

```

### 3. 모델 학습 및 저장
```bash
python save_models.py
# python model_training.py는 모델 학습만 진행!!
```

### 4. 웹 대시보드 실행
```bash
python quick_fix_validation.py
streamlit run dashboard_blue.py
```

## 프로젝트 구조
seoul-apartment-prediction/
├── data/
│   ├── raw/
│   │   └── 20250604_182224_seoul_real_estate.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
├── models/
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── linear_regression_model.pkl
│   └── scaler.pkl
├── results/
│   ├── prediction_comparison.csv
│   └── accuracy_summary.json
├── simple_preprocessing.py      # 데이터 전처리
├── save_models.py              # 모델 학습 및 저장
├── quick_fix_validation.py     # 결과 파일 생성
├── dashboard_blue.py           # 웹 대시보드
├── requirements.txt            # 패키지 의존성
└── README.md


## 🎯 성과 목표
- [ ] 데이터 수집 완료
- [ ] EDA 및 전처리 
- [ ] 모델 학습 (XGBoost, RF, Linear)
- [ ] 웹 대시보드 구축
- [ ] 성능 검증


