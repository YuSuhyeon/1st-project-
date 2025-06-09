# 🏠 Seoul Apartment Price Prediction

서울 아파트 실거래가 데이터를 활용한 가격 예측 ML 프로젝트

## 📊 프로젝트 개요
- **목적**: 2022-2024년 실거래 데이터로 2025년 아파트 가격 예측
- **데이터**: 136,069건 서울시 아파트 실거래가 데이터
- **목표**: MAPE < 15%, R² > 0.65

## 🚀 빠른 시작 순서
1️⃣ 환경 설정
bashpip install -r requirements.txt
2️⃣ 데이터 수집 (선택사항)
bash# 방법 A: 이미 데이터가 있다면
mkdir -p data/raw
# 20250604_182224_seoul_real_estate.csv 파일을 data/raw/ 폴더에 복사

# 방법 B: 데이터를 새로 수집한다면
# data_collection.ipynb 실행 (Jupyter Notebook에서)

3️⃣ 데이터 전처리
bashpython simple_preprocessing.py
4️⃣ 모델 학습 및 저장
bashpython save_models.py
5️⃣ 결과 파일 생성
bashpython quick_fix_validation.py
6️⃣ 웹 대시보드 실행
bashstreamlit run dashboard_blue.py


## 🚀 순서 설명
pip install -r requirements.txt  \

data_collection.ipynb 실행시켜서 raw 데이터 생성
# data/raw/ 폴더 생성 후 데이터 파일 배치
mkdir -p data/raw
# 20250604_182224_seoul_real_estate.csv 파일을 data/raw/ 폴더에 복사
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

data_collection.ipynb 실행시켜서 raw 데이터 생성
# data/raw/ 폴더 생성 후 데이터 파일 배치
mkdir -p data/raw
# 20250604_182224_seoul_real_estate.csv 파일을 data/raw/ 폴더에 복사


📂 data/raw/ 폴더에 필요한 파일
🎯 필수 파일
data/raw/20250604_182224_seoul_real_estate.csv
이 파일이 반드시 data/raw/ 폴더에 있어야 함!
📋 데이터 파일 준비 방법
1️⃣ 폴더 구조 만들기
bashmkdir -p data/raw
2️⃣ 데이터 파일 배치
프로젝트폴더/
├── data/
│   └── raw/
│       └── 20250604_182224_seoul_real_estate.csv  ← 이 파일이 필요!
├── simple_preprocessing.py
├── save_models.py
└── 기타 파일들...

🔍 데이터 파일 정보
파일명: 20250604_182224_seoul_real_estate.csv
내용: 서울시 아파트 실거래가 데이터 (2022-2025년)
용량: 대략 수십 MB
컬럼: 거래일자, 자치구, 전용면적, 거래금액, 층수, 건축년도 등
```

python simple_preprocessing.py

### 3. 모델 학습 및 저장
```bash
python save_models.py
# python model_training.py는 모델 학습만 진행!! 
```

### 4. 웹 대시보드 실행
```bash
streamlit run dashboard_blue.py
```

## 🎯 성과 목표
- [ ] 데이터 수집 완료
- [ ] EDA 및 전처리 
- [ ] 모델 학습 (XGBoost, RF, Linear)
- [ ] 웹 대시보드 구축
- [ ] 성능 검증


