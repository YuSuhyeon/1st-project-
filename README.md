# 🏠 Seoul Apartment Price Prediction

서울 아파트 실거래가 데이터를 활용한 가격 예측 ML 프로젝트

## 📊 프로젝트 개요
- **목적**: 2022-2024년 실거래 데이터로 2025년 아파트 가격 예측
- **데이터**: 123,641건 서울시 아파트 실거래가 데이터
- **목표**: MAPE < 15%, R² > 0.65

## 🚀 빠른 시작

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 데이터 전처리
```bash
python data_preprocessing.py
```

### 3. 모델 학습 및 저장장
```bash
python save_models.py
# python model_training.py는 모델 학습만 진행!! 
```

### 4. 웹 대시보드 실행
```bash
streamlit streamlit run dashboard_blue.py
```

## 🎯 성과 목표
- [ ] 데이터 수집 완료
- [ ] EDA 및 전처리 
- [ ] 모델 학습 (XGBoost, RF, Linear)
- [ ] 웹 대시보드 구축
- [ ] 성능 검증

## 📈 모델 성능
- **XGBoost**: MAPE %, R² 
- **Random Forest**: MAPE %, R²
- **Linear Regression**: MAPE %, R²

