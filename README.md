# 🏠 Seoul Apartment Price Prediction

서울 아파트 실거래가 데이터를 활용한 가격 예측 ML 프로젝트

## 📊 프로젝트 개요
- **목적**: 2022-2024년 실거래 데이터로 2025년 아파트 가격 예측
- **데이터**: 136,672건 서울시 아파트 실거래가 데이터  - 학습 데이터: 103,251건(2022~2024) / 테스트 데이터: 30,112건(2025)
- **목표**: MAPE < 15%, R² > 0.7

## 🚀 빠른 시작

### 1. 환경 설정
```bash
pip install -r requirements.txt
```

### 2. 데이터 전처리
```bash
python 8_feature.py
```

### 3. 모델 학습 및 저장
```bash
python model_training_2025_prediction.py
```

### 4. 웹 대시보드 실행
```bash
streamlit run streamlit_dashboard_2025.py
```

### 5. 2025 데이터로 모델의 예측정확도 검증 및 리포트 생성
```bash
python all_models_validation_2025.py
```

## 🎯 성과 목표
- [ㅇ] 데이터 수집 완료
- [ㅇ] EDA 및 전처리 
- [ㅇ] 모델 학습 (XGBoost, RF, Linear)
- [ㅇ] 웹 대시보드 구축
- [ㅇ] 성능 검증



