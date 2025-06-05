"""
학습된 모델 저장하기
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def retrain_and_save_models():
    """모델 재학습 및 저장"""
    print("🚀 모델 재학습 및 저장 시작")
    print("=" * 50)
    
    # 1. 데이터 로드
    print("📂 데이터 로드...")
    try:
        X_train = pd.read_csv('data/processed/X_train.csv')
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_train = pd.read_csv('data/processed/y_train.csv')['THING_AMT']
        y_test = pd.read_csv('data/processed/y_test.csv')['THING_AMT']
        
        print(f"✅ 학습 데이터: {X_train.shape}")
        print(f"✅ 테스트 데이터: {X_test.shape}")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return
    
    # 2. models 폴더 생성
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"📁 models 폴더 생성: {models_dir.absolute()}")
    
    # 3. Random Forest 학습 및 저장
    print(f"\n🌲 Random Forest 학습...")
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train, y_train)
    rf_path = models_dir / "random_forest_model.pkl"
    joblib.dump(rf_model, rf_path)
    print(f"✅ Random Forest 저장: {rf_path}")
    
    # 4. XGBoost 학습 및 저장
    print(f"\n🚀 XGBoost 학습...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_model.fit(X_train, y_train)
    xgb_path = models_dir / "xgboost_model.pkl"
    joblib.dump(xgb_model, xgb_path)
    print(f"✅ XGBoost 저장: {xgb_path}")
    
    # 5. Linear Regression 학습 및 저장
    print(f"\n📈 Linear Regression 학습...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    lr_path = models_dir / "linear_regression_model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    
    joblib.dump(lr_model, lr_path)
    joblib.dump(scaler, scaler_path)
    print(f"✅ Linear Regression 저장: {lr_path}")
    print(f"✅ Scaler 저장: {scaler_path}")
    
    # 6. 성능 평가 및 저장
    print(f"\n📊 성능 평가...")
    from sklearn.metrics import mean_absolute_percentage_error, r2_score
    
    # Random Forest 평가
    rf_pred = rf_model.predict(X_test)
    rf_mape = mean_absolute_percentage_error(y_test, rf_pred) * 100
    rf_r2 = r2_score(y_test, rf_pred)
    
    # XGBoost 평가
    xgb_pred = xgb_model.predict(X_test)
    xgb_mape = mean_absolute_percentage_error(y_test, xgb_pred) * 100
    xgb_r2 = r2_score(y_test, xgb_pred)
    
    # Linear Regression 평가
    X_test_scaled = scaler.transform(X_test)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_mape = mean_absolute_percentage_error(y_test, lr_pred) * 100
    lr_r2 = r2_score(y_test, lr_pred)
    
    # 결과 저장
    results = {
        'Random Forest': {
            'test_mape': rf_mape,
            'test_r2': rf_r2
        },
        'XGBoost': {
            'test_mape': xgb_mape,
            'test_r2': xgb_r2
        },
        'Linear Regression': {
            'test_mape': lr_mape,
            'test_r2': lr_r2
        }
    }
    
    import json
    results_path = models_dir / "model_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 결과 저장: {results_path}")
    
    # 7. 성능 요약
    print(f"\n🏆 모델 성능 요약:")
    print(f"📊 Random Forest - MAPE: {rf_mape:.2f}%, R²: {rf_r2:.3f}")
    print(f"📊 XGBoost - MAPE: {xgb_mape:.2f}%, R²: {xgb_r2:.3f}")
    print(f"📊 Linear Regression - MAPE: {lr_mape:.2f}%, R²: {lr_r2:.3f}")
    
    # 8. 저장된 파일 확인
    print(f"\n💾 저장된 파일들:")
    for file_path in models_dir.glob("*"):
        print(f"✅ {file_path.name}")
    
    print(f"\n🎉 모델 저장 완료!")
    print(f"📋 다음 단계: streamlit run dashboard_fixed.py")

if __name__ == "__main__":
    retrain_and_save_models()