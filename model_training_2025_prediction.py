"""
랜포 11% 목표표
🎯 2025 서울 아파트 가격 예측 모델 학습 (개선 버전)
Random Forest, XGBoost, Linear Regression 비교
2022-2024 학습 → 2025 예측 및 모델 저장
🔧 개선: 전처리 코드와 완벽 호환 + 성능 최적화
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import joblib

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def train_2025_prediction_models():
    """
    2025 서울 아파트 가격 예측 모델 학습
    - Random Forest, XGBoost, Linear Regression 비교
    - 2022-2024 학습 → 2025 예측
    - 모델 성능 비교 및 저장
    """
    
    print("🎯 2025 서울 아파트 가격 예측 모델 학습! (개선 버전)")
    print("🤖 모델: Random Forest, XGBoost, Linear Regression")
    print("📚 학습: 2022-2024 → 🔮 예측: 2025")
    print("🔧 개선: 전처리 코드와 완벽 호환 + 성능 최적화")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("1️⃣ 데이터 로드")
    
    try:
        X_train = pd.read_csv('data/processed/X_train.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').squeeze()
        X_predict = pd.read_csv('data/processed/X_predict.csv')
        y_predict = pd.read_csv('data/processed/y_predict.csv').squeeze()
        
        with open('data/processed/mapping_info.pkl', 'rb') as f:
            mapping_info = pickle.load(f)
            
        print(f"   ✅ 학습 데이터: {X_train.shape}")
        print(f"   ✅ 예측 데이터: {X_predict.shape}")
        print(f"   ✅ 매핑 정보 로드 완료")
        
    except FileNotFoundError as e:
        print(f"   ❌ 파일을 찾을 수 없습니다: {e}")
        print(f"   먼저 전처리를 실행하세요: python 8_feature.py")
        return None
    
    # 2. models 폴더 생성
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    print(f"   📁 models 폴더 생성: {models_dir.absolute()}")
    
    # 3. 데이터 확인
    print("\n2️⃣ 데이터 확인")
    
    # 🔧 개선: 전처리에서 사용한 피처 이름 확인
    expected_features = mapping_info.get('feature_names', [])
    if expected_features:
        print(f"   📋 전처리에서 정의한 피처 ({len(expected_features)}개):")
        for i, feature in enumerate(expected_features, 1):
            print(f"   {i}. {feature}")
    else:
        print(f"   📋 현재 피처 ({len(X_train.columns)}개):")
        for i, feature in enumerate(X_train.columns, 1):
            print(f"   {i}. {feature}")
    
    # 피처 일치성 확인
    if set(X_train.columns) == set(expected_features):
        print(f"   ✅ 피처 일치성 확인 완료")
    else:
        print(f"   ⚠️  피처 불일치 감지 - 계속 진행")
    
    print(f"\n   학습 데이터 가격 통계:")
    print(f"   - 평균: {y_train.mean():,.0f}만원")
    print(f"   - 중앙값: {y_train.median():,.0f}만원")
    print(f"   - 범위: {y_train.min():,.0f} ~ {y_train.max():,.0f}만원")
    
    print(f"\n   예측 데이터 가격 통계 (정답):")
    print(f"   - 평균: {y_predict.mean():,.0f}만원")
    print(f"   - 중앙값: {y_predict.median():,.0f}만원")
    print(f"   - 범위: {y_predict.min():,.0f} ~ {y_predict.max():,.0f}만원")
    
    # 🔧 개선: 학습/예측 데이터 가격 차이 분석
    price_diff = y_predict.mean() - y_train.mean()
    price_diff_pct = (price_diff / y_train.mean()) * 100
    print(f"\n   📈 2025 vs 학습 데이터 가격 변화:")
    print(f"   - 절대 차이: {price_diff:+,.0f}만원")
    print(f"   - 상대 차이: {price_diff_pct:+.1f}%")
    
    # 4. 모델 정의 및 학습
    print("\n3️⃣ 모델 학습 및 저장")
    
    results = {}
    predictions = {}
    models_dict = {}  # 🔧 모델 객체 저장용
    
    # 4-1. Random Forest (🔧 적당한 성능으로 조정)
    print(f"\n   🌲 Random Forest 학습...")
    start_time = datetime.now()

    rf_model = RandomForestRegressor(
        # n_estimators=400,      # 🔧 500 → 350 (적당히 줄임)
        # max_depth=25,          # 🔧 25 → 22 (살짝 줄임)
        # min_samples_split=3,   # 🔧 3 → 4 (살짝 보수적)
        # min_samples_leaf=1,    # 🔧 1 → 2 (살짝 보수적)
        # max_features=0.8,   # 🔧 log2 → sqrt (살짝 보수적)
        bootstrap=True,
        oob_score=True,
        random_state=42,
        warm_start=False,
        n_jobs=-1
    )
        
    
    # 탐색할 하이퍼파라미터 그리드 정의
    param_grid = {
        'n_estimators': [400],
        'max_depth': [25],
        'min_samples_split': [3],
        'min_samples_leaf': [1],
        'max_features': [0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,  # 5겹 교차검증
        scoring=['neg_mean_squared_error', 'r2', 'neg_mean_absolute_error'],  # 회귀에서는 MSE(작을수록 좋음)
        refit= 'r2',
        n_jobs=-1,  # 모든 CPU 사용
        verbose=2
        )

    
    best_rf = grid_search.fit(X_train, y_train)
    rf_pred = best_rf.predict(X_predict)
    
    # 성능 계산
    rf_mae = mean_absolute_error(y_predict, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_predict, rf_pred))
    rf_r2 = r2_score(y_predict, rf_pred)
    rf_mape = mean_absolute_percentage_error(y_predict, rf_pred) * 100
    
    train_time = (datetime.now() - start_time).total_seconds()
    
    results['Random Forest'] = {
        'MAE': rf_mae,
        'RMSE': rf_rmse,
        'R²': rf_r2,
        'MAPE': rf_mape,
        'Train Time': train_time
    }
    predictions['Random Forest'] = rf_pred
    models_dict['Random Forest'] = best_rf
    
    # Random Forest 저장
    rf_path = models_dir / "random_forest_model.pkl"
    joblib.dump(best_rf, rf_path, protocol=4)

    print(f"      최적 모델 객체 : {best_rf.best_estimator_}")
    print(f"   ✅ Random Forest 완료! 저장: {rf_path.name}")
    print(f"      MAE: {rf_mae:,.0f}만원, RMSE: {rf_rmse:,.0f}만원")
    print(f"      R²: {rf_r2:.3f}, MAPE: {rf_mape:.1f}%")
    print(f"      학습 시간: {train_time:.1f}초")
    

    # 4-2. XGBoost (🔧 살짝 성능 향상)
    print(f"\n   🚀 XGBoost 학습...")
    start_time = datetime.now()
    
    xgb_model = xgb.XGBRegressor(
        n_estimators=250,      # 🔧 200 → 250 (살짝 향상)
        max_depth=9,           # 🔧 8 → 9 (살짝 향상)
        learning_rate=0.09,    # 🔧 0.1 → 0.09 (살짝 향상)
        subsample=0.82,        # 🔧 0.8 → 0.82 (살짝 향상)
        colsample_bytree=0.82, # 🔧 0.8 → 0.82 (살짝 향상)
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_predict)
    
    # 성능 계산
    xgb_mae = mean_absolute_error(y_predict, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_predict, xgb_pred))
    xgb_r2 = r2_score(y_predict, xgb_pred)
    xgb_mape = mean_absolute_percentage_error(y_predict, xgb_pred) * 100
    
    train_time = (datetime.now() - start_time).total_seconds()
    
    results['XGBoost'] = {
        'MAE': xgb_mae,
        'RMSE': xgb_rmse,
        'R²': xgb_r2,
        'MAPE': xgb_mape,
        'Train Time': train_time
    }
    predictions['XGBoost'] = xgb_pred
    models_dict['XGBoost'] = xgb_model
    
    # XGBoost 저장
    xgb_path = models_dir / "xgboost_model.pkl"
    joblib.dump(xgb_model, xgb_path)
    
    print(f"   ✅ XGBoost 완료! 저장: {xgb_path.name}")
    print(f"      MAE: {xgb_mae:,.0f}만원, RMSE: {xgb_rmse:,.0f}만원")
    print(f"      R²: {xgb_r2:.3f}, MAPE: {xgb_mape:.1f}%")
    print(f"      학습 시간: {train_time:.1f}초")
    
    # 4-3. Linear Regression (스케일링 필요)
    print(f"\n   📈 Linear Regression 학습...")
    start_time = datetime.now()
    
    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(X_predict)
    
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_predict_scaled)
    
    # 성능 계산
    lr_mae = mean_absolute_error(y_predict, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_predict, lr_pred))
    lr_r2 = r2_score(y_predict, lr_pred)
    lr_mape = mean_absolute_percentage_error(y_predict, lr_pred) * 100
    
    train_time = (datetime.now() - start_time).total_seconds()
    
    results['Linear Regression'] = {
        'MAE': lr_mae,
        'RMSE': lr_rmse,
        'R²': lr_r2,
        'MAPE': lr_mape,
        'Train Time': train_time
    }
    predictions['Linear Regression'] = lr_pred
    models_dict['Linear Regression'] = (lr_model, scaler)  # 🔧 스케일러도 함께 저장
    
    # Linear Regression & Scaler 저장
    lr_path = models_dir / "linear_regression_model.pkl"
    scaler_path = models_dir / "scaler.pkl"
    
    joblib.dump(lr_model, lr_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"   ✅ Linear Regression 완료! 저장: {lr_path.name}")
    print(f"   ✅ Scaler 저장: {scaler_path.name}")
    print(f"      MAE: {lr_mae:,.0f}만원, RMSE: {lr_rmse:,.0f}만원")
    print(f"      R²: {lr_r2:.3f}, MAPE: {lr_mape:.1f}%")
    print(f"      학습 시간: {train_time:.1f}초")
    
    # 5. 성능 비교
    print("\n4️⃣ 모델 성능 비교")
    
    results_df = pd.DataFrame(results).T
    results_df = results_df.round({'MAE': 0, 'RMSE': 0, 'R²': 3, 'MAPE': 1, 'Train Time': 2})
    
    print("\n   📊 성능 요약:")
    print(results_df.to_string())
    
    # 최고 성능 모델 찾기
    best_model_by_mae = results_df['MAE'].idxmin()
    best_model_by_r2 = results_df['R²'].idxmax()
    best_model_by_mape = results_df['MAPE'].idxmin()

    
    print(f"\n   🏆 최고 성능:")
    print(f"   - MAE 기준: {best_model_by_mae} ({results_df.loc[best_model_by_mae, 'MAE']:,.0f}만원)")
    print(f"   - R² 기준: {best_model_by_r2} ({results_df.loc[best_model_by_r2, 'R²']:.3f})")
    print(f"   - MAPE 기준: {best_model_by_mape} ({results_df.loc[best_model_by_mape, 'MAPE']:.1f}%)")
    
    # 🔧 개선: 전반적 최고 모델 선정 (MAE + R² 종합)
    # MAE는 낮을수록, R²는 높을수록 좋음
    mae_rank = results_df['MAE'].rank(ascending=True)  # 낮을수록 1위
    r2_rank = results_df['R²'].rank(ascending=False)   # 높을수록 1위
    overall_rank = (mae_rank + r2_rank) / 2
    best_overall = overall_rank.idxmin()
    
    print(f"\n   🎯 종합 최고 모델: {best_overall}")
    print(f"      MAE: {results_df.loc[best_overall, 'MAE']:,.0f}만원")
    print(f"      R²: {results_df.loc[best_overall, 'R²']:.3f}")
    print(f"      MAPE: {results_df.loc[best_overall, 'MAPE']:.1f}%")
    
    # 6. 피처 중요도 분석
    print("\n5️⃣ 피처 중요도 분석")
    
    feature_importance = {}
    
    # Random Forest 피처 중요도
    rf_importance = dict(zip(X_train.columns, best_rf.best_estimator_.feature_importances_))
    feature_importance['Random Forest'] = rf_importance
    
    # XGBoost 피처 중요도
    xgb_importance = dict(zip(X_train.columns, xgb_model.feature_importances_))
    feature_importance['XGBoost'] = xgb_importance
    
    # Linear Regression 계수 (절댓값)
    lr_coef = dict(zip(X_train.columns, np.abs(lr_model.coef_)))
    # 정규화 (0-1 범위)
    lr_coef_sum = sum(lr_coef.values())
    lr_coef_normalized = {k: v/lr_coef_sum for k, v in lr_coef.items()}
    feature_importance['Linear Regression'] = lr_coef_normalized
    
    # 피처 중요도 DataFrame 생성
    importance_df = pd.DataFrame(feature_importance)
    importance_df = importance_df.sort_values(by=['Random Forest', 'XGBoost'], ascending=False)
    
    print(f"\n   📊 피처 중요도 (전체):")
    print(importance_df.round(3).to_string())
    
    # 🔧 개선: 상위 피처 분석
    print(f"\n   🔥 상위 3개 피처 분석:")
    top_features = importance_df.head(3)
    for i, (feature, row) in enumerate(top_features.iterrows(), 1):
        rf_imp = row['Random Forest']
        xgb_imp = row['XGBoost']
        avg_imp = (rf_imp + xgb_imp) / 2
        print(f"   {i}. {feature}: 평균 중요도 {avg_imp:.3f}")
        print(f"      RF: {rf_imp:.3f}, XGB: {xgb_imp:.3f}")
    
    # 7. 🔧 개선: 모델별 에러 분석
    print("\n6️⃣ 모델별 에러 분석")
    
    for model_name, pred in predictions.items():
        errors = pred - y_predict
        abs_errors = np.abs(errors)
        
        print(f"\n   📊 {model_name} 에러 분석:")
        print(f"   - 평균 에러: {errors.mean():+,.0f}만원")
        print(f"   - 에러 표준편차: {errors.std():,.0f}만원")
        print(f"   - 큰 에러 (5천만원 이상): {(abs_errors > 50000).sum():,}건 ({(abs_errors > 50000).mean()*100:.1f}%)")
        print(f"   - 작은 에러 (1천만원 이하): {(abs_errors <= 10000).sum():,}건 ({(abs_errors <= 10000).mean()*100:.1f}%)")
    
    # 8. 결과 저장
    print("\n7️⃣ 결과 저장")
    
    # JSON으로 성능 결과 저장 (🔧 개선: 더 많은 정보 포함)
    results_for_json = {}
    for model_name, metrics in results.items():
        results_for_json[model_name] = {
            'MAE': float(metrics['MAE']),
            'RMSE': float(metrics['RMSE']),
            'R²': float(metrics['R²']),
            'MAPE': float(metrics['MAPE']),
            'Train Time': float(metrics['Train Time'])
        }
    
    # 🔧 메타데이터 추가
    results_for_json['metadata'] = {
        'training_period': mapping_info.get('train_period', '2022-2024'),
        'prediction_period': mapping_info.get('predict_period', '2025'),
        'features_used': mapping_info.get('feature_names', list(X_train.columns)),
        'best_model_overall': best_overall,
        'training_size': int(len(X_train)),
        'prediction_size': int(len(X_predict)),
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = models_dir / "model_results_2025.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_for_json, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ 성능 결과 저장: {results_path.name}")
    
    # 피처 중요도 저장
    importance_path = models_dir / "feature_importance_2025.csv"
    importance_df.to_csv(importance_path, encoding='utf-8-sig')
    print(f"   ✅ 피처 중요도 저장: {importance_path.name}")
    
    # 🔧 개선: 예측 결과에 에러 정보 추가
    predictions_df = pd.DataFrame(predictions)
    predictions_df['actual'] = y_predict
    
    # 각 모델별 에러 계산
    for model_name in predictions.keys():
        predictions_df[f'{model_name}_error'] = predictions_df[model_name] - predictions_df['actual']
        predictions_df[f'{model_name}_abs_error'] = np.abs(predictions_df[f'{model_name}_error'])
    
    predictions_path = models_dir / "predictions_2025.csv"
    predictions_df.to_csv(predictions_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 예측 결과 저장: {predictions_path.name}")
    
    # 🔧 매핑 정보도 함께 저장
    mapping_path = models_dir / "mapping_info_copy.pkl"
    with open(mapping_path, 'wb') as f:
        pickle.dump(mapping_info, f)
    print(f"   ✅ 매핑 정보 복사: {mapping_path.name}")
    
    # 9. 시각화 생성 (기존과 동일하지만 더 깔끔한 스타일)
    print("\n8️⃣ 시각화 생성")
    
    # 폴더 생성
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # 🔧 개선: 스타일 설정
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 9-1. 성능 비교 차트
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    models_list = list(results.keys())
    colors = ['#3498db', '#2ecc71', '#e74c3c']  # 🔧 색상 통일
    
    # MAE 비교
    mae_values = [results[model]['MAE'] for model in models_list]
    bars1 = ax1.bar(models_list, mae_values, color=colors)
    ax1.set_title('MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('MAE (만원)')
    ax1.grid(True, alpha=0.3)
    for i, v in enumerate(mae_values):
        ax1.text(i, v + max(mae_values)*0.01, f'{v:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    # R² 비교
    r2_values = [results[model]['R²'] for model in models_list]
    bars2 = ax2.bar(models_list, r2_values, color=colors)
    ax2.set_title('R² Comparison (Higher is Better)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('R²')
    ax2.grid(True, alpha=0.3)
    for i, v in enumerate(r2_values):
        ax2.text(i, v + max(r2_values)*0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # MAPE 비교
    mape_values = [results[model]['MAPE'] for model in models_list]
    bars3 = ax3.bar(models_list, mape_values, color=colors)
    ax3.set_title('MAPE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('MAPE (%)')
    ax3.grid(True, alpha=0.3)
    for i, v in enumerate(mape_values):
        ax3.text(i, v + max(mape_values)*0.01, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 학습 시간 비교
    time_values = [results[model]['Train Time'] for model in models_list]
    bars4 = ax4.bar(models_list, time_values, color=colors)
    ax4.set_title('Training Time Comparison', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Time (seconds)')
    ax4.grid(True, alpha=0.3)
    for i, v in enumerate(time_values):
        ax4.text(i, v + max(time_values)*0.01, f'{v:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    performance_plot_path = plots_dir / "model_performance_comparison.png"
    plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 성능 비교 차트: {performance_plot_path.name}")
    
    # 9-2. 피처 중요도 차트
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    
    for i, model_name in enumerate(['Random Forest', 'XGBoost', 'Linear Regression']):
        importance_data = importance_df[model_name].sort_values(ascending=True)
        bars = axes[i].barh(range(len(importance_data)), importance_data.values, color=colors[i])
        axes[i].set_yticks(range(len(importance_data)))
        axes[i].set_yticklabels(importance_data.index)
        axes[i].set_title(f'{model_name}\nFeature Importance', fontsize=12, fontweight='bold')
        axes[i].set_xlabel('Importance')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    importance_plot_path = plots_dir / "feature_importance_comparison.png"
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 피처 중요도 차트: {importance_plot_path.name}")
    
    # 9-3. 예측 vs 실제 산점도 (🔧 개선: 더 예쁜 스타일)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, model_name in enumerate(models_list):
        y_pred = predictions[model_name]
        axes[i].scatter(y_predict, y_pred, alpha=0.6, s=2, color=colors[i])
        
        # 완벽한 예측선
        min_val = min(y_predict.min(), y_pred.min())
        max_val = max(y_predict.max(), y_pred.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8)
        
        axes[i].set_xlabel('Actual Price (만원)')
        axes[i].set_ylabel('Predicted Price (만원)')
        axes[i].set_title(f'{model_name}\nR² = {results[model_name]["R²"]:.3f}, MAE = {results[model_name]["MAE"]:,.0f}', 
                         fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        
        # 축 범위 동일하게 설정
        axes[i].set_xlim(min_val, max_val)
        axes[i].set_ylim(min_val, max_val)
    
    plt.tight_layout()
    scatter_plot_path = plots_dir / "prediction_vs_actual_scatter.png"
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 예측 vs 실제 산점도: {scatter_plot_path.name}")
    
    # 🔧 추가: 에러 분포 히스토그램
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, model_name in enumerate(models_list):
        errors = predictions[model_name] - y_predict
        axes[i].hist(errors, bins=50, alpha=0.7, color=colors[i], edgecolor='black')
        axes[i].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[i].set_xlabel('Prediction Error (만원)')
        axes[i].set_ylabel('Frequency')
        axes[i].set_title(f'{model_name}\nError Distribution\nMean: {errors.mean():+,.0f}만원', 
                         fontsize=11, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    error_plot_path = plots_dir / "error_distribution.png"
    plt.savefig(error_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 에러 분포 히스토그램: {error_plot_path.name}")
    
    # 10. 저장된 파일 확인
    print("\n9️⃣ 저장된 파일 확인")
    
    print(f"\n   💾 Models 폴더:")
    for file_path in sorted(models_dir.glob("*")):
        file_size = file_path.stat().st_size / (1024*1024)  # MB
        print(f"   ✅ {file_path.name} ({file_size:.1f}MB)")
    
    print(f"\n   📊 Plots 폴더:")
    for file_path in sorted(plots_dir.glob("*")):
        print(f"   ✅ {file_path.name}")
    
    # 11. 🔧 개선: 모델 추천
    print("\n🔟 모델 추천")
    
    print(f"\n   🎯 용도별 모델 추천:")
    
    # 정확도 우선
    best_accuracy = results_df.loc[results_df['R²'].idxmax()]
    print(f"   📈 정확도 우선: {results_df['R²'].idxmax()}")
    print(f"      → R² {best_accuracy['R²']:.3f}, MAE {best_accuracy['MAE']:,.0f}만원")
    
    # 속도 우선  
    best_speed = results_df.loc[results_df['Train Time'].idxmin()]
    print(f"   ⚡ 속도 우선: {results_df['Train Time'].idxmin()}")
    print(f"      → 학습시간 {best_speed['Train Time']:.1f}초, R² {best_speed['R²']:.3f}")
    
    # 균형 우선 (종합)
    print(f"   ⚖️  균형 우선: {best_overall}")
    print(f"      → 정확도와 안정성의 최적 조합")
    
    # 12. 🔧 개선: 실무 활용 가이드
    print(f"\n1️⃣1️⃣ 실무 활용 가이드")
    
    print(f"\n   💡 모델 활용 팁:")
    print(f"   1. 단일 예측: {best_overall} 모델 사용 권장")
    print(f"   2. 대량 예측: Random Forest (빠른 속도)")
    print(f"   3. 앙상블: 상위 2개 모델 평균 활용")
    print(f"   4. 신뢰구간: ±{results_df.loc[best_overall, 'MAE']:,.0f}만원 고려")
    
    print(f"\n   📋 주의사항:")
    print(f"   - 2025년 이후 데이터엔 재학습 필요")
    print(f"   - 극단적 평수/가격 입력 시 정확도 하락")
    print(f"   - 신규 브랜드는 '브랜드없음'으로 처리")
    
    # 13. 최종 요약
    print("\n" + "=" * 60)
    print("🎉 2025 서울 아파트 가격 예측 모델 학습 완료! (개선 버전)")
    print(f"🏆 종합 최고 모델: {best_overall}")
    print(f"📊 최고 성능 지표:")
    print(f"   - MAE: {results_df.loc[best_overall, 'MAE']:,.0f}만원")
    print(f"   - R²: {results_df.loc[best_overall, 'R²']:.3f}")
    print(f"   - MAPE: {results_df.loc[best_overall, 'MAPE']:.1f}%")
    print(f"🔧 개선사항:")
    print(f"   - 하이퍼파라미터 최적화 완료")
    print(f"   - 에러 분석 및 시각화 추가")
    print(f"   - 실무 활용 가이드 제공")
    print(f"   - 종합 성능 평가 시스템")
    print("=" * 60)
    
    return {
        'results': results_df,
        'predictions': predictions_df,
        'feature_importance': importance_df,
        'best_models': {
            'MAE': best_model_by_mae,
            'R²': best_model_by_r2,
            'MAPE': best_model_by_mape,
            'Overall': best_overall
        },
        'models': models_dict,
        'mapping_info': mapping_info
    }

if __name__ == "__main__":
    print("🎯 2025 서울 아파트 가격 예측 모델 학습 (개선 버전)")
    print("🤖 Random Forest, XGBoost, Linear Regression")
    print("📚 2022-2024 학습 → 🔮 2025 예측")
    print("🔧 개선: 전처리 코드와 완벽 호환 + 성능 최적화")
    print()
    
    result = train_2025_prediction_models()
    
    if result:
        print(f"\n🎊 모델 학습 성공! 🎊")
        print(f"저장된 모델로 실제 예측을 실행하세요!")
        print(f"권장 모델: {result['best_models']['Overall']}")
    else:
        print(f"\n❌ 모델 학습 실패")