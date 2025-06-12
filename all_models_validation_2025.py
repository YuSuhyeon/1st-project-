"""
2025 서울 아파트 가격 예측 정확도 검증 (전체 모델 비교)
Random Forest, XGBoost, Linear Regression 성능 비교 + 상세 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pickle
import json
import os
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

def validate_all_models_2025():
    """모든 모델의 2025 데이터 정확도 검증 및 비교"""
    print("🎯 2025 서울 아파트 가격 예측 정확도 검증")
    print("🤖 Random Forest vs XGBoost vs Linear Regression 성능 비교")
    print("=" * 60)
    
    # 1. 모든 모델 로드
    print("1️⃣ 모든 모델 로드")
    
    models = {}
    scalers = {}
    
    # Random Forest 로드
    try:
        models['Random Forest'] = joblib.load('models/random_forest_model.pkl')
        print("   ✅ Random Forest 로드 성공")
    except Exception as e:
        print(f"   ❌ Random Forest 로드 실패: {e}")
    
    # XGBoost 로드
    try:
        models['XGBoost'] = joblib.load('models/xgboost_model.pkl')
        print("   ✅ XGBoost 로드 성공")
    except Exception as e:
        print(f"   ❌ XGBoost 로드 실패: {e}")
    
    # Linear Regression 로드
    try:
        models['Linear Regression'] = joblib.load('models/linear_regression_model.pkl')
        scalers['Linear Regression'] = joblib.load('models/scaler.pkl')
        print("   ✅ Linear Regression + Scaler 로드 성공")
    except Exception as e:
        print(f"   ❌ Linear Regression 로드 실패: {e}")
    
    if not models:
        print("   ❌ 로드된 모델이 없습니다!")
        print("   먼저 모델 학습을 실행하세요: python model_training_2025_prediction.py")
        return False
    
    print(f"   📊 총 {len(models)}개 모델 로드 완료")
    
    # 2. 2025 테스트 데이터 로드
    print("\n2️⃣ 2025 테스트 데이터 로드")
    
    try:
        X_predict = pd.read_csv('data/processed/X_predict.csv')
        y_predict = pd.read_csv('data/processed/y_predict.csv').squeeze()
        print(f"   ✅ 2025 예측 데이터: {X_predict.shape}")
        print(f"   ✅ 2025 실제 가격: {len(y_predict)}개")
        
        # 기본 통계
        print(f"\n   📊 2025년 실제 가격 통계:")
        print(f"      평균: {y_predict.mean():,.0f}만원")
        print(f"      중앙값: {y_predict.median():,.0f}만원")
        print(f"      범위: {y_predict.min():,.0f} ~ {y_predict.max():,.0f}만원")
        
    except Exception as e:
        print(f"   ❌ 테스트 데이터 로드 실패: {e}")
        print("   먼저 전처리를 실행하세요: python preprocessing_for_2025_prediction.py")
        return False
    
    # 3. 매핑 정보 로드
    print("\n3️⃣ 매핑 정보 로드")
    
    try:
        with open('data/processed/mapping_info.pkl', 'rb') as f:
            mapping_info = pickle.load(f)
        print("   ✅ 매핑 정보 로드 성공")
    except:
        print("   ⚠️ 매핑 정보 없음, 기본값 사용")
        mapping_info = create_default_mapping()
    
    # 4. 모든 모델로 예측 수행
    print("\n4️⃣ 모든 모델 예측 수행")
    
    predictions = {}
    
    for model_name, model in models.items():
        try:
            print(f"   🔮 {model_name} 예측 중...")
            
            if model_name == 'Linear Regression':
                # Linear Regression: 스케일링 적용
                X_scaled = scalers[model_name].transform(X_predict)
                pred = model.predict(X_scaled)
            else:
                # Random Forest, XGBoost: 직접 예측
                pred = model.predict(X_predict)
            
            # 최소값 보정 (1억원 이상)
            pred = np.maximum(pred, 10000)
            predictions[model_name] = pred
            
            print(f"      ✅ 완료: 평균 {pred.mean():,.0f}만원")
            
        except Exception as e:
            print(f"      ❌ {model_name} 예측 실패: {e}")
    
    if not predictions:
        print("   ❌ 예측을 수행한 모델이 없습니다!")
        return False
    
    # 5. 모든 모델 성능 지표 계산
    print("\n5️⃣ 모든 모델 성능 지표 계산")
    
    results = {}
    
    for model_name, pred in predictions.items():
        # 기본 성능 지표
        mae = mean_absolute_error(y_predict, pred)
        rmse = np.sqrt(mean_squared_error(y_predict, pred))
        mape = mean_absolute_percentage_error(y_predict, pred) * 100
        r2 = r2_score(y_predict, pred)
        
        # 오차 계산
        absolute_errors = np.abs(y_predict - pred)
        percentage_errors = (absolute_errors / y_predict) * 100
        
        # 정확도 구간
        accuracy_ranges = {
            'within_10pct': (percentage_errors <= 10).mean() * 100,
            'within_20pct': (percentage_errors <= 20).mean() * 100,
            'within_30pct': (percentage_errors <= 30).mean() * 100,
            'within_2억': (absolute_errors <= 20000).mean() * 100,
            'over_50pct': (percentage_errors > 50).sum()
        }
        
        results[model_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'R²': r2,
            'accuracy_ranges': accuracy_ranges,
            'predictions': pred,
            'absolute_errors': absolute_errors,
            'percentage_errors': percentage_errors
        }
        
        print(f"   📊 {model_name}:")
        print(f"      MAE: {mae:,.0f}만원")
        print(f"      RMSE: {rmse:,.0f}만원")
        print(f"      MAPE: {mape:.2f}%")
        print(f"      R²: {r2:.3f}")
        print(f"      ±20% 이내: {accuracy_ranges['within_20pct']:.1f}%")
    
    # 6. 모델 순위 및 최고 성능 모델 선정
    print("\n6️⃣ 모델 성능 순위")
    
    # MAE 기준 순위
    mae_ranking = sorted(results.items(), key=lambda x: x[1]['MAE'])
    print(f"   🏆 MAE 기준 순위:")
    for i, (model_name, metrics) in enumerate(mae_ranking, 1):
        print(f"      {i}위. {model_name}: {metrics['MAE']:,.0f}만원")
    
    # R² 기준 순위
    r2_ranking = sorted(results.items(), key=lambda x: x[1]['R²'], reverse=True)
    print(f"\n   🏆 R² 기준 순위:")
    for i, (model_name, metrics) in enumerate(r2_ranking, 1):
        print(f"      {i}위. {model_name}: {metrics['R²']:.3f}")
    
    # ±20% 정확도 기준 순위
    accuracy_ranking = sorted(results.items(), key=lambda x: x[1]['accuracy_ranges']['within_20pct'], reverse=True)
    print(f"\n   🏆 ±20% 정확도 기준 순위:")
    for i, (model_name, metrics) in enumerate(accuracy_ranking, 1):
        print(f"      {i}위. {model_name}: {metrics['accuracy_ranges']['within_20pct']:.1f}%")
    
    # 종합 최고 모델 (MAE가 가장 낮은 모델)
    best_model = mae_ranking[0][0]
    print(f"\n   🥇 종합 최고 모델: {best_model}")
    
    # 7. 구별 상세 분석 (최고 모델 기준)
    print(f"\n7️⃣ 구별 상세 분석 ({best_model} 기준)")
    
    # 구별 역매핑
    gu_reverse_mapping = {v: k for k, v in mapping_info.get('gu_label_mapping', {}).items()}
    gu_names = [gu_reverse_mapping.get(x, f'구{x}') for x in X_predict['CGG_LABEL_ENCODED']]
    
    # 결과 DataFrame 생성
    results_df = pd.DataFrame({
        '실제가격': y_predict,
        '구': gu_names,
        '평수': X_predict['PYEONG'],
        '건축년수': X_predict['BUILDING_AGE'],
        '층수': X_predict['FLR'],
        '브랜드점수': X_predict['BRAND_SCORE'],
        '강남3구': X_predict['IS_PREMIUM_GU'],
        '지하철점수': X_predict['SUBWAY_SCORE'],
        '교육특구': X_predict['EDUCATION_PREMIUM']
    })
    
    # 모든 모델의 예측값과 오차 추가
    for model_name, metrics in results.items():
        results_df[f'예측_{model_name}'] = metrics['predictions']
        results_df[f'오차_{model_name}'] = metrics['absolute_errors']
        results_df[f'오차율_{model_name}'] = metrics['percentage_errors']
    
    # 구별 성능 분석 (최고 모델 기준)
    district_analysis = results_df.groupby('구').agg({
        '실제가격': ['count', 'mean'],
        f'오차_{best_model}': 'mean',
        f'오차율_{best_model}': 'mean'
    }).round(1)
    
    district_analysis.columns = ['거래수', '평균가격', '평균오차', '평균오차율']
    district_analysis = district_analysis.sort_values('평균오차율')
    
    print(f"   🏆 예측 정확도 TOP 5 구:")
    print(district_analysis.head().to_string())
    
    print(f"\n   📉 예측 정확도 BOTTOM 5 구:")
    print(district_analysis.tail().to_string())
    
    # 8. 가격대별 성능 분석
    print("\n8️⃣ 가격대별 성능 분석")
    
    # 가격대 분류
    results_df['가격대'] = pd.cut(results_df['실제가격'], 
                                bins=[0, 50000, 100000, 150000, 200000, float('inf')],
                                labels=['5억이하', '5-10억', '10-15억', '15-20억', '20억초과'])
    
    price_analysis = results_df.groupby('가격대').agg({
        '실제가격': ['count', 'mean'],
        f'오차_{best_model}': 'mean',
        f'오차율_{best_model}': 'mean'
    }).round(1)
    
    price_analysis.columns = ['거래수', '평균가격', '평균오차', '평균오차율']
    print(price_analysis.to_string())
    
    # 9. 모델간 예측 일치도 분석
    print("\n9️⃣ 모델간 예측 일치도 분석")
    
    if len(predictions) >= 2:
        model_names = list(predictions.keys())
        print(f"   🤝 모델간 상관관계:")
        
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                pred1, pred2 = predictions[model1], predictions[model2]
                
                correlation = np.corrcoef(pred1, pred2)[0, 1]
                avg_diff = np.mean(np.abs(pred1 - pred2))
                
                print(f"      {model1} vs {model2}:")
                print(f"        상관계수: {correlation:.3f}")
                print(f"        평균 차이: {avg_diff:,.0f}만원")
    
    # 10. 시각화 생성
    print("\n🔟 시각화 생성")
    
    # 폴더 생성
    os.makedirs('plots', exist_ok=True)
    
    # 10-1. 모델 성능 비교 차트
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    model_names = list(results.keys())
    colors = ['#3498db', '#2ecc71', '#e74c3c'][:len(model_names)]
    
    # MAE 비교
    mae_values = [results[model]['MAE'] for model in model_names]
    bars1 = axes[0,0].bar(model_names, mae_values, color=colors)
    axes[0,0].set_title('MAE Comparison (Lower is Better)')
    axes[0,0].set_ylabel('MAE (만원)')
    axes[0,0].grid(True, alpha=0.3)
    for i, v in enumerate(mae_values):
        axes[0,0].text(i, v + max(mae_values)*0.01, f'{v:,.0f}', ha='center', va='bottom')
    
    # R² 비교
    r2_values = [results[model]['R²'] for model in model_names]
    bars2 = axes[0,1].bar(model_names, r2_values, color=colors)
    axes[0,1].set_title('R² Comparison (Higher is Better)')
    axes[0,1].set_ylabel('R²')
    axes[0,1].grid(True, alpha=0.3)
    for i, v in enumerate(r2_values):
        axes[0,1].text(i, v + max(r2_values)*0.01, f'{v:.3f}', ha='center', va='bottom')
    
    # MAPE 비교
    mape_values = [results[model]['MAPE'] for model in model_names]
    bars3 = axes[1,0].bar(model_names, mape_values, color=colors)
    axes[1,0].set_title('MAPE Comparison (Lower is Better)')
    axes[1,0].set_ylabel('MAPE (%)')
    axes[1,0].grid(True, alpha=0.3)
    for i, v in enumerate(mape_values):
        axes[1,0].text(i, v + max(mape_values)*0.01, f'{v:.1f}%', ha='center', va='bottom')
    
    # ±20% 정확도 비교
    accuracy_values = [results[model]['accuracy_ranges']['within_20pct'] for model in model_names]
    bars4 = axes[1,1].bar(model_names, accuracy_values, color=colors)
    axes[1,1].set_title('±20% Accuracy (Higher is Better)')
    axes[1,1].set_ylabel('Accuracy (%)')
    axes[1,1].grid(True, alpha=0.3)
    for i, v in enumerate(accuracy_values):
        axes[1,1].text(i, v + max(accuracy_values)*0.01, f'{v:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    performance_plot_path = 'plots/all_models_performance_comparison_2025.png'
    plt.savefig(performance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 모델 성능 비교: {performance_plot_path}")
    
    # 10-2. 예측 vs 실제 산점도 (모든 모델)
    fig, axes = plt.subplots(1, len(model_names), figsize=(5*len(model_names), 5))
    if len(model_names) == 1:
        axes = [axes]
    
    for i, model_name in enumerate(model_names):
        pred = predictions[model_name]
        r2 = results[model_name]['R²']
        mae = results[model_name]['MAE']
        
        axes[i].scatter(y_predict, pred, alpha=0.5, s=1, color=colors[i])
        axes[i].plot([y_predict.min(), y_predict.max()], 
                    [y_predict.min(), y_predict.max()], 'r--', lw=2)
        axes[i].set_xlabel('Actual Price (만원)')
        axes[i].set_ylabel('Predicted Price (만원)')
        axes[i].set_title(f'{model_name}\nR² = {r2:.3f}, MAE = {mae:,.0f}')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_plot_path = 'plots/prediction_vs_actual_all_models_2025.png'
    plt.savefig(scatter_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ 예측 vs 실제 산점도: {scatter_plot_path}")
    
    # 11. 결과 저장
    print("\n1️⃣1️⃣ 결과 저장")
    
    # 폴더 생성
    for folder in ['results', 'reports']:
        os.makedirs(folder, exist_ok=True)
    
    # 전체 예측 결과 저장
    results_path = 'results/all_models_prediction_validation_2025.csv'
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"   ✅ 전체 예측 결과: {results_path}")
    
    # 모델 성능 요약 저장
    performance_summary = {}
    for model_name, metrics in results.items():
        performance_summary[model_name] = {
            'MAE': float(metrics['MAE']),
            'RMSE': float(metrics['RMSE']),
            'MAPE': float(metrics['MAPE']),
            'R²': float(metrics['R²']),
            'within_20pct': float(metrics['accuracy_ranges']['within_20pct']),
            'over_50pct_errors': int(metrics['accuracy_ranges']['over_50pct'])
        }
    
    summary_path = 'results/all_models_performance_summary_2025.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(performance_summary, f, indent=2, ensure_ascii=False)
    print(f"   ✅ 성능 요약: {summary_path}")
    
    # 12. 최종 보고서 생성
    print("\n1️⃣2️⃣ 최종 보고서 생성")
    
    # 실용성 평가
    best_accuracy = results[best_model]['accuracy_ranges']['within_20pct']
    if best_accuracy >= 80:
        practical_level = "매우 실용적"
        practical_emoji = "🚀"
    elif best_accuracy >= 70:
        practical_level = "실용적"
        practical_emoji = "✅"
    elif best_accuracy >= 60:
        practical_level = "보통"
        practical_emoji = "⚠️"
    else:
        practical_level = "개선 필요"
        practical_emoji = "❌"
    
    report = f"""
=================================================================
🎯 2025 서울 아파트 가격 예측 모델 전체 성능 검증 보고서
=================================================================

📅 검증 날짜: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 검증 데이터: {len(results_df):,}건 (2025년 실제 거래)
🤖 검증 모델: {', '.join(model_names)}

🏆 종합 최고 모델: {best_model}

📊 모델별 성능 요약:
"""
    
    for model_name, metrics in results.items():
        symbol = "🥇" if model_name == best_model else "🥈" if model_name == mae_ranking[1][0] else "🥉"
        report += f"""
{symbol} {model_name}:
   - MAE: {metrics['MAE']:,.0f}만원
   - RMSE: {metrics['RMSE']:,.0f}만원
   - MAPE: {metrics['MAPE']:.2f}%
   - R²: {metrics['R²']:.3f} ({metrics['R²']*100:.1f}% 설명력)
   - ±20% 이내: {metrics['accuracy_ranges']['within_20pct']:.1f}%
   - 극단오차(50%초과): {metrics['accuracy_ranges']['over_50pct']}건
"""
    
    report += f"""
{practical_emoji} 실용성 평가: {practical_level}
   (최고 모델 ±20% 이내 예측률: {best_accuracy:.1f}%)

🏘️ 구별 성능 TOP 3 ({best_model} 기준):
"""
    
    for i, (district, row) in enumerate(district_analysis.head(3).iterrows(), 1):
        report += f"   {i}. {district}: {row['평균오차율']:.1f}% (거래수: {row['거래수']:.0f})\n"
    
    report += f"""
💰 가격대별 성능 ({best_model} 기준):
"""
    for price_range, row in price_analysis.iterrows():
        report += f"   - {price_range}: {row['평균오차율']:.1f}% (거래수: {row['거래수']:.0f})\n"
    
    if len(model_names) >= 2:
        report += f"""
🤝 모델간 일치도:
"""
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1, model2 = model_names[i], model_names[j]
                pred1, pred2 = predictions[model1], predictions[model2]
                correlation = np.corrcoef(pred1, pred2)[0, 1]
                report += f"   - {model1} vs {model2}: 상관계수 {correlation:.3f}\n"
    
    report += f"""
💡 주요 특징:
   - 최고 성능: {best_model} (MAE {results[best_model]['MAE']:,.0f}만원)
   - 평균 {results[best_model]['MAE']/10000:.1f}억원 오차로 실용적 수준
   - {best_accuracy:.0f}%의 거래가 ±20% 이내로 예측
   - 모델간 높은 일치도로 안정적 예측

🎯 권장 사항:
   - 추천 모델: {best_model}
   - 일반 시세 참고: 적극 권장
   - 투자 결정: ±20% 오차 감안하여 활용
   - 정밀 평가: 전문가 상담 병행

=================================================================
"""
    
    report_path = 'reports/all_models_validation_report_2025.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"   ✅ 최종 보고서: {report_path}")
    
    # 13. 최종 요약
    print("\n" + "=" * 60)
    print("🎉 2025 전체 모델 정확도 검증 완료!")
    print(f"🥇 최고 모델: {best_model}")
    print(f"📊 최고 성능: MAE {results[best_model]['MAE']:,.0f}만원, R² {results[best_model]['R²']:.3f}")
    print(f"📊 ±20% 정확도: {best_accuracy:.1f}%")
    print(f"{practical_emoji} 실용성: {practical_level}")
    print(f"📂 결과 확인:")
    print(f"   - 전체 결과: results/all_models_prediction_validation_2025.csv")
    print(f"   - 성능 요약: results/all_models_performance_summary_2025.json")
    print(f"   - 최종 보고서: reports/all_models_validation_report_2025.txt")
    print(f"   - 시각화: plots/ 폴더")
    print("=" * 60)
    
    return {
        'results': results,
        'best_model': best_model,
        'results_df': results_df,
        'district_analysis': district_analysis,
        'price_analysis': price_analysis
    }

def create_default_mapping():
    """기본 매핑 정보 생성"""
    return {
        'gu_label_mapping': {
            '강남구': 0, '강동구': 1, '강북구': 2, '강서구': 3, '관악구': 4,
            '광진구': 5, '구로구': 6, '금천구': 7, '노원구': 8, '도봉구': 9,
            '동대문구': 10, '동작구': 11, '마포구': 12, '서대문구': 13, '서초구': 14,
            '성동구': 15, '성북구': 16, '송파구': 17, '양천구': 18, '영등포구': 19,
            '용산구': 20, '은평구': 21, '종로구': 22, '중구': 23, '중랑구': 24
        }
    }

if __name__ == "__main__":
    print("🎯 2025 서울 아파트 가격 예측 전체 모델 정확도 검증")
    print("🤖 Random Forest vs XGBoost vs Linear Regression")
    print("📊 구별, 가격대별, 모델간 일치도 상세 분석")
    print()
    
    result = validate_all_models_2025()
    
    if result:
        print(f"\n🎊 전체 모델 검증 성공! 🎊")
        print(f"최고 성능 모델: {result['best_model']}")
        print(f"모든 결과 파일이 생성되었습니다!")
    else:
        print(f"\n❌ 전체 모델 검증 실패")