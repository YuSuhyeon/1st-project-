"""
서울 아파트 가격 예측 - 머신러닝 모델 학습
XGBoost, Random Forest, Linear Regression
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class ApartmentPriceModeler:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.results = {}
        
    def load_data(self):
        """전처리된 데이터 로드"""
        print("📂 전처리된 데이터 로드")
        print("=" * 40)
        
        try:
            self.X_train = pd.read_csv('data/processed/X_train.csv')
            self.X_test = pd.read_csv('data/processed/X_test.csv')
            self.y_train = pd.read_csv('data/processed/y_train.csv')['THING_AMT']
            self.y_test = pd.read_csv('data/processed/y_test.csv')['THING_AMT']
            
            print(f"✅ 학습 데이터: {self.X_train.shape}")
            print(f"✅ 테스트 데이터: {self.X_test.shape}")
            print(f"✅ 학습 타겟 평균: {self.y_train.mean():,.0f}만원")
            print(f"✅ 테스트 타겟 평균: {self.y_test.mean():,.0f}만원")
            
            return True
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False
    
    def train_xgboost(self):
        """XGBoost 모델 학습"""
        print(f"\n🚀 XGBoost 모델 학습")
        print("-" * 30)
        
        # XGBoost 매개변수
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # 모델 생성 및 학습
        self.models['XGBoost'] = xgb.XGBRegressor(**xgb_params)
        
        print("🔄 XGBoost 학습 중...")
        self.models['XGBoost'].fit(self.X_train, self.y_train)
        print("✅ XGBoost 학습 완료!")
    
    def train_random_forest(self):
        """Random Forest 모델 학습"""
        print(f"\n🌲 Random Forest 모델 학습")
        print("-" * 30)
        
        # Random Forest 매개변수
        rf_params = {
            'n_estimators': 150,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # 모델 생성 및 학습
        self.models['Random Forest'] = RandomForestRegressor(**rf_params)
        
        print("🔄 Random Forest 학습 중...")
        self.models['Random Forest'].fit(self.X_train, self.y_train)
        print("✅ Random Forest 학습 완료!")
    
    def train_linear_regression(self):
        """Linear Regression 모델 학습"""
        print(f"\n📈 Linear Regression 모델 학습")
        print("-" * 30)
        
        # 선형회귀를 위한 데이터 스케일링
        print("🔄 데이터 스케일링...")
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        # 모델 생성 및 학습
        self.models['Linear Regression'] = LinearRegression()
        
        print("🔄 Linear Regression 학습 중...")
        self.models['Linear Regression'].fit(X_train_scaled, self.y_train)
        print("✅ Linear Regression 학습 완료!")
    
    def evaluate_model(self, model_name, model):
        """개별 모델 평가"""
        print(f"\n📊 {model_name} 성능 평가")
        print("-" * 30)
        
        # 예측
        if model_name == 'Linear Regression':
            X_test_scaled = self.scaler.transform(self.X_test)
            X_train_scaled = self.scaler.transform(self.X_train)
            y_pred_test = model.predict(X_test_scaled)
            y_pred_train = model.predict(X_train_scaled)
        else:
            y_pred_test = model.predict(self.X_test)
            y_pred_train = model.predict(self.X_train)
        
        # 음수 예측값 제거 (가격은 양수여야 함)
        y_pred_test = np.maximum(y_pred_test, 1000)  # 최소 1000만원
        y_pred_train = np.maximum(y_pred_train, 1000)
        
        # 메트릭 계산
        test_mape = mean_absolute_percentage_error(self.y_test, y_pred_test) * 100
        test_r2 = r2_score(self.y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
        
        train_mape = mean_absolute_percentage_error(self.y_train, y_pred_train) * 100
        train_r2 = r2_score(self.y_train, y_pred_train)
        
        # 결과 저장
        self.results[model_name] = {
            'test_mape': test_mape,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'train_mape': train_mape,
            'train_r2': train_r2,
            'y_pred_test': y_pred_test,
            'y_pred_train': y_pred_train
        }
        
        # 결과 출력
        print(f"🎯 테스트 성능:")
        print(f"  MAPE: {test_mape:.2f}%")
        print(f"  R²: {test_r2:.3f}")
        print(f"  RMSE: {test_rmse:,.0f}만원")
        
        print(f"📚 학습 성능:")
        print(f"  MAPE: {train_mape:.2f}%")
        print(f"  R²: {train_r2:.3f}")
        
        # 과적합 체크
        overfitting = train_mape - test_mape
        if overfitting > 5:
            print(f"⚠️ 과적합 가능성: {overfitting:.1f}%")
        else:
            print(f"✅ 적절한 일반화: {overfitting:.1f}%")
        
        return test_mape, test_r2, test_rmse
    
    def compare_models(self):
        """모델 성능 비교"""
        print(f"\n🏆 모델 성능 비교")
        print("=" * 60)
        
        # 비교 테이블 생성
        comparison_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Test_MAPE(%)': [self.results[m]['test_mape'] for m in self.results.keys()],
            'Test_R²': [self.results[m]['test_r2'] for m in self.results.keys()],
            'Test_RMSE(만원)': [self.results[m]['test_rmse'] for m in self.results.keys()],
            'Train_MAPE(%)': [self.results[m]['train_mape'] for m in self.results.keys()],
            'Train_R²': [self.results[m]['train_r2'] for m in self.results.keys()]
        })
        
        # 성능순 정렬 (MAPE 기준)
        comparison_df = comparison_df.sort_values('Test_MAPE(%)')
        
        print("📊 성능 비교표:")
        print(comparison_df.round(2).to_string(index=False))
        
        # 최고 성능 모델
        best_model = comparison_df.iloc[0]['Model']
        best_mape = comparison_df.iloc[0]['Test_MAPE(%)']
        best_r2 = comparison_df.iloc[0]['Test_R²']
        
        print(f"\n🥇 최고 성능 모델: {best_model}")
        print(f"   MAPE: {best_mape:.2f}%")
        print(f"   R²: {best_r2:.3f}")
        
        # 목표 달성 여부
        print(f"\n🎯 목표 달성 여부:")
        if best_mape <= 15:
            print(f"✅ MAPE 목표 달성: {best_mape:.2f}% ≤ 15%")
        else:
            print(f"❌ MAPE 목표 미달성: {best_mape:.2f}% > 15%")
        
        if best_r2 >= 0.65:
            print(f"✅ R² 목표 달성: {best_r2:.3f} ≥ 0.65")
        else:
            print(f"❌ R² 목표 미달성: {best_r2:.3f} < 0.65")
        
        return comparison_df
    
    def analyze_feature_importance(self):
        """피처 중요도 분석"""
        print(f"\n🎯 피처 중요도 분석")
        print("-" * 40)
        
        # XGBoost와 Random Forest 피처 중요도
        feature_names = self.X_train.columns
        
        importance_df = pd.DataFrame({
            'Feature': feature_names
        })
        
        if 'XGBoost' in self.results:
            xgb_importance = self.models['XGBoost'].feature_importances_
            importance_df['XGBoost'] = xgb_importance
        
        if 'Random Forest' in self.results:
            rf_importance = self.models['Random Forest'].feature_importances_
            importance_df['Random_Forest'] = rf_importance
        
        # 평균 중요도 계산
        numeric_cols = [col for col in importance_df.columns if col != 'Feature']
        if numeric_cols:
            importance_df['Average'] = importance_df[numeric_cols].mean(axis=1)
            importance_df = importance_df.sort_values('Average', ascending=False)
        
        print("🏆 피처 중요도 순위:")
        for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
            feature = row['Feature']
            avg_importance = row['Average'] if 'Average' in row else 0
            print(f"  {i:2d}. {feature}: {avg_importance:.3f}")
        
        return importance_df
    
    def create_prediction_plots(self):
        """예측 결과 시각화"""
        print(f"\n📈 예측 결과 시각화")
        print("-" * 30)
        
        # 최고 성능 모델의 예측값
        best_model = min(self.results.keys(), 
                        key=lambda x: self.results[x]['test_mape'])
        y_pred = self.results[best_model]['y_pred_test']
        
        # 2x2 플롯
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 실제값 vs 예측값 산점도
        axes[0,0].scatter(self.y_test, y_pred, alpha=0.5, s=20)
        axes[0,0].plot([self.y_test.min(), self.y_test.max()], 
                      [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0,0].set_xlabel('실제값 (만원)')
        axes[0,0].set_ylabel('예측값 (만원)')
        axes[0,0].set_title(f'{best_model}: 실제값 vs 예측값')
        
        # 2. 잔차 플롯
        residuals = self.y_test - y_pred
        axes[0,1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[0,1].axhline(y=0, color='r', linestyle='--')
        axes[0,1].set_xlabel('예측값 (만원)')
        axes[0,1].set_ylabel('잔차 (만원)')
        axes[0,1].set_title('잔차 플롯')
        
        # 3. 모델별 MAPE 비교
        models = list(self.results.keys())
        mapes = [self.results[m]['test_mape'] for m in models]
        bars = axes[1,0].bar(range(len(models)), mapes, 
                           color=['gold', 'silver', 'orange'][:len(models)])
        axes[1,0].set_xticks(range(len(models)))
        axes[1,0].set_xticklabels(models, rotation=45)
        axes[1,0].set_ylabel('MAPE (%)')
        axes[1,0].set_title('모델별 MAPE 비교')
        
        # 값 표시
        for bar, mape in zip(bars, mapes):
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                         f'{mape:.1f}%', ha='center', va='bottom')
        
        # 4. 오차 분포 히스토그램
        error_percent = (residuals / self.y_test) * 100
        axes[1,1].hist(error_percent, bins=30, alpha=0.7, color='skyblue')
        axes[1,1].axvline(x=0, color='r', linestyle='--')
        axes[1,1].set_xlabel('상대 오차 (%)')
        axes[1,1].set_ylabel('빈도')
        axes[1,1].set_title('예측 오차 분포')
        
        plt.tight_layout()
        plt.show()
        
        # 오차 통계
        print(f"📊 예측 오차 통계:")
        print(f"  평균 절대 오차: {np.abs(residuals).mean():,.0f}만원")
        print(f"  오차 표준편차: {residuals.std():,.0f}만원")
        print(f"  ±20% 내 예측 비율: {(np.abs(error_percent) <= 20).mean()*100:.1f}%")
        print(f"  ±10% 내 예측 비율: {(np.abs(error_percent) <= 10).mean()*100:.1f}%")
    
    def save_models(self):
        """모델 저장"""
        print(f"\n💾 모델 저장")
        print("-" * 30)
        
        # 디렉토리 생성
        Path("models").mkdir(exist_ok=True)
        
        # 각 모델 저장
        for name, model in self.models.items():
            filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
            joblib.dump(model, filename)
            print(f"✅ {name} 저장: {filename}")
        
        # 스케일러 저장 (Linear Regression용)
        joblib.dump(self.scaler, "models/scaler.pkl")
        print(f"✅ Scaler 저장: models/scaler.pkl")
        
        # 결과 저장
        results_summary = {}
        for name, result in self.results.items():
            results_summary[name] = {
                'test_mape': result['test_mape'],
                'test_r2': result['test_r2'],
                'test_rmse': result['test_rmse'],
                'train_mape': result['train_mape'],
                'train_r2': result['train_r2']
            }
        
        import json
        with open('models/model_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        print(f"✅ 결과 저장: models/model_results.json")
    
    def run_full_pipeline(self):
        """전체 모델링 파이프라인 실행"""
        print("🤖 머신러닝 모델 학습 파이프라인 시작")
        print("=" * 60)
        
        # 1. 데이터 로드
        if not self.load_data():
            return
        
        # 2. 모델 학습
        self.train_xgboost()
        self.train_random_forest()
        self.train_linear_regression()
        
        # 3. 모델 평가
        for name, model in self.models.items():
            self.evaluate_model(name, model)
        
        # 4. 성능 비교
        comparison_df = self.compare_models()
        
        # 5. 피처 중요도 분석
        importance_df = self.analyze_feature_importance()
        
        # 6. 시각화
        self.create_prediction_plots()
        
        # 7. 모델 저장
        self.save_models()
        
        print(f"\n🎉 모델링 완료!")
        print("=" * 60)
        print(f"✅ 3개 모델 학습 완료")
        print(f"✅ 성능 평가 완료")
        print(f"✅ 피처 중요도 분석 완료")
        print(f"✅ 시각화 완료")
        print(f"✅ 모델 저장 완료")
        
        # 최종 요약
        best_model = comparison_df.iloc[0]['Model']
        best_mape = comparison_df.iloc[0]['Test_MAPE(%)']
        
        print(f"\n🏆 최종 결과:")
        print(f"  최고 성능 모델: {best_model}")
        print(f"  MAPE: {best_mape:.2f}%")
        print(f"  목표 달성: {'✅' if best_mape <= 15 else '❌'}")

def main():
    """메인 실행 함수"""
    modeler = ApartmentPriceModeler()
    modeler.run_full_pipeline()

if __name__ == "__main__":
    main()