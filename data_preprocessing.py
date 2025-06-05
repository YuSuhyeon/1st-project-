"""
서울 아파트 가격 예측 - 모델링을 위한 데이터 전처리
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class DataPreprocessor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.train_data = None
        self.test_data = None
        self.encoders = {}
        self.scaler = None
        self.feature_columns = []
        
    def load_and_clean_data(self):
        """데이터 로드 및 기본 정리"""
        print("🔄 데이터 로드 및 기본 정리")
        print("=" * 50)
        
        # 데이터 로드
        self.df = pd.read_csv(self.data_path)
        print(f"📊 원본 데이터: {len(self.df):,}건")
        
        # 날짜 처리
        self.df['CTRT_DAY'] = pd.to_datetime(self.df['CTRT_DAY'])
        self.df['YEAR'] = self.df['CTRT_DAY'].dt.year
        self.df['MONTH'] = self.df['CTRT_DAY'].dt.month
        self.df['QUARTER'] = self.df['CTRT_DAY'].dt.quarter
        self.df['DAY_OF_YEAR'] = self.df['CTRT_DAY'].dt.dayofyear
        
        # 2022-2025년 데이터만 사용
        self.df = self.df[(self.df['YEAR'] >= 2022) & (self.df['YEAR'] <= 2025)].copy()
        print(f"📊 기간 필터링 후: {len(self.df):,}건 (2022-2025)")
        
        # 수치형 변환
        numeric_cols = ['THING_AMT', 'ARCH_AREA', 'LAND_AREA', 'FLR', 'ARCH_YR']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 필수 컬럼 결측치 제거
        essential_cols = ['THING_AMT', 'ARCH_AREA', 'CGG_NM']
        before_count = len(self.df)
        self.df = self.df.dropna(subset=essential_cols)
        after_count = len(self.df)
        print(f"📊 필수 컬럼 결측치 제거: {before_count:,} → {after_count:,}건 (-{before_count-after_count:,}건)")
        
        return self.df
    
    def create_features(self):
        """파생변수 생성"""
        print(f"\n🔧 파생변수 생성")
        print("-" * 30)
        
        # 기본 파생변수
        if 'ARCH_AREA' in self.df.columns:
            self.df['PYEONG'] = self.df['ARCH_AREA'] * 0.3025
            self.df['PRICE_PER_SQM'] = self.df['THING_AMT'] / self.df['ARCH_AREA']
            self.df['PRICE_PER_PYEONG'] = self.df['THING_AMT'] / self.df['PYEONG']
            print(f"✅ 평수, 평당가격, 평방미터당 가격 생성")
        
        if 'ARCH_YR' in self.df.columns:
            self.df['BUILDING_AGE'] = 2025 - self.df['ARCH_YR']
            print(f"✅ 건물나이 생성")
        
        # 범주형 파생변수
        if 'PYEONG' in self.df.columns:
            self.df['PYEONG_GROUP'] = pd.cut(
                self.df['PYEONG'], 
                bins=[0, 15, 25, 35, 50, 100],
                labels=['소형', '중소형', '중형', '대형', '초대형']
            )
            print(f"✅ 평형대 그룹 생성")
        
        if 'BUILDING_AGE' in self.df.columns:
            self.df['AGE_GROUP'] = pd.cut(
                self.df['BUILDING_AGE'],
                bins=[0, 5, 10, 20, 30, 100],
                labels=['신축', '준신축', '보통', '노후', '매우노후']
            )
            print(f"✅ 건물연령 그룹 생성")
        
        if 'FLR' in self.df.columns:
            # 층수 그룹
            self.df['FLOOR_GROUP'] = pd.cut(
                self.df['FLR'],
                bins=[0, 5, 10, 15, 100],
                labels=['저층', '중저층', '중고층', '고층']
            )
            print(f"✅ 층수 그룹 생성")
        
        # 계절 정보
        season_map = {12: '겨울', 1: '겨울', 2: '겨울',
                     3: '봄', 4: '봄', 5: '봄',
                     6: '여름', 7: '여름', 8: '여름',
                     9: '가을', 10: '가을', 11: '가을'}
        self.df['SEASON'] = self.df['MONTH'].map(season_map)
        print(f"✅ 계절 정보 생성")
        
        # 거래유형 간소화
        if 'DCLR_SE' in self.df.columns:
            self.df['IS_DIRECT_TRADE'] = (self.df['DCLR_SE'] == '직거래').astype(int)
            print(f"✅ 직거래 여부 생성")
        
        # 지역별 평균가격 대비 비율
        district_avg = self.df.groupby('CGG_NM')['THING_AMT'].mean()
        self.df['DISTRICT_PRICE_RATIO'] = self.df.apply(
            lambda x: x['THING_AMT'] / district_avg[x['CGG_NM']], axis=1
        )
        print(f"✅ 지역 평균가격 대비 비율 생성")
        
        # 가격 분위수
        self.df['PRICE_QUARTILE'] = pd.qcut(
            self.df['THING_AMT'], 
            q=4, 
            labels=['하위25%', '중하위25%', '중상위25%', '상위25%']
        )
        print(f"✅ 가격 분위수 생성")
        
        print(f"📊 총 컬럼 수: {len(self.df.columns)}개")
        
    def handle_outliers(self):
        """이상치 처리"""
        print(f"\n🚨 이상치 처리")
        print("-" * 30)
        
        # 가격 이상치 처리 (IQR 방법)
        Q1 = self.df['THING_AMT'].quantile(0.05)  # 5% 하위
        Q3 = self.df['THING_AMT'].quantile(0.95)  # 95% 상위
        
        outlier_count = len(self.df[(self.df['THING_AMT'] < Q1) | (self.df['THING_AMT'] > Q3)])
        print(f"📊 가격 이상치 (5%-95% 범위 밖): {outlier_count:,}건 ({outlier_count/len(self.df)*100:.1f}%)")
        print(f"  → 하한: {Q1:,.0f}만원, 상한: {Q3:,.0f}만원")
        
        # 면적 이상치 확인
        if 'ARCH_AREA' in self.df.columns:
            area_Q1 = self.df['ARCH_AREA'].quantile(0.05)
            area_Q3 = self.df['ARCH_AREA'].quantile(0.95)
            area_outliers = len(self.df[(self.df['ARCH_AREA'] < area_Q1) | (self.df['ARCH_AREA'] > area_Q3)])
            print(f"📊 면적 이상치 (5%-95% 범위 밖): {area_outliers:,}건")
            print(f"  → 하한: {area_Q1:.1f}㎡, 상한: {area_Q3:.1f}㎡")
        
        # 극단적 이상치만 제거 (1%-99% 범위)
        price_low = self.df['THING_AMT'].quantile(0.01)
        price_high = self.df['THING_AMT'].quantile(0.99)
        
        before_count = len(self.df)
        self.df = self.df[
            (self.df['THING_AMT'] >= price_low) & 
            (self.df['THING_AMT'] <= price_high)
        ].copy()
        after_count = len(self.df)
        
        print(f"🔧 극단 이상치 제거 (1%-99% 유지): {before_count:,} → {after_count:,}건 (-{before_count-after_count:,}건)")
        
    def split_data(self):
        """학습/테스트 데이터 분할"""
        print(f"\n📊 학습/테스트 데이터 분할")
        print("-" * 30)
        
        # 시계열 분할: 2022-2024 학습, 2025 테스트
        self.train_data = self.df[self.df['YEAR'] <= 2024].copy()
        self.test_data = self.df[self.df['YEAR'] == 2025].copy()
        
        print(f"🎯 학습 데이터 (2022-2024): {len(self.train_data):,}건")
        print(f"🧪 테스트 데이터 (2025): {len(self.test_data):,}건")
        print(f"📊 분할 비율: {len(self.train_data)/(len(self.train_data)+len(self.test_data))*100:.1f}% : {len(self.test_data)/(len(self.train_data)+len(self.test_data))*100:.1f}%")
        
        # 연도별 통계
        year_stats = self.df.groupby('YEAR').agg({
            'THING_AMT': ['mean', 'count']
        }).round(0)
        
        print(f"\n📅 연도별 데이터 분포:")
        for year in year_stats.index:
            mean_price = year_stats.loc[year, ('THING_AMT', 'mean')]
            count = year_stats.loc[year, ('THING_AMT', 'count')]
            data_type = "학습" if year <= 2024 else "테스트"
            print(f"  {year}년: {count:,}건, 평균 {mean_price:,.0f}만원 ({data_type})")
    
    def select_features(self):
        """모델링용 피처 선택"""
        print(f"\n🎯 모델링용 피처 선택")
        print("-" * 30)
        
        # 후보 피처들
        candidate_features = [
            # 기본 정보
            'CGG_NM',           # 자치구 (가장 중요!)
            'STDG_CD',          # 법정동코드
            
            # 물리적 속성
            'ARCH_AREA',        # 전용면적
            'PYEONG',           # 평수
            'FLR',              # 층수
            'BUILDING_AGE',     # 건물나이
            
            # 시간 정보
            'YEAR',             # 거래년도
            'MONTH',            # 거래월
            'QUARTER',          # 분기
            'SEASON',           # 계절
            
            # 파생변수
            'PYEONG_GROUP',     # 평형대
            'AGE_GROUP',        # 건물연령그룹
            'FLOOR_GROUP',      # 층수그룹
            'IS_DIRECT_TRADE',  # 직거래여부
            'DISTRICT_PRICE_RATIO',  # 지역가격비율
            
            # 가격 관련 (타겟 제외)
            'PRICE_PER_SQM',    # 평방미터당 가격 (상관성 높을 수 있어 제외 고려)
        ]
        
        # 존재하는 피처만 선택
        available_features = []
        for feature in candidate_features:
            if feature in self.train_data.columns:
                # 결측치 비율 확인
                missing_ratio = self.train_data[feature].isnull().sum() / len(self.train_data)
                if missing_ratio < 0.5:  # 결측치 50% 미만만 사용
                    available_features.append(feature)
                    print(f"✅ {feature}: 결측치 {missing_ratio*100:.1f}%")
                else:
                    print(f"❌ {feature}: 결측치 {missing_ratio*100:.1f}% (제외)")
            else:
                print(f"❌ {feature}: 컬럼 없음")
        
        # PRICE_PER_SQM 제거 (타겟과 강한 상관관계)
        if 'PRICE_PER_SQM' in available_features:
            available_features.remove('PRICE_PER_SQM')
            print(f"❌ PRICE_PER_SQM: 타겟과 강한 상관관계로 제외")
        
        self.feature_columns = available_features
        print(f"\n🎯 최종 선택된 피처 ({len(self.feature_columns)}개):")
        for i, feature in enumerate(self.feature_columns, 1):
            print(f"  {i:2d}. {feature}")
    
    def encode_categorical_features(self):
        """범주형 변수 인코딩"""
        print(f"\n🔤 범주형 변수 인코딩")
        print("-" * 30)
        
        categorical_features = [
            'CGG_NM', 'STDG_CD', 'PYEONG_GROUP', 'AGE_GROUP', 
            'FLOOR_GROUP', 'SEASON'
        ]
        
        # 선택된 피처 중 범주형만 처리
        categorical_to_encode = [f for f in categorical_features if f in self.feature_columns]
        
        for feature in categorical_to_encode:
            print(f"🔤 {feature} 인코딩...")
            
            # 학습 데이터로 LabelEncoder 학습
            self.encoders[feature] = LabelEncoder()
            
            # 결측치를 'Unknown'으로 채우기
            self.train_data[feature] = self.train_data[feature].fillna('Unknown').astype(str)
            self.test_data[feature] = self.test_data[feature].fillna('Unknown').astype(str)
            
            # 학습 데이터로 인코더 학습
            self.encoders[feature].fit(self.train_data[feature])
            
            # 학습 데이터 변환
            self.train_data[feature] = self.encoders[feature].transform(self.train_data[feature])
            
            # 테스트 데이터 변환 (미지의 값 처리)
            test_values = self.test_data[feature].copy()
            unknown_mask = ~test_values.isin(self.encoders[feature].classes_)
            test_values[unknown_mask] = 'Unknown'
            
            # Unknown이 인코더에 없으면 추가
            if 'Unknown' not in self.encoders[feature].classes_:
                # 새로운 인코더로 다시 학습 (Unknown 포함)
                combined_values = list(self.encoders[feature].classes_) + ['Unknown']
                self.encoders[feature] = LabelEncoder()
                self.encoders[feature].fit(combined_values)
                
                # 다시 변환
                self.train_data[feature] = self.encoders[feature].transform(
                    self.train_data[feature].map(
                        dict(zip(range(len(self.encoders[feature].classes_)-1), 
                                self.encoders[feature].classes_[:-1]))
                    ).fillna('Unknown').astype(str)
                )
            
            self.test_data[feature] = self.encoders[feature].transform(test_values)
            
            unique_count = len(self.encoders[feature].classes_)
            print(f"  → {unique_count}개 고유값으로 인코딩 완료")
    
    def prepare_final_datasets(self):
        """최종 모델링용 데이터셋 준비"""
        print(f"\n🎯 최종 데이터셋 준비")
        print("-" * 30)
        
        # 수치형 피처 결측치 처리
        numeric_features = [f for f in self.feature_columns 
                          if f not in ['CGG_NM', 'STDG_CD', 'PYEONG_GROUP', 'AGE_GROUP', 'FLOOR_GROUP', 'SEASON']]
        
        for feature in numeric_features:
            if feature in self.train_data.columns:
                # 중위값으로 결측치 대체
                median_value = self.train_data[feature].median()
                self.train_data[feature] = self.train_data[feature].fillna(median_value)
                self.test_data[feature] = self.test_data[feature].fillna(median_value)
                print(f"📊 {feature}: 중위값 {median_value:.2f}로 결측치 대체")
        
        # 피처와 타겟 분리
        X_train = self.train_data[self.feature_columns].copy()
        y_train = self.train_data['THING_AMT'].copy()
        
        X_test = self.test_data[self.feature_columns].copy()
        y_test = self.test_data['THING_AMT'].copy()
        
        print(f"\n📊 최종 데이터셋 요약:")
        print(f"  학습 피처: {X_train.shape}")
        print(f"  학습 타겟: {y_train.shape}")
        print(f"  테스트 피처: {X_test.shape}")
        print(f"  테스트 타겟: {y_test.shape}")
        
        # 데이터 타입 확인
        print(f"\n📋 피처별 데이터 타입:")
        for feature in self.feature_columns:
            dtype = X_train[feature].dtype
            unique_count = X_train[feature].nunique()
            print(f"  {feature}: {dtype} (고유값 {unique_count}개)")
        
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """전처리된 데이터 저장"""
        print(f"\n💾 전처리 데이터 저장")
        print("-" * 30)
        
        # 디렉토리 생성
        import os
        os.makedirs('data/processed', exist_ok=True)
        
        # 피처 데이터 저장
        X_train.to_csv('data/processed/X_train.csv', index=False)
        X_test.to_csv('data/processed/X_test.csv', index=False)
        
        # 타겟 데이터 저장
        y_train.to_csv('data/processed/y_train.csv', index=False, header=['THING_AMT'])
        y_test.to_csv('data/processed/y_test.csv', index=False, header=['THING_AMT'])
        
        # 전체 전처리 데이터 저장
        train_full = X_train.copy()
        train_full['THING_AMT'] = y_train
        train_full.to_csv('data/processed/train_processed.csv', index=False)
        
        test_full = X_test.copy()
        test_full['THING_AMT'] = y_test
        test_full.to_csv('data/processed/test_processed.csv', index=False)
        
        print(f"✅ 전처리 데이터 저장 완료:")
        print(f"  📁 data/processed/X_train.csv")
        print(f"  📁 data/processed/X_test.csv")
        print(f"  📁 data/processed/y_train.csv")
        print(f"  📁 data/processed/y_test.csv")
        print(f"  📁 data/processed/train_processed.csv")
        print(f"  📁 data/processed/test_processed.csv")
    
    def run_preprocessing(self):
        """전체 전처리 프로세스 실행"""
        print("🚀 모델링용 데이터 전처리 시작")
        print("=" * 60)
        
        # 1단계: 데이터 로드 및 정리
        self.load_and_clean_data()
        
        # 2단계: 파생변수 생성
        self.create_features()
        
        # 3단계: 이상치 처리
        self.handle_outliers()
        
        # 4단계: 데이터 분할
        self.split_data()
        
        # 5단계: 피처 선택
        self.select_features()
        
        # 6단계: 범주형 변수 인코딩
        self.encode_categorical_features()
        
        # 7단계: 최종 데이터셋 준비
        X_train, X_test, y_train, y_test = self.prepare_final_datasets()
        
        # 8단계: 데이터 저장
        self.save_processed_data(X_train, X_test, y_train, y_test)
        
        print(f"\n🎉 데이터 전처리 완료!")
        print("=" * 60)
        print(f"✅ 데이터 정리 완료")
        print(f"✅ 파생변수 생성 완료")
        print(f"✅ 이상치 처리 완료")
        print(f"✅ 학습/테스트 분할 완료")
        print(f"✅ 피처 선택 완료 ({len(self.feature_columns)}개)")
        print(f"✅ 범주형 변수 인코딩 완료")
        print(f"✅ 최종 데이터셋 준비 완료")
        print(f"✅ 데이터 저장 완료")
        
        print(f"\n📋 다음 단계: 머신러닝 모델 학습")
        print(f"  🤖 XGBoost, Random Forest, Linear Regression")
        print(f"  📊 모델 성능 평가 및 비교")
        print(f"  🎯 피처 중요도 분석")
        
        return X_train, X_test, y_train, y_test

def main():
    """메인 실행 함수"""
    preprocessor = DataPreprocessor("data/raw/20250604_182224_seoul_real_estate.csv")
    X_train, X_test, y_train, y_test = preprocessor.run_preprocessing()
    
    # 간단한 통계 요약
    print(f"\n📊 전처리 결과 요약:")
    print(f"  학습 데이터: {len(X_train):,}건")
    print(f"  테스트 데이터: {len(X_test):,}건")
    print(f"  피처 개수: {len(X_train.columns)}개")
    print(f"  평균 타겟값 (학습): {y_train.mean():,.0f}만원")
    print(f"  평균 타겟값 (테스트): {y_test.mean():,.0f}만원")

if __name__ == "__main__":
    main()