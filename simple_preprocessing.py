"""
간단한 데이터 전처리 (Categorical 오류 방지)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def simple_preprocessing():
    """간단한 전처리"""
    print("🚀 간단한 데이터 전처리 시작")
    print("=" * 50)
    
    # 1. 데이터 로드
    df = pd.read_csv("data/raw/20250604_182224_seoul_real_estate.csv")
    print(f"📊 원본 데이터: {len(df):,}건")
    
    # 2. 날짜 처리
    df['CTRT_DAY'] = pd.to_datetime(df['CTRT_DAY'])
    df['YEAR'] = df['CTRT_DAY'].dt.year
    df['MONTH'] = df['CTRT_DAY'].dt.month
    df['QUARTER'] = df['CTRT_DAY'].dt.quarter
    
    # 3. 2022-2025년 데이터만 사용
    df = df[(df['YEAR'] >= 2022) & (df['YEAR'] <= 2025)].copy()
    
    # 4. 수치형 변환
    numeric_cols = ['THING_AMT', 'ARCH_AREA', 'FLR', 'ARCH_YR']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 5. 파생변수 생성 (문자열로)
    df['PYEONG'] = df['ARCH_AREA'] * 0.3025
    df['BUILDING_AGE'] = 2025 - df['ARCH_YR']
    
    # 평형대 (문자열로 직접 생성)
    def get_pyeong_group(pyeong):
        if pd.isna(pyeong):
            return 'Unknown'
        elif pyeong < 15:
            return '소형'
        elif pyeong < 25:
            return '중소형'
        elif pyeong < 35:
            return '중형'
        elif pyeong < 50:
            return '대형'
        else:
            return '초대형'
    
    df['PYEONG_GROUP'] = df['PYEONG'].apply(get_pyeong_group)
    
    # 계절 (문자열로)
    season_map = {12: '겨울', 1: '겨울', 2: '겨울',
                 3: '봄', 4: '봄', 5: '봄',
                 6: '여름', 7: '여름', 8: '여름',
                 9: '가을', 10: '가을', 11: '가을'}
    df['SEASON'] = df['MONTH'].map(season_map)
    
    # 거래유형
    if 'DCLR_SE' in df.columns:
        df['IS_DIRECT_TRADE'] = (df['DCLR_SE'] == '직거래').astype(int)
    else:
        df['IS_DIRECT_TRADE'] = 0
    
    print(f"✅ 파생변수 생성 완료")
    
    # 6. 필수 컬럼 결측치 제거
    essential_cols = ['THING_AMT', 'ARCH_AREA', 'CGG_NM']
    df = df.dropna(subset=essential_cols)
    print(f"📊 결측치 제거 후: {len(df):,}건")
    
    # 7. 극단 이상치 제거 (1%-99%)
    price_low = df['THING_AMT'].quantile(0.01)
    price_high = df['THING_AMT'].quantile(0.99)
    df = df[(df['THING_AMT'] >= price_low) & (df['THING_AMT'] <= price_high)]
    print(f"📊 이상치 제거 후: {len(df):,}건")
    
    # 8. 학습/테스트 분할
    train_data = df[df['YEAR'] <= 2024].copy()
    test_data = df[df['YEAR'] == 2025].copy()
    
    print(f"🎯 학습 데이터: {len(train_data):,}건")
    print(f"🧪 테스트 데이터: {len(test_data):,}건")
    
    # 9. 피처 선택 (간단하게)
    feature_columns = [
        'CGG_NM',           # 자치구
        'ARCH_AREA',        # 전용면적
        'PYEONG',           # 평수
        'FLR',              # 층수
        'BUILDING_AGE',     # 건물나이
        'YEAR',             # 거래년도
        'MONTH',            # 거래월
        'PYEONG_GROUP',     # 평형대
        'SEASON',           # 계절
        'IS_DIRECT_TRADE'   # 직거래여부
    ]
    
    print(f"🎯 선택된 피처: {len(feature_columns)}개")
    
    # 10. 범주형 변수 인코딩
    categorical_features = ['CGG_NM', 'PYEONG_GROUP', 'SEASON']
    encoders = {}
    
    for feature in categorical_features:
        print(f"🔤 {feature} 인코딩...")
        
        # 결측치 처리
        train_data[feature] = train_data[feature].fillna('Unknown')
        test_data[feature] = test_data[feature].fillna('Unknown')
        
        # 모든 고유값 수집
        all_values = sorted(list(set(train_data[feature].unique()) | set(test_data[feature].unique())))
        
        # LabelEncoder
        encoders[feature] = LabelEncoder()
        encoders[feature].fit(all_values)
        
        # 변환
        train_data[feature] = encoders[feature].transform(train_data[feature])
        test_data[feature] = encoders[feature].transform(test_data[feature])
        
        print(f"  → {len(all_values)}개 고유값으로 인코딩 완료")
    
    # 11. 결측치 처리 (수치형)
    numeric_features = ['ARCH_AREA', 'PYEONG', 'FLR', 'BUILDING_AGE']
    for feature in numeric_features:
        median_val = train_data[feature].median()
        train_data[feature] = train_data[feature].fillna(median_val)
        test_data[feature] = test_data[feature].fillna(median_val)
    
    # 12. 최종 데이터셋
    X_train = train_data[feature_columns].copy()
    y_train = train_data['THING_AMT'].copy()
    X_test = test_data[feature_columns].copy()
    y_test = test_data['THING_AMT'].copy()
    
    print(f"\n📊 최종 데이터셋:")
    print(f"  학습 피처: {X_train.shape}")
    print(f"  학습 타겟: {y_train.shape}")
    print(f"  테스트 피처: {X_test.shape}")
    print(f"  테스트 타겟: {y_test.shape}")
    
    # 13. 데이터 저장
    import os
    os.makedirs('data/processed', exist_ok=True)
    
    X_train.to_csv('data/processed/X_train.csv', index=False)
    X_test.to_csv('data/processed/X_test.csv', index=False)
    y_train.to_csv('data/processed/y_train.csv', index=False, header=['THING_AMT'])
    y_test.to_csv('data/processed/y_test.csv', index=False, header=['THING_AMT'])
    
    print(f"\n💾 데이터 저장 완료!")
    print(f"✅ data/processed/X_train.csv")
    print(f"✅ data/processed/X_test.csv")
    print(f"✅ data/processed/y_train.csv")
    print(f"✅ data/processed/y_test.csv")
    
    print(f"\n🎉 전처리 완료!")
    print(f"📋 다음 단계: 모델 학습")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = simple_preprocessing()


    