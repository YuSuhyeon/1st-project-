"""
데이터 문제 진단 스크립트
"""

import pandas as pd
import numpy as np

def diagnose_data():
    """데이터 문제 진단"""
    print("🔍 데이터 문제 진단 시작")
    print("=" * 50)
    
    # 1. 데이터 로드
    df = pd.read_csv("data/raw/20250604_182224_seoul_real_estate.csv")
    print(f"📊 총 데이터: {len(df):,}건")
    
    # 2. 날짜 관련 컬럼들 체크
    print(f"\n📅 날짜 관련 컬럼들:")
    date_columns = ['RCPT_YR', 'CTRT_DAY', 'RTRCN_DAY', 'ARCH_YR']
    for col in date_columns:
        if col in df.columns:
            print(f"  {col}: {df[col].dtype}")
            if df[col].dtype == 'object':
                print(f"    샘플: {df[col].dropna().head(3).tolist()}")
            else:
                print(f"    범위: {df[col].min()} ~ {df[col].max()}")
    
    # 3. CTRT_DAY에서 연도 추출해보기
    df['CTRT_DAY'] = pd.to_datetime(df['CTRT_DAY'], errors='coerce')
    df['YEAR_FROM_CTRT'] = df['CTRT_DAY'].dt.year
    
    print(f"\n📊 CTRT_DAY에서 추출한 연도 분포:")
    year_counts = df['YEAR_FROM_CTRT'].value_counts().sort_index()
    for year, count in year_counts.items():
        if pd.notna(year):
            print(f"  {int(year)}년: {count:,}건")
    
    # 4. RCPT_YR과 CTRT_DAY 연도 비교
    if 'RCPT_YR' in df.columns:
        print(f"\n🔍 RCPT_YR vs CTRT_DAY 연도 비교:")
        comparison = df[['RCPT_YR', 'YEAR_FROM_CTRT']].dropna()
        print(f"  RCPT_YR 고유값: {sorted(comparison['RCPT_YR'].unique())}")
        print(f"  CTRT_DAY 연도 고유값: {sorted(comparison['YEAR_FROM_CTRT'].unique())}")
        
        # 불일치 체크
        mismatch = comparison[comparison['RCPT_YR'] != comparison['YEAR_FROM_CTRT']]
        if len(mismatch) > 0:
            print(f"  ⚠️ 불일치 데이터: {len(mismatch):,}건")
            print(f"  불일치 샘플:")
            print(mismatch.head())
    
    # 5. 이상한 데이터 샘플 출력
    print(f"\n🚨 문제가 될 수 있는 데이터:")
    
    # 2022년 이전 데이터
    old_data = df[df['YEAR_FROM_CTRT'] < 2022]
    if len(old_data) > 0:
        print(f"  2022년 이전 데이터: {len(old_data):,}건")
        print(f"  연도별 분포: {old_data['YEAR_FROM_CTRT'].value_counts().sort_index().head(10).to_dict()}")
    
    # 미래 데이터
    future_data = df[df['YEAR_FROM_CTRT'] > 2025]
    if len(future_data) > 0:
        print(f"  2025년 이후 데이터: {len(future_data):,}건")
    
    # 6. CTRT_DAY 원본 데이터 체크
    print(f"\n📅 CTRT_DAY 원본 샘플:")
    ctrt_samples = df['CTRT_DAY'].dropna().head(20)
    for i, date in enumerate(ctrt_samples):
        if pd.notna(date):
            print(f"  {i+1:2d}: {date} → {date.year}년")
    
    # 7. 정상 데이터 필터링 결과
    normal_data = df[(df['YEAR_FROM_CTRT'] >= 2022) & (df['YEAR_FROM_CTRT'] <= 2025)]
    print(f"\n✅ 정상 범위 데이터 (2022-2025): {len(normal_data):,}건")
    
    normal_year_counts = normal_data['YEAR_FROM_CTRT'].value_counts().sort_index()
    for year, count in normal_year_counts.items():
        percentage = (count / len(normal_data)) * 100
        print(f"  {int(year)}년: {count:,}건 ({percentage:.1f}%)")
    
    return df, normal_data

if __name__ == "__main__":
    df, normal_data = diagnose_data()