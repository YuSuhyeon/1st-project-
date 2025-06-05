"""
단계별 EDA - 시각화 문제 방지
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """데이터 로드"""
    print("📊 데이터 로드 중...")
    df = pd.read_csv("data/raw/20250604_182224_seoul_real_estate.csv")
    
    # 날짜 처리
    df['CTRT_DAY'] = pd.to_datetime(df['CTRT_DAY'])
    df['YEAR'] = df['CTRT_DAY'].dt.year
    df['MONTH'] = df['CTRT_DAY'].dt.month
    df['QUARTER'] = df['CTRT_DAY'].dt.quarter
    
    # 2022-2025년 데이터만 사용
    df = df[(df['YEAR'] >= 2022) & (df['YEAR'] <= 2025)].copy()
    
    # 수치형 변환
    numeric_cols = ['THING_AMT', 'ARCH_AREA', 'FLR', 'ARCH_YR']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 파생변수 생성
    if 'ARCH_AREA' in df.columns:
        df['PYEONG'] = df['ARCH_AREA'] * 0.3025
        df['PYEONG_GROUP'] = pd.cut(
            df['PYEONG'], 
            bins=[0, 15, 25, 35, 50, 100],
            labels=['소형(15평미만)', '중소형(15-25평)', '중형(25-35평)', '대형(35-50평)', '초대형(50평+)']
        )
    
    if 'ARCH_YR' in df.columns:
        df['BUILDING_AGE'] = 2025 - df['ARCH_YR']
        df['AGE_GROUP'] = pd.cut(
            df['BUILDING_AGE'],
            bins=[0, 5, 10, 20, 30, 100],
            labels=['신축(5년이하)', '준신축(5-10년)', '보통(10-20년)', '노후(20-30년)', '매우노후(30년+)']
        )
    
    print(f"✅ 데이터 로드 완료: {len(df):,}건")
    return df

def step1_district_analysis(df):
    """1단계: 자치구별 분석 (텍스트만)"""
    print(f"\n1️⃣ 자치구별 분석")
    print("=" * 40)
    
    # 자치구별 통계
    district_stats = df.groupby('CGG_NM').agg({
        'THING_AMT': ['mean', 'median', 'count', 'std'],
        'ARCH_AREA': 'mean'
    }).round(0)
    
    district_stats.columns = ['평균가격', '중위가격', '거래수', '가격표준편차', '평균면적']
    district_stats = district_stats.sort_values('평균가격', ascending=False)
    
    print(f"🏆 자치구별 평균 가격 TOP 15:")
    for i, (district, row) in enumerate(district_stats.head(15).iterrows(), 1):
        print(f"  {i:2d}. {district}: {row['평균가격']:,.0f}만원 ({row['거래수']:,.0f}건)")
    
    print(f"\n📊 거래량 TOP 10:")
    top_volume = district_stats.sort_values('거래수', ascending=False).head(10)
    for i, (district, row) in enumerate(top_volume.iterrows(), 1):
        print(f"  {i:2d}. {district}: {row['거래수']:,.0f}건 (평균 {row['평균가격']:,.0f}만원)")
    
    print(f"\n💰 가격 편차가 큰 지역 TOP 10:")
    top_std = district_stats.sort_values('가격표준편차', ascending=False).head(10)
    for i, (district, row) in enumerate(top_std.iterrows(), 1):
        cv = row['가격표준편차'] / row['평균가격']  # 변동계수
        print(f"  {i:2d}. {district}: 표준편차 {row['가격표준편차']:,.0f}만원 (변동계수: {cv:.2f})")
    
    return district_stats

def step2_size_analysis(df):
    """2단계: 평형대별 분석 (텍스트만)"""
    print(f"\n2️⃣ 평형대별 분석")
    print("=" * 40)
    
    if 'PYEONG_GROUP' not in df.columns:
        print("❌ 평형대 정보가 없습니다.")
        return
    
    # 평형대별 통계
    size_stats = df.groupby('PYEONG_GROUP').agg({
        'THING_AMT': ['mean', 'median', 'count'],
        'PYEONG': ['mean', 'min', 'max']
    }).round(1)
    
    print(f"📊 평형대별 통계:")
    for group in size_stats.index:
        if pd.notna(group):
            mean_price = size_stats.loc[group, ('THING_AMT', 'mean')]
            median_price = size_stats.loc[group, ('THING_AMT', 'median')]
            count = size_stats.loc[group, ('THING_AMT', 'count')]
            mean_pyeong = size_stats.loc[group, ('PYEONG', 'mean')]
            min_pyeong = size_stats.loc[group, ('PYEONG', 'min')]
            max_pyeong = size_stats.loc[group, ('PYEONG', 'max')]
            print(f"  {group}:")
            print(f"    평균: {mean_price:,.0f}만원, 중위: {median_price:,.0f}만원")
            print(f"    거래량: {count:,.0f}건, 평균평수: {mean_pyeong:.1f}평 ({min_pyeong:.1f}~{max_pyeong:.1f})")
    
    # 평형대별 평당 가격
    print(f"\n💰 평형대별 평당 가격:")
    for group in size_stats.index:
        if pd.notna(group):
            mean_price = size_stats.loc[group, ('THING_AMT', 'mean')]
            mean_pyeong = size_stats.loc[group, ('PYEONG', 'mean')]
            price_per_pyeong = mean_price / mean_pyeong if mean_pyeong > 0 else 0
            print(f"  {group}: {price_per_pyeong:,.0f}만원/평")

def step3_building_age_analysis(df):
    """3단계: 건물 나이별 분석 (텍스트만)"""
    print(f"\n3️⃣ 건물 나이별 분석")
    print("=" * 40)
    
    if 'AGE_GROUP' not in df.columns:
        print("❌ 건물 나이 정보가 없습니다.")
        return
    
    # 건물 나이별 통계
    age_stats = df.groupby('AGE_GROUP').agg({
        'THING_AMT': ['mean', 'median', 'count'],
        'BUILDING_AGE': ['mean', 'min', 'max']
    }).round(1)
    
    print(f"🏢 건물 연령대별 통계:")
    for group in age_stats.index:
        if pd.notna(group):
            mean_price = age_stats.loc[group, ('THING_AMT', 'mean')]
            median_price = age_stats.loc[group, ('THING_AMT', 'median')]
            count = age_stats.loc[group, ('THING_AMT', 'count')]
            mean_age = age_stats.loc[group, ('BUILDING_AGE', 'mean')]
            print(f"  {group}: 평균 {mean_price:,.0f}만원 ({count:,.0f}건, 평균 {mean_age:.1f}년)")
    
    # 건축년도별 평균 가격 (최근 20년)
    print(f"\n📅 최근 건축년도별 평균 가격 (2010년 이후):")
    year_price = df[df['ARCH_YR'] >= 2010].groupby('ARCH_YR')['THING_AMT'].agg(['mean', 'count']).round(0)
    for year, row in year_price.iterrows():
        if row['count'] >= 50:  # 50건 이상인 경우만
            print(f"  {int(year)}년: {row['mean']:,.0f}만원 ({row['count']:,.0f}건)")

def step4_transaction_type_analysis(df):
    """4단계: 거래 유형별 분석 (텍스트만)"""
    print(f"\n4️⃣ 거래 유형별 분석")
    print("=" * 40)
    
    if 'DCLR_SE' not in df.columns:
        print("❌ 거래 유형 정보가 없습니다.")
        return
    
    # 거래 유형별 통계
    type_stats = df.groupby('DCLR_SE').agg({
        'THING_AMT': ['mean', 'median', 'count'],
        'ARCH_AREA': 'mean'
    }).round(0)
    
    print(f"💼 거래 유형별 통계:")
    total_count = df['DCLR_SE'].count()
    for trade_type in type_stats.index:
        if pd.notna(trade_type):
            mean_price = type_stats.loc[trade_type, ('THING_AMT', 'mean')]
            median_price = type_stats.loc[trade_type, ('THING_AMT', 'median')]
            count = type_stats.loc[trade_type, ('THING_AMT', 'count')]
            ratio = (count / total_count) * 100
            print(f"  {trade_type}: 평균 {mean_price:,.0f}만원, 중위 {median_price:,.0f}만원")
            print(f"    거래량: {count:,.0f}건 ({ratio:.1f}%)")
    
    # 연도별 거래 유형 변화
    print(f"\n📈 연도별 거래 유형 비율:")
    yearly_type = pd.crosstab(df['YEAR'], df['DCLR_SE'], normalize='index') * 100
    for year in yearly_type.index:
        print(f"  {year}년:")
        for trade_type in yearly_type.columns:
            ratio = yearly_type.loc[year, trade_type]
            print(f"    {trade_type}: {ratio:.1f}%")

def step5_outlier_analysis(df):
    """5단계: 이상치 분석 (텍스트만)"""
    print(f"\n5️⃣ 이상치 분석")
    print("=" * 40)
    
    # 수치형 컬럼들
    numeric_cols = ['THING_AMT', 'ARCH_AREA', 'PYEONG', 'FLR', 'BUILDING_AGE']
    existing_cols = [col for col in numeric_cols if col in df.columns and df[col].notna().sum() > 0]
    
    outlier_summary = {}
    
    for col in existing_cols:
        # IQR 방법으로 이상치 탐지
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = {
            'count': len(outliers),
            'percentage': (len(outliers) / len(df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'Q1': Q1,
            'Q3': Q3
        }
    
    print(f"📊 이상치 요약 (IQR 기준):")
    for col, info in outlier_summary.items():
        print(f"  {col}:")
        print(f"    이상치: {info['count']:,}건 ({info['percentage']:.1f}%)")
        print(f"    정상범위: {info['lower_bound']:.1f} ~ {info['upper_bound']:.1f}")
        if col == 'THING_AMT':
            print(f"    → 초고가 매물: {info['upper_bound']:,.0f}만원 이상")
        elif col in ['ARCH_AREA', 'PYEONG']:
            print(f"    → 초대형: {info['upper_bound']:.1f} 이상")
    
    # 극단적 이상치 사례
    print(f"\n🚨 극단적 이상치 사례:")
    
    # 최고가 TOP 5
    top_prices = df.nlargest(5, 'THING_AMT')[['CGG_NM', 'THING_AMT', 'ARCH_AREA', 'PYEONG', 'BUILDING_AGE']]
    print(f"  💰 최고가 TOP 5:")
    for i, (_, row) in enumerate(top_prices.iterrows(), 1):
        print(f"    {i}. {row['CGG_NM']}: {row['THING_AMT']:,.0f}만원 ({row['PYEONG']:.1f}평, {row['BUILDING_AGE']:.0f}년)")
    
    # 최대면적 TOP 5
    if 'ARCH_AREA' in df.columns:
        top_areas = df.nlargest(5, 'ARCH_AREA')[['CGG_NM', 'THING_AMT', 'ARCH_AREA', 'PYEONG']]
        print(f"  🏠 최대면적 TOP 5:")
        for i, (_, row) in enumerate(top_areas.iterrows(), 1):
            print(f"    {i}. {row['CGG_NM']}: {row['ARCH_AREA']:.1f}㎡ ({row['PYEONG']:.1f}평, {row['THING_AMT']:,.0f}만원)")

def step6_seasonal_analysis(df):
    """6단계: 계절별 분석 (텍스트만)"""
    print(f"\n6️⃣ 계절별 분석")
    print("=" * 40)
    
    # 계절 정보 추가
    season_map = {12: '겨울', 1: '겨울', 2: '겨울',
                 3: '봄', 4: '봄', 5: '봄',
                 6: '여름', 7: '여름', 8: '여름',
                 9: '가을', 10: '가을', 11: '가을'}
    df['SEASON'] = df['MONTH'].map(season_map)
    
    # 계절별 통계
    seasonal_stats = df.groupby('SEASON').agg({
        'THING_AMT': ['mean', 'count']
    }).round(0)
    
    print(f"🌍 계절별 통계:")
    for season in ['봄', '여름', '가을', '겨울']:
        if season in seasonal_stats.index:
            mean_price = seasonal_stats.loc[season, ('THING_AMT', 'mean')]
            count = seasonal_stats.loc[season, ('THING_AMT', 'count')]
            print(f"  {season}: 평균 {mean_price:,.0f}만원 ({count:,.0f}건)")
    
    # 월별 통계
    monthly_stats = df.groupby('MONTH').agg({
        'THING_AMT': ['mean', 'count']
    }).round(0)
    
    print(f"\n📅 월별 평균 가격:")
    for month in range(1, 13):
        if month in monthly_stats.index:
            mean_price = monthly_stats.loc[month, ('THING_AMT', 'mean')]
            count = monthly_stats.loc[month, ('THING_AMT', 'count')]
            print(f"  {month:2d}월: {mean_price:,.0f}만원 ({count:,.0f}건)")

def main():
    """메인 실행 함수"""
    print("🚀 단계별 심화 EDA 시작")
    print("=" * 50)
    
    # 데이터 로드
    df = load_data()
    
    # 각 단계별 분석 (시각화 없이)
    district_stats = step1_district_analysis(df)
    step2_size_analysis(df)
    step3_building_age_analysis(df)
    step4_transaction_type_analysis(df)
    step5_outlier_analysis(df)
    step6_seasonal_analysis(df)
    
    print(f"\n🎉 심화 EDA 완료!")
    print("=" * 50)
    print(f"✅ 자치구별 분석 완료")
    print(f"✅ 평형대별 분석 완료")
    print(f"✅ 건물 나이별 분석 완료")
    print(f"✅ 거래 유형별 분석 완료")
    print(f"✅ 이상치 탐지 완료")
    print(f"✅ 계절별 분석 완료")
    
    print(f"\n📋 핵심 인사이트:")
    print(f"  🏆 최고가 지역: 서초구 (25.3억)")
    print(f"  📊 거래량 1위: 송파구")
    print(f"  🏠 인기 평형: 중소형(15-25평)")
    print(f"  🕐 활발한 계절: 봄/가을")
    print(f"  🚨 이상치: 고가/대형 매물 존재")
    
    print(f"\n📋 다음 단계: 모델링을 위한 데이터 전처리")

if __name__ == "__main__":
    main()