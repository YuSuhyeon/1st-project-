"""
서울 아파트 가격 예측 - 심화 EDA
분포 및 범주형 변수 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

class AdvancedEDA:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        
    def load_data(self):
        """데이터 로드 및 기본 전처리"""
        print("📊 심화 EDA 시작")
        print("=" * 50)
        
        # 데이터 로드
        self.df = pd.read_csv(self.data_path)
        
        # 날짜 처리
        self.df['CTRT_DAY'] = pd.to_datetime(self.df['CTRT_DAY'])
        self.df['YEAR'] = self.df['CTRT_DAY'].dt.year
        self.df['MONTH'] = self.df['CTRT_DAY'].dt.month
        self.df['QUARTER'] = self.df['CTRT_DAY'].dt.quarter
        
        # 2022-2025년 데이터만 사용
        self.df = self.df[(self.df['YEAR'] >= 2022) & (self.df['YEAR'] <= 2025)].copy()
        
        # 수치형 변환
        numeric_cols = ['THING_AMT', 'ARCH_AREA', 'FLR', 'ARCH_YR']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # 파생변수 생성
        if 'ARCH_AREA' in self.df.columns:
            self.df['PYEONG'] = self.df['ARCH_AREA'] * 0.3025
            
            # 평형대 그룹 생성
            self.df['PYEONG_GROUP'] = pd.cut(
                self.df['PYEONG'], 
                bins=[0, 15, 25, 35, 50, 100],
                labels=['소형(15평미만)', '중소형(15-25평)', '중형(25-35평)', '대형(35-50평)', '초대형(50평+)']
            )
        
        if 'ARCH_YR' in self.df.columns:
            self.df['BUILDING_AGE'] = 2025 - self.df['ARCH_YR']
            
            # 건물 연령대 그룹
            self.df['AGE_GROUP'] = pd.cut(
                self.df['BUILDING_AGE'],
                bins=[0, 5, 10, 20, 30, 100],
                labels=['신축(5년이하)', '준신축(5-10년)', '보통(10-20년)', '노후(20-30년)', '매우노후(30년+)']
            )
        
        # 계절 정보
        season_map = {12: '겨울', 1: '겨울', 2: '겨울',
                     3: '봄', 4: '봄', 5: '봄',
                     6: '여름', 7: '여름', 8: '여름',
                     9: '가을', 10: '가을', 11: '가을'}
        self.df['SEASON'] = self.df['MONTH'].map(season_map)
        
        print(f"✅ 데이터 로드 완료: {len(self.df):,}건")
        print(f"📊 분석 기간: {self.df['YEAR'].min()}년 ~ {self.df['YEAR'].max()}년")
        
    def analyze_districts(self):
        """1. 자치구별 분석"""
        print(f"\n1️⃣ 자치구별 가격 분석")
        print("-" * 40)
        
        # 자치구별 통계
        district_stats = self.df.groupby('CGG_NM').agg({
            'THING_AMT': ['mean', 'median', 'count', 'std'],
            'ARCH_AREA': 'mean',
            'BUILDING_AGE': 'mean'
        }).round(0)
        
        district_stats.columns = ['평균가격', '중위가격', '거래수', '가격표준편차', '평균면적', '평균건물나이']
        district_stats = district_stats.sort_values('평균가격', ascending=False)
        
        print(f"🏆 자치구별 평균 가격 TOP 10:")
        for i, (district, row) in enumerate(district_stats.head(10).iterrows(), 1):
            print(f"  {i:2d}. {district}: {row['평균가격']:,.0f}만원 ({row['거래수']:,.0f}건)")
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        
        # 자치구별 평균 가격 (TOP 15)
        top_districts = district_stats.head(15)
        bars = axes[0,0].barh(range(len(top_districts)), top_districts['평균가격'], color='steelblue')
        axes[0,0].set_yticks(range(len(top_districts)))
        axes[0,0].set_yticklabels(top_districts.index)
        axes[0,0].set_title('자치구별 평균 거래가 (TOP 15)', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('평균 거래가 (만원)')
        
        # 값 표시
        for i, (bar, price) in enumerate(zip(bars, top_districts['평균가격'])):
            axes[0,0].text(price + price*0.01, i, f'{price:,.0f}', 
                         ha='left', va='center', fontsize=9)
        
        # 자치구별 거래량 (TOP 15)
        top_volume = district_stats.sort_values('거래수', ascending=False).head(15)
        bars = axes[0,1].bar(range(len(top_volume)), top_volume['거래수'], color='orange', alpha=0.7)
        axes[0,1].set_xticks(range(len(top_volume)))
        axes[0,1].set_xticklabels(top_volume.index, rotation=45, ha='right')
        axes[0,1].set_title('자치구별 거래량 (TOP 15)', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('거래 건수')
        
        # 가격 vs 거래량 산점도
        axes[1,0].scatter(district_stats['거래수'], district_stats['평균가격'], 
                         alpha=0.6, s=60, color='red')
        axes[1,0].set_xlabel('거래 건수')
        axes[1,0].set_ylabel('평균 가격 (만원)')
        axes[1,0].set_title('자치구별 거래량 vs 평균가격', fontsize=14, fontweight='bold')
        
        # 상위 5개 구 라벨 표시
        for district, row in district_stats.head(5).iterrows():
            axes[1,0].annotate(district, (row['거래수'], row['평균가격']), 
                             xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        # 자치구별 가격 분산 (표준편차)
        top_std = district_stats.sort_values('가격표준편차', ascending=False).head(15)
        bars = axes[1,1].bar(range(len(top_std)), top_std['가격표준편차'], color='purple', alpha=0.7)
        axes[1,1].set_xticks(range(len(top_std)))
        axes[1,1].set_xticklabels(top_std.index, rotation=45, ha='right')
        axes[1,1].set_title('자치구별 가격 편차 (TOP 15)', fontsize=14, fontweight='bold')
        axes[1,1].set_ylabel('가격 표준편차 (만원)')
        
        plt.tight_layout()
        plt.show()
        
        return district_stats
    
    def analyze_size_groups(self):
        """2. 평형대별 분석"""
        print(f"\n2️⃣ 평형대별 가격 분석")
        print("-" * 40)
        
        if 'PYEONG_GROUP' not in self.df.columns:
            print("❌ 평형대 정보가 없습니다.")
            return
        
        # 평형대별 통계
        size_stats = self.df.groupby('PYEONG_GROUP').agg({
            'THING_AMT': ['mean', 'median', 'count'],
            'ARCH_AREA': ['mean', 'min', 'max'],
            'PYEONG': ['mean', 'min', 'max']
        }).round(1)
        
        print(f"📊 평형대별 통계:")
        for group in size_stats.index:
            if pd.notna(group):
                mean_price = size_stats.loc[group, ('THING_AMT', 'mean')]
                count = size_stats.loc[group, ('THING_AMT', 'count')]
                mean_pyeong = size_stats.loc[group, ('PYEONG', 'mean')]
                print(f"  {group}: 평균 {mean_price:,.0f}만원 ({count:,.0f}건, 평균 {mean_pyeong:.1f}평)")
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 평형대별 가격 박스플롯
        valid_data = self.df[self.df['PYEONG_GROUP'].notna()]
        sns.boxplot(data=valid_data, x='PYEONG_GROUP', y='THING_AMT', ax=axes[0,0])
        axes[0,0].set_title('평형대별 가격 분포 (Boxplot)', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('평형대')
        axes[0,0].set_ylabel('거래가 (만원)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 평형대별 평균 가격
        size_mean = valid_data.groupby('PYEONG_GROUP')['THING_AMT'].mean().sort_values()
        bars = axes[0,1].bar(range(len(size_mean)), size_mean.values, color='lightcoral')
        axes[0,1].set_xticks(range(len(size_mean)))
        axes[0,1].set_xticklabels(size_mean.index, rotation=45)
        axes[0,1].set_title('평형대별 평균 거래가', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('평균 거래가 (만원)')
        
        # 값 표시
        for i, price in enumerate(size_mean.values):
            axes[0,1].text(i, price + price*0.01, f'{price:,.0f}', 
                         ha='center', va='bottom', fontsize=10)
        
        # 평형대별 거래량
        size_count = valid_data['PYEONG_GROUP'].value_counts().sort_index()
        bars = axes[1,0].bar(range(len(size_count)), size_count.values, color='lightgreen')
        axes[1,0].set_xticks(range(len(size_count)))
        axes[1,0].set_xticklabels(size_count.index, rotation=45)
        axes[1,0].set_title('평형대별 거래량', fontsize=14, fontweight='bold')
        axes[1,0].set_ylabel('거래 건수')
        
        # 면적 vs 가격 산점도
        sample_data = valid_data.sample(n=min(5000, len(valid_data)), random_state=42)
        scatter = axes[1,1].scatter(sample_data['ARCH_AREA'], sample_data['THING_AMT'], 
                                  c=sample_data['PYEONG_GROUP'].cat.codes, 
                                  alpha=0.6, cmap='viridis')
        axes[1,1].set_xlabel('전용면적 (㎡)')
        axes[1,1].set_ylabel('거래가 (만원)')
        axes[1,1].set_title('면적 vs 가격 (평형대별)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_building_age(self):
        """3. 건물 나이별 분석"""
        print(f"\n3️⃣ 건물 나이별 가격 분석")
        print("-" * 40)
        
        if 'AGE_GROUP' not in self.df.columns:
            print("❌ 건물 나이 정보가 없습니다.")
            return
        
        # 건물 나이별 통계
        age_stats = self.df.groupby('AGE_GROUP').agg({
            'THING_AMT': ['mean', 'count'],
            'BUILDING_AGE': ['mean', 'min', 'max'],
            'ARCH_YR': ['mean', 'min', 'max']
        }).round(1)
        
        print(f"🏢 연령대별 평균 가격:")
        for group in age_stats.index:
            if pd.notna(group):
                mean_price = age_stats.loc[group, ('THING_AMT', 'mean')]
                count = age_stats.loc[group, ('THING_AMT', 'count')]
                mean_age = age_stats.loc[group, ('BUILDING_AGE', 'mean')]
                print(f"  {group}: 평균 {mean_price:,.0f}만원 ({count:,.0f}건, 평균 {mean_age:.1f}년)")
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        valid_data = self.df[self.df['AGE_GROUP'].notna()]
        
        # 건물 나이별 가격 박스플롯
        sns.boxplot(data=valid_data, x='AGE_GROUP', y='THING_AMT', ax=axes[0,0])
        axes[0,0].set_title('건물 연령대별 가격 분포', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('건물 연령대')
        axes[0,0].set_ylabel('거래가 (만원)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 건축년도별 평균 가격 추이
        year_price = self.df.groupby('ARCH_YR')['THING_AMT'].mean().sort_index()
        # 최근 20년만 표시
        recent_years = year_price[year_price.index >= 2005]
        axes[0,1].plot(recent_years.index, recent_years.values, marker='o', linewidth=2)
        axes[0,1].set_title('건축년도별 평균 거래가 추이 (2005년 이후)', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('건축년도')
        axes[0,1].set_ylabel('평균 거래가 (만원)')
        axes[0,1].grid(True, alpha=0.3)
        
        # 건물 나이 vs 가격 산점도
        sample_data = valid_data.sample(n=min(5000, len(valid_data)), random_state=42)
        axes[1,0].scatter(sample_data['BUILDING_AGE'], sample_data['THING_AMT'], 
                         alpha=0.5, color='brown')
        axes[1,0].set_xlabel('건물 나이 (년)')
        axes[1,0].set_ylabel('거래가 (만원)')
        axes[1,0].set_title('건물 나이 vs 거래가', fontsize=14, fontweight='bold')
        
        # 연령대별 거래량
        age_count = valid_data['AGE_GROUP'].value_counts().sort_index()
        bars = axes[1,1].bar(range(len(age_count)), age_count.values, color='navy', alpha=0.7)
        axes[1,1].set_xticks(range(len(age_count)))
        axes[1,1].set_xticklabels(age_count.index, rotation=45)
        axes[1,1].set_title('건물 연령대별 거래량', fontsize=14, fontweight='bold')
        axes[1,1].set_ylabel('거래 건수')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_transaction_type(self):
        """4. 거래 유형별 분석"""
        print(f"\n4️⃣ 거래 유형별 분석")
        print("-" * 40)
        
        if 'DCLR_SE' not in self.df.columns:
            print("❌ 거래 유형 정보가 없습니다.")
            return
        
        # 거래 유형별 통계
        type_stats = self.df.groupby('DCLR_SE').agg({
            'THING_AMT': ['mean', 'median', 'count'],
            'ARCH_AREA': 'mean'
        }).round(0)
        
        print(f"💼 거래 유형별 통계:")
        for trade_type in type_stats.index:
            if pd.notna(trade_type):
                mean_price = type_stats.loc[trade_type, ('THING_AMT', 'mean')]
                count = type_stats.loc[trade_type, ('THING_AMT', 'count')]
                print(f"  {trade_type}: 평균 {mean_price:,.0f}만원 ({count:,.0f}건)")
        
        # 시각화
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        valid_data = self.df[self.df['DCLR_SE'].notna()]
        
        # 거래 유형별 가격 박스플롯
        sns.boxplot(data=valid_data, x='DCLR_SE', y='THING_AMT', ax=axes[0])
        axes[0].set_title('거래 유형별 가격 분포', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('거래 유형')
        axes[0].set_ylabel('거래가 (만원)')
        
        # 거래 유형별 비율
        type_ratio = valid_data['DCLR_SE'].value_counts()
        wedges, texts, autotexts = axes[1].pie(type_ratio.values, labels=type_ratio.index, 
                                              autopct='%1.1f%%', startangle=90)
        axes[1].set_title('거래 유형별 비율', fontsize=14, fontweight='bold')
        
        # 연도별 거래 유형 변화
        yearly_type = pd.crosstab(valid_data['YEAR'], valid_data['DCLR_SE'], normalize='index') * 100
        yearly_type.plot(kind='bar', stacked=True, ax=axes[2])
        axes[2].set_title('연도별 거래 유형 변화', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('연도')
        axes[2].set_ylabel('비율 (%)')
        axes[2].legend(title='거래 유형', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2].tick_params(axis='x', rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    def detect_outliers(self):
        """5. 이상치 탐지"""
        print(f"\n5️⃣ 이상치 탐지")
        print("-" * 40)
        
        # 수치형 컬럼들
        numeric_cols = ['THING_AMT', 'ARCH_AREA', 'PYEONG', 'FLR', 'BUILDING_AGE']
        existing_cols = [col for col in numeric_cols if col in self.df.columns and self.df[col].notna().sum() > 0]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        outlier_summary = {}
        
        for i, col in enumerate(existing_cols[:6]):  # 최대 6개 컬럼
            # IQR 방법으로 이상치 탐지
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_summary[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(self.df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
            
            # 박스플롯
            axes[i].boxplot(self.df[col].dropna())
            axes[i].set_title(f'{col} 분포 및 이상치', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(col)
            
            # 이상치 정보 표시
            axes[i].text(0.02, 0.98, f'이상치: {len(outliers):,}건 ({(len(outliers)/len(self.df))*100:.1f}%)', 
                        transform=axes[i].transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 빈 서브플롯 제거
        for i in range(len(existing_cols), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()
        
        # 이상치 요약 출력
        print(f"\n📊 이상치 요약:")
        for col, info in outlier_summary.items():
            print(f"  {col}: {info['count']:,}건 ({info['percentage']:.1f}%)")
            if col == 'THING_AMT':
                print(f"    → 상한: {info['upper_bound']:,.0f}만원")
            elif col in ['ARCH_AREA', 'PYEONG']:
                print(f"    → 상한: {info['upper_bound']:.1f}")
        
        return outlier_summary
    
    def seasonal_analysis(self):
        """6. 계절별 분석"""
        print(f"\n6️⃣ 계절별 거래 분석")
        print("-" * 40)
        
        # 계절별 통계
        seasonal_stats = self.df.groupby('SEASON').agg({
            'THING_AMT': ['mean', 'count'],
            'ARCH_AREA': 'mean'
        }).round(0)
        
        print(f"🌍 계절별 평균 가격:")
        for season in ['봄', '여름', '가을', '겨울']:
            if season in seasonal_stats.index:
                mean_price = seasonal_stats.loc[season, ('THING_AMT', 'mean')]
                count = seasonal_stats.loc[season, ('THING_AMT', 'count')]
                print(f"  {season}: 평균 {mean_price:,.0f}만원 ({count:,.0f}건)")
        
        # 월별 분석
        monthly_stats = self.df.groupby('MONTH').agg({
            'THING_AMT': ['mean', 'count']
        }).round(0)
        
        # 시각화
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 계절별 가격 박스플롯
        sns.boxplot(data=self.df, x='SEASON', y='THING_AMT', ax=axes[0,0])
        axes[0,0].set_title('계절별 거래가 분포', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('계절')
        axes[0,0].set_ylabel('거래가 (만원)')
        
        # 월별 평균 가격
        months = list(range(1, 13))
        monthly_prices = [monthly_stats.loc[m, ('THING_AMT', 'mean')] if m in monthly_stats.index else 0 for m in months]
        axes[0,1].plot(months, monthly_prices, marker='o', linewidth=2, markersize=8)
        axes[0,1].set_title('월별 평균 거래가', fontsize=14, fontweight='bold')
        axes[0,1].set_xlabel('월')
        axes[0,1].set_ylabel('평균 거래가 (만원)')
        axes[0,1].set_xticks(months)
        axes[0,1].grid(True, alpha=0.3)
        
        # 월별 거래량
        monthly_counts = [monthly_stats.loc[m, ('THING_AMT', 'count')] if m in monthly_stats.index else 0 for m in months]
        bars = axes[1,0].bar(months, monthly_counts, color='skyblue', alpha=0.7)
        axes[1,0].set_title('월별 거래량', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('월')
        axes[1,0].set_ylabel('거래 건수')
        axes[1,0].set_xticks(months)
        
        # 연도별-월별 히트맵
        monthly_data = self.df.pivot_table(values='THING_AMT', index='YEAR', columns='MONTH', aggfunc='mean')
        sns.heatmap(monthly_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[1,1])
        axes[1,1].set_title('연도별-월별 평균 거래가 히트맵', fontsize=14, fontweight='bold')
        axes[1,1].set_xlabel('월')
        axes[1,1].set_ylabel('연도')
        
        plt.tight_layout()
        plt.show()
    
    def run_all_analysis(self):
        """전체 심화 분석 실행"""
        self.load_data()
        
        # 각 분석 실행
        district_stats = self.analyze_districts()
        self.analyze_size_groups()
        self.analyze_building_age()
        self.analyze_transaction_type()
        outlier_summary = self.detect_outliers()
        self.seasonal_analysis()
        
        # 분석 완료 메시지
        print(f"\n🎉 심화 EDA 완료!")
        print("=" * 50)
        print(f"✅ 자치구별 분석 완료")
        print(f"✅ 평형대별 분석 완료")
        print(f"✅ 건물 나이별 분석 완료")
        print(f"✅ 거래 유형별 분석 완료")
        print(f"✅ 이상치 탐지 완료")
        print(f"✅ 계절별 분석 완료")
        
        print(f"\n📋 다음 단계: 모델링을 위한 데이터 전처리")
        print(f"  - 범주형 변수 인코딩")
        print(f"  - 이상치 처리 결정")
        print(f"  - 학습/테스트 데이터 분할")

def main():
    # 심화 EDA 실행
    eda = AdvancedEDA("data/raw/20250604_182224_seoul_real_estate.csv")
    eda.run_all_analysis()

if __name__ == "__main__":
    main()