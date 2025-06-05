"""
Seoul Apartment Price Prediction - Data Preprocessor
서울 아파트 실거래가 데이터 전처리 및 EDA 모듈
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from pathlib import Path
import sys

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
plt.rcParams['axes.unicode_minus'] = False

class SeoulApartmentPreprocessor:
    def __init__(self, data_path="data/raw/20250604_182224_seoul_real_estate.csv"):
        """
        데이터 전처리기 초기화
        """
        self.data_path = data_path
        self.df = None
        self.train_data = None
        self.test_data = None
        self.feature_columns = []
        
        print("🏠 Seoul Apartment Price Prediction - 데이터 전처리 시작")
        print("=" * 60)
        
    def load_data(self):
        """데이터 로드"""
        try:
            self.df = pd.read_csv(self.data_path, encoding='utf-8-sig')
            print(f"✅ 데이터 로드 완료: {len(self.df):,}건")
            print(f"📊 컬럼 수: {len(self.df.columns)}개")
            
            # 기본 컬럼 정보 출력
            print(f"📋 컬럼 목록:")
            for i, col in enumerate(self.df.columns, 1):
                print(f"  {i:2d}. {col}")
            
            # 날짜 컬럼 변환
            if 'CTRT_DAY' in self.df.columns:
                self.df['CTRT_DAY'] = pd.to_datetime(self.df['CTRT_DAY'])
                # 🎯 중요: 실제 계약일 기준으로 연도 설정
                self.df['YEAR'] = self.df['CTRT_DAY'].dt.year
                self.df['MONTH'] = self.df['CTRT_DAY'].dt.month
                self.df['QUARTER'] = self.df['CTRT_DAY'].dt.quarter
                
                print(f"\n📅 계약일 기준 연도 분포:")
                year_dist = self.df['YEAR'].value_counts().sort_index()
                for year, count in year_dist.items():
                    print(f"  {year}년: {count:,}건")
            
            # 🔧 데이터 정리: 2022-2025년 데이터만 사용
            original_len = len(self.df)
            self.df = self.df[(self.df['YEAR'] >= 2022) & (self.df['YEAR'] <= 2025)].copy()
            filtered_len = len(self.df)
            
            print(f"\n🔧 데이터 필터링:")
            print(f"  원본: {original_len:,}건")
            print(f"  필터링 후 (2022-2025): {filtered_len:,}건")
            print(f"  제외된 데이터: {original_len - filtered_len:,}건")
            
            # 수치형 컬럼 변환
            numeric_cols = ['THING_AMT', 'ARCH_AREA', 'LAND_AREA', 'FLR', 'ARCH_YR', 'PYEONG', 'PRICE_PER_PYEONG']
            for col in numeric_cols:
                if col in self.df.columns:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            return True
        except Exception as e:
            print(f"❌ 데이터 로드 실패: {e}")
            return False
    
    def basic_info(self):
        """기본 데이터 정보"""
        if self.df is None:
            print("❌ 데이터가 로드되지 않았습니다.")
            return
        
        print("\n📋 데이터 기본 정보")
        print("-" * 40)
        
        # 기간 정보
        print(f"📅 데이터 기간: {self.df['CTRT_DAY'].min().strftime('%Y-%m-%d')} ~ {self.df['CTRT_DAY'].max().strftime('%Y-%m-%d')}")
        
        # 실제 연도 확인
        actual_years = sorted(self.df['YEAR'].unique())
        print(f"📊 실제 포함된 연도: {actual_years}")
        
        # 연도별 분포
        print(f"\n📊 연도별 거래 건수:")
        year_counts = self.df['YEAR'].value_counts().sort_index()
        total_count = 0
        for year, count in year_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {year}년: {count:,}건 ({percentage:.1f}%)")
            total_count += count
        
        print(f"  총합: {total_count:,}건")
        
        # 지역별 분포 (상위 10개)
        print(f"\n📍 자치구별 거래 건수 (상위 10개):")
        district_counts = self.df['CGG_NM'].value_counts().head(10)
        for district, count in district_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {district}: {count:,}건 ({percentage:.1f}%)")
        
        # 가격 기본 통계
        print(f"\n💰 거래금액 기본 통계:")
        price_stats = self.df['THING_AMT'].describe()
        print(f"  평균: {price_stats['mean']:,.0f}만원")
        print(f"  중위수: {price_stats['50%']:,.0f}만원")
        print(f"  최저: {price_stats['min']:,.0f}만원")
        print(f"  최고: {price_stats['max']:,.0f}만원")
        print(f"  표준편차: {price_stats['std']:,.0f}만원")
        
        # 월별 분포 (계절성 확인)
        print(f"\n📅 월별 거래 건수:")
        month_counts = self.df['MONTH'].value_counts().sort_index()
        for month, count in month_counts.items():
            print(f"  {month:2d}월: {count:,}건")
        
    def data_quality_check(self):
        """데이터 품질 체크"""
        print("\n🔍 데이터 품질 체크")
        print("-" * 40)
        
        # 결측치 체크
        missing_data = self.df.isnull().sum()
        missing_percent = (missing_data / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            '결측치_개수': missing_data,
            '결측치_비율(%)': missing_percent
        })
        
        missing_df = missing_df[missing_df['결측치_개수'] > 0].sort_values('결측치_개수', ascending=False)
        
        if not missing_df.empty:
            print("⚠️ 결측치 발견:")
            for col, row in missing_df.iterrows():
                print(f"  {col}: {row['결측치_개수']:,}개 ({row['결측치_비율(%)']:.1f}%)")
        else:
            print("✅ 결측치 없음")
        
        # 이상치 체크 (IQR 방법)
        Q1 = self.df['THING_AMT'].quantile(0.25)
        Q3 = self.df['THING_AMT'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = self.df[(self.df['THING_AMT'] < lower_bound) | (self.df['THING_AMT'] > upper_bound)]
        print(f"\n📊 거래금액 이상치:")
        print(f"  이상치 건수: {len(outliers):,}건 ({len(outliers)/len(self.df)*100:.1f}%)")
        print(f"  정상 범위: {lower_bound:,.0f}만원 ~ {upper_bound:,.0f}만원")
        
        return missing_df, outliers
    
    def feature_engineering(self):
        """피처 엔지니어링"""
        print("\n🔧 피처 엔지니어링")
        print("-" * 40)
        
        # 기존 크롤링에서 생성된 파생변수들 확인
        existing_features = []
        if 'PYEONG' in self.df.columns:
            existing_features.append('PYEONG')
        if 'PRICE_PER_PYEONG' in self.df.columns:
            existing_features.append('PRICE_PER_PYEONG')
        if 'PYEONG_GROUP' in self.df.columns:
            existing_features.append('PYEONG_GROUP')
        if 'ARCH_DECADE' in self.df.columns:
            existing_features.append('ARCH_DECADE')
        if 'PRICE_EUK' in self.df.columns:
            existing_features.append('PRICE_EUK (억원단위)')
        
        if existing_features:
            print(f"✅ 이미 존재하는 파생변수: {', '.join(existing_features)}")
        
        # 새로운 피처들 생성
        # 1. 건물 나이 (계약연도 기준)
        self.df['BUILDING_AGE'] = self.df['YEAR'] - self.df['ARCH_YR']
        
        # 2. 평방미터당 가격
        self.df['PRICE_PER_SQM'] = self.df['THING_AMT'] / self.df['ARCH_AREA']
        
        # 3. 계절 정보
        season_map = {
            12: '겨울', 1: '겨울', 2: '겨울',
            3: '봄', 4: '봄', 5: '봄',
            6: '여름', 7: '여름', 8: '여름',
            9: '가을', 10: '가을', 11: '가을'
        }
        self.df['SEASON'] = self.df['MONTH'].map(season_map)
        
        # 4. 고층/저층 구분
        self.df['FLOOR_GROUP'] = pd.cut(
            self.df['FLR'],
            bins=[0, 5, 10, 20, float('inf')],
            labels=['저층(1-5층)', '중층(6-10층)', '고층(11-20층)', '초고층(21층이상)']
        )
        
        # 5. 상세 건물 나이 구간
        self.df['AGE_GROUP'] = pd.cut(
            self.df['BUILDING_AGE'],
            bins=[0, 5, 10, 20, 30, float('inf')],
            labels=['5년이하', '6-10년', '11-20년', '21-30년', '30년초과']
        )
        
        # 6. 거래 유형 (직거래/중개거래)
        if 'DCLR_SE' in self.df.columns:
            self.df['IS_DIRECT_TRADE'] = (self.df['DCLR_SE'] == '직거래').astype(int)
        
        # 7. 가격 구간 세분화
        price_percentiles = self.df['THING_AMT'].quantile([0.25, 0.5, 0.75])
        self.df['PRICE_QUARTILE'] = pd.cut(
            self.df['THING_AMT'],
            bins=[0, price_percentiles[0.25], price_percentiles[0.5], price_percentiles[0.75], float('inf')],
            labels=['하위25%', '중하위25%', '중상위25%', '상위25%']
        )
        
        # 8. 지역 프리미엄 (구별 평균가격 대비)
        district_avg_price = self.df.groupby('CGG_NM')['THING_AMT'].transform('mean')
        self.df['DISTRICT_PRICE_RATIO'] = self.df['THING_AMT'] / district_avg_price
        
        print(f"✅ 피처 엔지니어링 완료")
        print(f"📊 총 {len(self.df.columns)}개 컬럼")
        
        # 새로 생성된 피처들
        new_features = [
            'BUILDING_AGE', 'PRICE_PER_SQM', 'SEASON', 'FLOOR_GROUP', 
            'AGE_GROUP', 'IS_DIRECT_TRADE', 'PRICE_QUARTILE', 'DISTRICT_PRICE_RATIO'
        ]
        print(f"🆕 새로 생성된 피처: {', '.join(new_features)}")
        
        return self.df
    
    def visualize_trends(self, save_plots=False):
        """기본 트렌드 시각화 - 실제 데이터만 (X축 강제 고정)"""
        print("\n📈 데이터 시각화")
        print("-" * 40)
        
        # 실제 데이터 연도 범위 확인
        actual_years = sorted(self.df['YEAR'].unique())
        print(f"📊 실제 데이터 연도: {actual_years}")
        
        # 1. 연도별 통계 계산 (실제 데이터만)
        yearly_stats = self.df.groupby('YEAR').agg({
            'THING_AMT': ['mean', 'count'],
            'PRICE_PER_PYEONG': 'mean'
        }).round(0)
        yearly_stats.columns = ['평균거래가', '거래건수', '평균평당가']
        
        # 2. 4개 차트 생성
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 차트 1: 연도별 평균 거래가 (X축 완전 고정)
        years = list(actual_years)  # 실제 연도만
        prices = [yearly_stats.loc[year, '평균거래가'] for year in years]
        
        axes[0,0].plot(range(len(years)), prices, marker='o', linewidth=3, markersize=10, color='#2E86AB')
        axes[0,0].set_title('연도별 평균 거래가 변화', fontsize=14, fontweight='bold')
        axes[0,0].set_ylabel('거래가 (만원)')
        axes[0,0].set_xlabel('연도')
        axes[0,0].grid(True, alpha=0.3)
        # 핵심: X축을 인덱스로 설정하고 라벨만 연도로
        axes[0,0].set_xticks(range(len(years)))
        axes[0,0].set_xticklabels(years)
        axes[0,0].set_xlim(-0.5, len(years)-0.5)
        
        # 값 표시
        for i, (year, price) in enumerate(zip(years, prices)):
            axes[0,0].annotate(f'{price:,.0f}만원', 
                             (i, price), textcoords="offset points", 
                             xytext=(0,10), ha='center', fontsize=9)
        
        # 차트 2: 연도별 거래량 (X축 완전 고정)
        counts = [yearly_stats.loc[year, '거래건수'] for year in years]
        
        bars = axes[0,1].bar(range(len(years)), counts, color='#F18F01', alpha=0.8, width=0.5)
        axes[0,1].set_title('연도별 거래량 변화', fontsize=14, fontweight='bold')
        axes[0,1].set_ylabel('거래 건수')
        axes[0,1].set_xlabel('연도')
        axes[0,1].set_xticks(range(len(years)))
        axes[0,1].set_xticklabels(years)
        axes[0,1].set_xlim(-0.5, len(years)-0.5)
        
        # 값 표시
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            axes[0,1].text(i, height + height*0.01,
                         f'{count:,}건', ha='center', va='bottom', fontsize=9)
        
        # 차트 3: 구별 평균 거래가 (상위 10개)
        district_price = self.df.groupby('CGG_NM')['THING_AMT'].mean().sort_values(ascending=True).tail(10)
        
        bars = axes[1,0].barh(range(len(district_price)), district_price.values, color='#A23B72')
        axes[1,0].set_yticks(range(len(district_price)))
        axes[1,0].set_yticklabels(district_price.index)
        axes[1,0].set_title('자치구별 평균 거래가 (상위 10개)', fontsize=14, fontweight='bold')
        axes[1,0].set_xlabel('평균 거래가 (만원)')
        
        # 값 표시
        for i, (bar, price) in enumerate(zip(bars, district_price.values)):
            axes[1,0].text(price + price*0.01, i, f'{price:,.0f}만원', 
                         ha='left', va='center', fontsize=9)
        
        # 차트 4: 평수별 평균 거래가
        if 'PYEONG_GROUP' in self.df.columns:
            # 기존 PYEONG_GROUP 사용
            pyeong_price = self.df.groupby('PYEONG_GROUP')['THING_AMT'].mean().dropna()
        else:
            # 새로 생성
            self.df['PYEONG_GROUP_SIMPLE'] = pd.cut(
                self.df['PYEONG'], 
                bins=[0, 10, 15, 20, 25, 30, 40, float('inf')],
                labels=['10평미만', '10-15평', '15-20평', '20-25평', '25-30평', '30-40평', '40평이상']
            )
            pyeong_price = self.df.groupby('PYEONG_GROUP_SIMPLE')['THING_AMT'].mean().dropna()
        
        if not pyeong_price.empty:
            bars = axes[1,1].bar(range(len(pyeong_price)), pyeong_price.values, color='#C73E1D')
            axes[1,1].set_xticks(range(len(pyeong_price)))
            axes[1,1].set_xticklabels(pyeong_price.index, rotation=45)
            axes[1,1].set_title('평수별 평균 거래가', fontsize=14, fontweight='bold')
            axes[1,1].set_ylabel('평균 거래가 (만원)')
            
            # 값 표시
            for i, (bar, price) in enumerate(zip(bars, pyeong_price.values)):
                height = bar.get_height()
                axes[1,1].text(i, height + height*0.01,
                             f'{price:,.0f}', ha='center', va='bottom', fontsize=9, rotation=90)
        
        plt.tight_layout()
        
        if save_plots:
            Path('outputs/figures').mkdir(parents=True, exist_ok=True)
            plt.savefig('outputs/figures/basic_trends.png', dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # 실제 데이터 요약 출력
        print(f"\n📊 연도별 데이터 요약:")
        for year in years:
            year_data = self.df[self.df['YEAR'] == year]
            avg_price = year_data['THING_AMT'].mean()
            count = len(year_data)
            print(f"  {year}년: {count:,}건, 평균 {avg_price:,.0f}만원")
            
        print(f"\n📈 연도별 증감률:")
        for i in range(1, len(years)):
            prev_year = years[i-1]
            curr_year = years[i]
            prev_price = prices[i-1]
            curr_price = prices[i]
            change_rate = ((curr_price - prev_price) / prev_price) * 100
            direction = "📈" if change_rate > 0 else "📉"
            print(f"  {prev_year}→{curr_year}: {direction} {change_rate:+.1f}%")
        
    def correlation_analysis(self):
        """상관관계 분석"""
        print("\n🔍 상관관계 분석")
        print("-" * 40)
        
        # 수치형 변수들 선택
        numeric_cols = [
            'THING_AMT', 'ARCH_AREA', 'LAND_AREA', 'FLR', 'ARCH_YR',
            'PYEONG', 'BUILDING_AGE', 'PRICE_PER_PYEONG', 'PRICE_PER_SQM',
            'YEAR', 'MONTH', 'QUARTER'
        ]
        
        # 존재하는 컬럼만 선택
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        
        correlation_matrix = self.df[available_cols].corr()
        
        # 히트맵 생성
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # 상삼각 마스크
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdYlBu_r', 
                   center=0,
                   square=True, 
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title('변수간 상관관계 분석', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()
        
        # 거래가와 강한 상관관계를 가진 변수들
        price_corr = correlation_matrix['THING_AMT'].abs().sort_values(ascending=False)
        print(f"\n💰 거래가(THING_AMT)와 상관관계가 높은 변수들:")
        for var, corr in price_corr.head(10).items():
            if var != 'THING_AMT':
                print(f"  {var}: {corr:.3f}")
        
        return correlation_matrix
    
    def prepare_modeling_data(self):
        """모델링을 위한 데이터 준비"""
        print("\n🤖 모델링 데이터 준비")
        print("-" * 40)
        
        # 데이터 타입 확인
        print("📊 주요 컬럼 데이터 타입:")
        key_columns = ['RCPT_YR', 'YEAR', 'THING_AMT', 'ARCH_AREA', 'FLR', 'BUILDING_AGE']
        for col in key_columns:
            if col in self.df.columns:
                print(f"  {col}: {self.df[col].dtype}")
        
        # ⚠️ 중요: 실제 계약일 기준으로 분할
        year_column = 'YEAR'  # CTRT_DAY에서 추출한 연도 사용
        
        self.train_data = self.df[self.df[year_column] <= 2024].copy()  # 학습용 (2022-2024)
        self.test_data = self.df[self.df[year_column] == 2025].copy()   # 테스트용 (2025)
        
        print(f"📚 학습 데이터 (2022-2024): {len(self.train_data):,}건")
        print(f"🧪 테스트 데이터 (2025): {len(self.test_data):,}건")
        print(f"⚠️  주의: 2025년 데이터는 모델 검증용으로만 사용!")
        
        # 학습 데이터 연도별 분포
        train_year_dist = self.train_data[year_column].value_counts().sort_index()
        print(f"\n📊 학습 데이터 연도별 분포 (2022-2024):")
        for year, count in train_year_dist.items():
            percentage = (count / len(self.train_data)) * 100
            print(f"  {year}년: {count:,}건 ({percentage:.1f}%)")
        
        # 모델링용 피처 선택 (결측치가 많은 컬럼 제외)
        self.feature_columns = [
            # 기본 정보
            'CGG_CD', 'STDG_CD',
            # 건물 특성 (핵심)
            'ARCH_AREA', 'FLR', 'ARCH_YR', 'BUILDING_AGE',
            # 기존 파생 변수
            'PYEONG',
            # 시간 정보 (2025년 제외하고 학습)
            'YEAR', 'MONTH', 'QUARTER',
            # 새로 생성한 피처들
            'PRICE_PER_SQM', 'DISTRICT_PRICE_RATIO'
        ]
        
        # 조건부 추가할 컬럼들
        
        # 1. LAND_AREA: 결측치가 적고 0이 아닌 값이 많으면 추가
        if 'LAND_AREA' in self.df.columns:
            land_area_missing_rate = self.df['LAND_AREA'].isnull().sum() / len(self.df)
            land_area_nonzero_rate = (self.df['LAND_AREA'] > 0).sum() / len(self.df)
            if land_area_missing_rate < 0.05 and land_area_nonzero_rate > 0.3:
                self.feature_columns.append('LAND_AREA')
                print(f"  ✅ LAND_AREA 추가 (결측률: {land_area_missing_rate:.1%}, 유효값: {land_area_nonzero_rate:.1%})")
            else:
                print(f"  ❌ LAND_AREA 제외 (결측률: {land_area_missing_rate:.1%}, 유효값: {land_area_nonzero_rate:.1%})")
        
        # 2. 거래 유형 관련 피처
        if 'IS_DIRECT_TRADE' in self.df.columns:
            self.feature_columns.append('IS_DIRECT_TRADE')
            print(f"  ✅ IS_DIRECT_TRADE 추가")
        elif 'DCLR_SE' in self.df.columns:
            # DCLR_SE에서 직거래 여부 파생변수 생성
            self.df['IS_DIRECT_TRADE'] = (self.df['DCLR_SE'] == '직거래').astype(int)
            self.feature_columns.append('IS_DIRECT_TRADE')
            print(f"  ✅ IS_DIRECT_TRADE 생성 및 추가")
        
        # 3. 중개업소 지역 정보 (결측치가 적으면)
        if 'OPBIZ_RESTAGNT_SGG_NM' in self.df.columns:
            opbiz_missing_rate = self.df['OPBIZ_RESTAGNT_SGG_NM'].isnull().sum() / len(self.df)
            if opbiz_missing_rate < 0.1:  # 10% 미만이면 사용
                # 중개업소가 같은 구인지 여부
                self.df['SAME_DISTRICT_BROKER'] = (
                    self.df['CGG_NM'] == self.df['OPBIZ_RESTAGNT_SGG_NM']
                ).astype(int)
                self.feature_columns.append('SAME_DISTRICT_BROKER')
                print(f"  ✅ SAME_DISTRICT_BROKER 생성 및 추가 (결측률: {opbiz_missing_rate:.1%})")
            else:
                print(f"  ❌ 중개업소 정보 제외 (결측률: {opbiz_missing_rate:.1%})")
        
        # 제외된 고결측 컬럼들 알림
        excluded_columns = ['RGHT_SE', 'RTRCN_DAY']  # 95% 이상 결측
        print(f"\n❌ 제외된 고결측 컬럼: {excluded_columns}")
        
        # 존재하는 피처만 선택
        available_features = [col for col in self.feature_columns if col in self.df.columns]
        excluded_features = [col for col in self.feature_columns if col not in self.df.columns]
        
        if excluded_features:
            print(f"⚠️ 존재하지 않는 피처: {excluded_features}")
        
        self.feature_columns = available_features
        
        # 범주형 변수 정의
        categorical_columns = ['CGG_CD', 'STDG_CD']
        
        # 결측치 처리 (학습 데이터 기준으로!)
        print("\n🔧 결측치 처리 (학습 데이터 기준):")
        for col in self.feature_columns:
            train_missing = self.train_data[col].isnull().sum()
            test_missing = self.test_data[col].isnull().sum()
            
            if train_missing > 0 or test_missing > 0:
                if col in categorical_columns:
                    # 범주형: 학습 데이터의 최빈값으로 채우기
                    mode_val = self.train_data[col].mode()
                    mode_val = mode_val[0] if not mode_val.empty else 'Unknown'
                    self.train_data[col] = self.train_data[col].fillna(mode_val)
                    self.test_data[col] = self.test_data[col].fillna(mode_val)
                    print(f"  {col}: {train_missing + test_missing}개 → '{mode_val}'로 대체")
                else:
                    # 수치형: 학습 데이터의 중위수로 채우기
                    median_val = self.train_data[col].median()
                    self.train_data[col] = self.train_data[col].fillna(median_val)
                    self.test_data[col] = self.test_data[col].fillna(median_val)
                    print(f"  {col}: {train_missing + test_missing}개 → {median_val}로 대체")
        
        print(f"\n✅ 선택된 피처: {len(self.feature_columns)}개")
        print(f"📋 피처 목록:")
        for i, col in enumerate(self.feature_columns, 1):
            col_type = "범주형" if col in categorical_columns else "수치형"
            print(f"  {i:2d}. {col} ({col_type})")
        
        # 데이터 저장
        self.save_processed_data()
        
        return self.train_data, self.test_data, self.feature_columns
        print("\n🔧 결측치 처리:")
        for col in self.feature_columns:
            train_missing = self.train_data[col].isnull().sum()
            test_missing = self.test_data[col].isnull().sum()
            
            if train_missing > 0 or test_missing > 0:
                if col in categorical_columns:
                    # 범주형: 최빈값으로 채우기
                    mode_val = self.train_data[col].mode()
                    mode_val = mode_val[0] if not mode_val.empty else 'Unknown'
                    self.train_data[col] = self.train_data[col].fillna(mode_val)
                    self.test_data[col] = self.test_data[col].fillna(mode_val)
                    print(f"  {col}: {train_missing + test_missing}개 → '{mode_val}'로 대체")
                else:
                    # 수치형: 중위수로 채우기
                    median_val = self.train_data[col].median()
                    self.train_data[col] = self.train_data[col].fillna(median_val)
                    self.test_data[col] = self.test_data[col].fillna(median_val)
                    print(f"  {col}: {train_missing + test_missing}개 → {median_val}로 대체")
        
        print(f"\n✅ 선택된 피처: {len(self.feature_columns)}개")
        print(f"📋 피처 목록:")
        for i, col in enumerate(self.feature_columns, 1):
            col_type = "범주형" if col in categorical_columns else "수치형"
            print(f"  {i:2d}. {col} ({col_type})")
        
        # 데이터 저장
        self.save_processed_data()
        
        return self.train_data, self.test_data, self.feature_columns
    
    def save_processed_data(self):
        """전처리된 데이터 저장"""
        try:
            # 폴더 생성
            Path("data/processed").mkdir(exist_ok=True)
            
            # 데이터 저장
            self.train_data.to_csv('data/processed/train_data_2022_2024.csv', 
                                 index=False, encoding='utf-8-sig')
            self.test_data.to_csv('data/processed/test_data_2025.csv', 
                                index=False, encoding='utf-8-sig')
            
            # 피처 정보 저장
            feature_info = {
                'feature_columns': self.feature_columns,
                'target_column': 'THING_AMT',
                'categorical_columns': ['CGG_CD', 'STDG_CD']
            }
            
            import json
            with open('data/processed/feature_info.json', 'w', encoding='utf-8') as f:
                json.dump(feature_info, f, ensure_ascii=False, indent=2)
            
            print(f"\n💾 전처리된 데이터 저장 완료!")
            print(f"  📁 학습 데이터: data/processed/train_data_2022_2024.csv")
            print(f"  📁 테스트 데이터: data/processed/test_data_2025.csv")
            print(f"  📁 피처 정보: data/processed/feature_info.json")
            
        except Exception as e:
            print(f"❌ 데이터 저장 실패: {e}")
    
    def generate_summary_report(self):
        """전체 요약 보고서"""
        print("\n" + "="*60)
        print("📊 서울 아파트 실거래가 데이터 분석 요약 보고서")
        print("="*60)
        
        print(f"\n📈 데이터 개요:")
        print(f"  • 총 거래건수: {len(self.df):,}건")
        print(f"  • 분석기간: {self.df['CTRT_DAY'].min().strftime('%Y-%m-%d')} ~ {self.df['CTRT_DAY'].max().strftime('%Y-%m-%d')}")
        print(f"  • 실제 연도: {sorted(self.df['YEAR'].unique())}")
        print(f"  • 대상지역: 서울시 {self.df['CGG_NM'].nunique()}개 자치구")
        
        print(f"\n💰 거래가 현황:")
        print(f"  • 평균 거래가: {self.df['THING_AMT'].mean():,.0f}만원")
        print(f"  • 중위 거래가: {self.df['THING_AMT'].median():,.0f}만원")
        print(f"  • 최고 거래가: {self.df['THING_AMT'].max():,.0f}만원")
        print(f"  • 평균 평당가: {self.df['PRICE_PER_PYEONG'].mean():,.0f}만원/평")
        
        print(f"\n🏠 건물 특성:")
        print(f"  • 평균 전용면적: {self.df['ARCH_AREA'].mean():.1f}㎡")
        print(f"  • 평균 평수: {self.df['PYEONG'].mean():.1f}평")
        print(f"  • 평균 건물나이: {self.df['BUILDING_AGE'].mean():.1f}년")
        print(f"  • 평균 층수: {self.df['FLR'].mean():.1f}층")
        
        # 연도별 증가율 (2022-2025 전체)
        actual_years = sorted(self.df['YEAR'].unique())
        if len(actual_years) > 1:
            yearly_avg = self.df.groupby('YEAR')['THING_AMT'].mean()
            growth_rates = yearly_avg.pct_change() * 100
            print(f"\n📈 연도별 평균 거래가 변화율:")
            for year, rate in growth_rates.dropna().items():
                direction = "📈" if rate > 0 else "📉"
                print(f"  • {year}년: {direction} {rate:+.1f}%")
        
        print(f"\n🎯 모델링 준비:")
        print(f"  • 학습 데이터: {len(self.train_data):,}건 (2022-2024)")
        print(f"  • 테스트 데이터: {len(self.test_data):,}건 (2025)")
        print(f"  • 피처 개수: {len(self.feature_columns)}개")
        
        print(f"\n✅ 전처리 완료! 다음 단계: 모델 학습")


def main():
    """메인 실행 함수"""
    # 전처리기 생성
    preprocessor = SeoulApartmentPreprocessor()
    
    # 1. 데이터 로드
    if not preprocessor.load_data():
        return None
    
    # 2. 기본 정보 확인
    preprocessor.basic_info()
    
    # 3. 데이터 품질 체크
    missing_df, outliers = preprocessor.data_quality_check()
    
    # 4. 피처 엔지니어링
    processed_df = preprocessor.feature_engineering()
    
    # 5. 시각화
    preprocessor.visualize_trends()
    
    # 6. 상관관계 분석
    correlation_matrix = preprocessor.correlation_analysis()
    
    # 7. 모델링 데이터 준비
    train_data, test_data, feature_columns = preprocessor.prepare_modeling_data()
    
    # 8. 요약 보고서
    preprocessor.generate_summary_report()
    
    return preprocessor

if __name__ == "__main__":
    preprocessor = main()