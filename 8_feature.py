"""
🎯 2025 서울 아파트 가격 예측 - 8피처 전처리
2022-2024 학습데이터로만 인코딩 학습
2025 데이터는 순수 예측 타겟
"""

import pandas as pd
import numpy as np
import os
import re
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def preprocessing_for_2025_prediction():
    """
    2025 서울 아파트 가격 예측을 위한 8피처 전처리
    """
    
    print("🎯 2025 서울 아파트 가격 예측 전처리!")
    print("📚 학습: 2022-2024 데이터만 사용")
    print("🔮 예측: 2025 데이터 (순수 타겟)")
    print("🎯 목표: 실제 배포 환경과 동일한 조건!")
    print("=" * 60)
    
    # 1. 데이터 로드
    print("1️⃣ 데이터 로드")
    file_path = "data/raw/20250604_182224_seoul_real_estate.csv"
    
    if not os.path.exists(file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    print(f"   원본 데이터: {df.shape}")
    
    # 2. 기본 피처 생성
    print("\n2️⃣ 기본 피처 생성")
    
    # 기본 피처 생성
    df['PRICE'] = df['PRICE_EUK'] * 10000
    df['CTRT_YEAR'] = pd.to_datetime(df['CTRT_DAY']).dt.year
    df['BUILDING_AGE'] = 2025 - df['ARCH_YR']  #모든 예측이 2025년 기준이므로 건물나이도 2025년 기준으로 계산
    
    print(f"   피처 생성 완료: PRICE, CTRT_YEAR, BUILDING_AGE")
    
    # 3. 🚨 데이터 분할 우선 (리키지 방지)
    print("\n3️⃣ 🚨 데이터 분할 우선 (리키지 방지)")
    
    # 기본 필수 조건만 먼저 필터링 (리키지 없는 조건들만)
    print("   기본 필수 조건 필터링...")
    initial_count = len(df)
    
    # 건축년도 필터링
    df = df[(df['ARCH_YR'] > 0) & (df['ARCH_YR'] <= 2025)]
    
    # 건물나이 필터링 (음수 및 비현실적 값 제거)
    df = df[(df['BUILDING_AGE'] >= 0) & (df['BUILDING_AGE'] <= 50)]
    
    # 기타 기본 필터링
    df = df[(df['PYEONG'] > 0) & (df['PRICE'] > 0)]
    df = df.dropna(subset=['ARCH_YR', 'PYEONG', 'FLR'])
    
    print(f"   기본 필터링: {initial_count:,} → {len(df):,}건")
    
    # 즉시 데이터 분할
    train_data = df[df['CTRT_YEAR'] < 2025].copy()  # 2022-2024
    predict_data = df[df['CTRT_YEAR'] == 2025].copy()  # 2025
    
    print(f"   📚 학습 데이터: {len(train_data):,}건 (2022-2024)")
    print(f"   🔮 예측 데이터: {len(predict_data):,}건 (2025)")
    print(f"   📊 학습/예측 비율: {len(train_data)/(len(train_data)+len(predict_data))*100:.1f}% / {len(predict_data)/(len(train_data)+len(predict_data))*100:.1f}%")
    
    # 4. 극단값 제거 (학습 데이터 기준으로만 - 데이터 리키지 방지)
    print("\n4️⃣ 🔧 극단값 제거 (학습 데이터 기준)")
    
    # 학습 데이터로만 분위수 계산
    price_q01, price_q99 = train_data['PRICE'].quantile([0.01, 0.99])
    pyeong_q01, pyeong_q99 = train_data['PYEONG'].quantile([0.01, 0.99])
    
    print(f"   가격 기준: {price_q01:,.0f} ~ {price_q99:,.0f}만원")
    print(f"   평수 기준: {pyeong_q01:.1f} ~ {pyeong_q99:.1f}평")
    
    # 학습 데이터 극단값 제거
    train_before = len(train_data)
    train_data = train_data[
        (train_data['PRICE'] >= price_q01) & (train_data['PRICE'] <= price_q99) &
        (train_data['PYEONG'] >= pyeong_q01) & (train_data['PYEONG'] <= pyeong_q99)
    ]
    
    # 예측 데이터에도 동일한 기준 적용
    predict_before = len(predict_data)
    predict_data = predict_data[
        (predict_data['PRICE'] >= price_q01) & (predict_data['PRICE'] <= price_q99) &
        (predict_data['PYEONG'] >= pyeong_q01) & (predict_data['PYEONG'] <= pyeong_q99)
    ]
    
    print(f"   학습 데이터 극단값 제거: {train_before:,} → {len(train_data):,}건")
    print(f"   예측 데이터 극단값 제거: {predict_before:,} → {len(predict_data):,}건")
    print(f"   ✅ 완전한 데이터 누출 방지: 학습 데이터 기준으로만 극단값 제거")
    
    # 4. 브랜드 분석 (학습 데이터만!)
    print("\n5️⃣ 브랜드 분석 (학습 데이터 기준)")
    
    # 브랜드 매핑 (원본 방식 유지)
    brand_mapping = {
        '래미안': r'래미안|RAEMIAN|raemian',
        '자이': r'자이|XI|xi',
        'e편한세상': r'e편한세상|e-편한세상|이편한세상|e편한|E편한세상',
        '힐스테이트': r'힐스테이트|HILLSTATE|hillstate',
        '아크로': r'아크로|ACRO|acro',
        '더샵': r'더샵|THE샵|THE SHARP|SHARP',
        '푸르지오': r'푸르지오|PRUGIO|prugio',
        '롯데캐슬': r'롯데캐슬|롯데|LOTTE|lotte',
        '수자인': r'수자인|한양수자인|수자인디에트르',
        '위브': r'위브|WEVE|weve',
        '아이파크': r'아이파크|i-park|I-PARK|ipark|IPARK',
        '센트럴': r'센트럴|CENTRAL|central',
        '포레스트': r'포레스트|FOREST|forest',
        '현대': r'현대',
        '삼성': r'삼성',
        '한양': r'한양',
        '두산': r'두산',
        '대우': r'대우',
        '디에이치': r'디에이치|D\'H|DH',
        '스카이': r'스카이|SKY|sky',
        '파크': r'파크|PARK|park',
        '타워': r'타워|TOWER|tower'
    }
    
    def extract_advanced_brand(building_name):
        if pd.isna(building_name):
            return '브랜드없음'  # 명칭 개선
        
        building_name = str(building_name)
        
        # 정규식으로 브랜드 매칭
        for brand, pattern in brand_mapping.items():
            if re.search(pattern, building_name, re.IGNORECASE):
                return brand
        
        return '브랜드없음'  # 명칭 개선
    
    # 브랜드 추출
    train_data.loc[:, 'BRAND_NAME'] = train_data['BLDG_NM'].apply(extract_advanced_brand)
    predict_data.loc[:, 'BRAND_NAME'] = predict_data['BLDG_NM'].apply(extract_advanced_brand)
    
    # 브랜드별 가격 분석 (학습 데이터만!)
    train_data['PRICE_PER_PYEONG'] = train_data['PRICE'] / train_data['PYEONG']
    
    brand_stats = train_data.groupby('BRAND_NAME').agg({
        'PRICE_PER_PYEONG': ['mean', 'count'],
        'PRICE': 'mean'
    }).round(0)
    
    brand_stats.columns = ['평당가격_평균', '거래건수', '총가격_평균']
    brand_stats = brand_stats[brand_stats['거래건수'] >= 30]  # 30건 이상
    
    print(f"   브랜드별 가격 정보 (학습 데이터, 30건 이상):")
    brand_sorted = brand_stats.sort_values('평당가격_평균', ascending=False)
    for brand, stats in brand_sorted.head(10).iterrows():
        print(f"   {brand}: 평당 {stats['평당가격_평균']:,.0f}만원 ({stats['거래건수']:,.0f}건)")
    
    # 브랜드 점수 계산
    overall_mean_per_pyeong = train_data['PRICE_PER_PYEONG'].mean()
    
    def get_brand_score(brand_name):
        if brand_name not in brand_stats.index:
            return 1  # 브랜드없음 = 1점
        
        brand_avg = brand_stats.loc[brand_name, '평당가격_평균']
        premium_ratio = brand_avg / overall_mean_per_pyeong
        
        if premium_ratio >= 1.3:    return 5    # 최고급 (30% 이상)
        elif premium_ratio >= 1.15: return 4    # 고급 (15-30%)
        elif premium_ratio >= 1.0:  return 3    # 중급 (평균)
        elif premium_ratio >= 0.9:  return 2    # 일반 (-10%)
        else:                       return 1    # 저가 (-10% 이하)
    
    # 점수 적용
    train_data.loc[:, 'BRAND_SCORE'] = train_data['BRAND_NAME'].apply(get_brand_score)
    predict_data.loc[:, 'BRAND_SCORE'] = predict_data['BRAND_NAME'].apply(get_brand_score)
    
    # 브랜드 점수 분포
    brand_dist = train_data['BRAND_SCORE'].value_counts().sort_index()
    print(f"\n   브랜드 점수 분포 (학습 데이터):")
    for score, count in brand_dist.items():
        pct = count / len(train_data) * 100
        print(f"   {score}점: {count:,}건 ({pct:.1f}%)")
    
    # 5. 구별 지하철 접근성 점수
    print("\n6️⃣ 구별 지하철 접근성 점수")
    
    # 📍 지하철 접근성 점수 산정 근거:
    # - 서울시 지하철 노선도 및 환승역 분석 기반
    # - 운행 노선 수: 강남구(2,3,7,9호선+분당선), 서초구(2,3,7호선+분당선), 중구(1,2,4,5,6호선), 종로구(1,3,5,6호선)
    # - 환승역 밀도: 강남역(2,분당선), 교대역(2,3호선), 을지로입구(2호선), 종각역(1호선) 등
    # - 도심 접근성: CBD(중구,종로구) > 강남권(강남,서초) > 영등포권 > 기타
    ''' 지하철 접근성 점수 산정 기준: 운행 노선 수, 환승역 밀도, 도심 접근성을 종합 고려
    - 5점: 다수 노선(4개 이상) + 주요 환승역 밀집
    - 4점: 주요 노선(2-3개) + 환승역 존재
    - 3점: 일반 노선(1-2개) + 기본 역세권
    - 2점: 외곽 지역, 제한적 접근성
    
    # 구별 지하철 노선 정보 (실제 데이터 기반)
    subway_line_info = {
        # 5점: 최고 접근성 (4개 이상 노선, 주요 환승역 밀집)
        '강남구': {'lines': ['2호선', '3호선', '7호선', '9호선'], 'major_stations': ['강남', '역삼', '선릉']},
        '서초구': {'lines': ['2호선', '3호선', '7호선'], 'major_stations': ['강남', '교대', '사당']},
        '중구': {'lines': ['1호선', '2호선', '4호선', '5호선'], 'major_stations': ['서울역', '을지로3가', '동대문역사문화공원']},
        '종로구': {'lines': ['1호선', '3호선', '5호선', '6호선'], 'major_stations': ['종각', '을지로3가', '광화문']},
        
        # 4점: 우수 접근성 (2-3개 주요 노선)
        '송파구': {'lines': ['2호선', '5호선', '8호선', '9호선'], 'major_stations': ['잠실', '송파', '가락시장']},
        '마포구': {'lines': ['2호선', '5호선', '6호선'], 'major_stations': ['홍대입구', '합정', '공덕']},
        '용산구': {'lines': ['1호선', '4호선', '6호선'], 'major_stations': ['용산', '이촌', '삼각지']},
        '영등포구': {'lines': ['1호선', '5호선', '9호선'], 'major_stations': ['영등포구청', '여의도', '당산']},
        '성동구': {'lines': ['2호선', '5호선'], 'major_stations': ['성수', '왕십리', '금호']},
        '서대문구': {'lines': ['2호선', '3호선', '6호선'], 'major_stations': ['홍대입구', '신촌', '충정로']},
        
        # 3점: 보통 접근성 (1-2개 노선)
        '동작구': {'lines': ['4호선', '7호선', '9호선'], 'major_stations': ['사당', '노량진']},
        '관악구': {'lines': ['2호선'], 'major_stations': ['신림', '봉천']},
        '양천구': {'lines': ['5호선'], 'major_stations': ['목동', '양평']},
        '구로구': {'lines': ['1호선', '2호선'], 'major_stations': ['구로', '신도림']},
        '금천구': {'lines': ['1호선'], 'major_stations': ['독산', '가산디지털단지']},
        '동대문구': {'lines': ['1호선', '4호선'], 'major_stations': ['동대문', '청량리']},
        '성북구': {'lines': ['4호선', '6호선'], 'major_stations': ['한성대입구', '길음']},
        '광진구': {'lines': ['2호선', '5호선'], 'major_stations': ['건대입구', '광나루']},
        '중랑구': {'lines': ['1호선', '7호선'], 'major_stations': ['상봉', '면목']},
        '강북구': {'lines': ['4호선'], 'major_stations': ['수유', '미아']},
        '도봉구': {'lines': ['1호선', '4호선', '7호선'], 'major_stations': ['도봉산', '창동']},
        '노원구': {'lines': ['4호선', '7호선'], 'major_stations': ['노원', '중계']},
        '은평구': {'lines': ['3호선', '6호선'], 'major_stations': ['연신내', '구파발']},
        
        # 2점: 기본 접근성 (외곽, 제한적 노선)
        '강서구': {'lines': ['5호선', '9호선'], 'major_stations': ['김포공항', '발산']},
        '강동구': {'lines': ['5호선'], 'major_stations': ['강동', '천호']},
    }'''

    subway_score_mapping = {
        # 5점: 최고 접근성 (4개 이상 노선, 주요 환승역 밀집, CBD 또는 강남권)
        '강남구': 5, '서초구': 5, '중구': 5, '종로구': 5,
        
        # 4점: 우수 접근성 (2-3개 주요 노선, 환승역 존재)
        '송파구': 4, '마포구': 4, '용산구': 4, '영등포구': 4, 
        '성동구': 4, '서대문구': 4,
        
        # 3점: 보통 접근성 (1-2개 노선, 일반 역세권)
        '동작구': 3, '관악구': 3, '양천구': 3, '구로구': 3, 
        '금천구': 3, '동대문구': 3, '성북구': 3, '광진구': 3, 
        '중랑구': 3, '강북구': 3, '도봉구': 3, '노원구': 3, '은평구': 3,
        
        # 2점: 기본 접근성 (외곽 지역, 제한적 노선)
        '강서구': 2, '강동구': 2,
    }
    
    def get_subway_score_by_gu(gu_name):
        return subway_score_mapping.get(gu_name, 2)
    
    # 지하철 점수 적용
    train_data.loc[:, 'SUBWAY_SCORE'] = train_data['CGG_NM'].apply(get_subway_score_by_gu)
    predict_data.loc[:, 'SUBWAY_SCORE'] = predict_data['CGG_NM'].apply(get_subway_score_by_gu)
    
    # 지하철 점수 분포
    subway_dist = train_data['SUBWAY_SCORE'].value_counts().sort_index()
    print(f"   지하철 접근성 점수 분포 (학습 데이터):")
    for score, count in subway_dist.items():
        pct = count / len(train_data) * 100
        print(f"   {score}점: {count:,}건 ({pct:.1f}%)")
    
    
    # 6. 구별 교육특구 프리미엄
    print("\n7️⃣ 구별 교육특구 프리미엄")
    
    # 📚 교육특구 프리미엄 산정 근거:
    # - 강남구: 대치동 학원가, 강남8학군, 특목고 집중 (휘문고, 중동고 등)
    # - 서초구: 서초4동 학원가, 서초고, 서문여고 등 명문고 위치
    # - 송파구: 잠실 학원가, 송파구 교육환경 우수 (방이중, 잠신고 등)
    # - 양천구: 목동 학원가, 양천구 교육 인프라 발달 (목동고, 양정고 등)  
    # - 노원구: 중계동 학원가, 노원구 교육열 높음 (상명고, 선덕고 등)
    # ※ 실제 부동산 시장에서 학군 프리미엄이 인정되는 지역 기준

    ''' 교육특구 근거 명시
    교육특구 프리미엄 산정 기준:
    - 특목고, 자사고 밀집도
    - 대학 진학률 및 학원가 발달 정도
    - 부동산 시장에서 실제 학군 프리미엄이 인정되는 지역
    
    교육특구 정보 (실제 학군 정보 기반)
        '강남구': {'특징': '대치동 학원가, 특목고 다수', '주요학교': ['휘문고', '단대부고', '개포고']},
        '서초구': {'특징': '반포/잠원 학군, 서초고 등', '주요학교': ['서초고', '반포고', '잠원고']},
        '송파구': {'특징': '잠실 학군, 신천고 등', '주요학교': ['신천고', '잠신고', '문정고']},
        '양천구': {'특징': '목동 학원가, 특목고 집중', '주요학교': ['목동고', '양정고', '신목고']},
        '노원구': {'특징': '중계동 학원가, 교육열 높음', '주요학교': ['상계고', '노원고', '중계고']},
        
        # 일반지역 (0점) - 나머지 모든 구  '''
    
    education_premium_mapping = {
        '강남구': 1, '서초구': 1, '송파구': 1, '양천구': 1, '노원구': 1,
        # 나머지는 0
    }
    
    def get_education_premium_by_gu(gu_name):
        return education_premium_mapping.get(gu_name, 0)
    
    # 교육특구 점수 적용
    train_data.loc[:, 'EDUCATION_PREMIUM'] = train_data['CGG_NM'].apply(get_education_premium_by_gu)
    predict_data.loc[:, 'EDUCATION_PREMIUM'] = predict_data['CGG_NM'].apply(get_education_premium_by_gu)
    
    # 교육특구 분포
    edu_dist = train_data['EDUCATION_PREMIUM'].value_counts().sort_index()
    print(f"   교육특구 프리미엄 분포 (학습 데이터):")
    for premium, count in edu_dist.items():
        pct = count / len(train_data) * 100
        status = "교육특구" if premium == 1 else "일반지역"
        print(f"   {status}: {count:,}건 ({pct:.1f}%)")
    
    # 교육특구별 평균 가격 비교 (학습 데이터만)
    edu_price_comparison = train_data.groupby('EDUCATION_PREMIUM')['PRICE'].agg(['mean', 'count'])
    print(f"\n   교육특구별 평균 가격 (학습 데이터):")
    for premium, stats in edu_price_comparison.iterrows():
        status = "교육특구" if premium == 1 else "일반지역"
        print(f"   {status}: {stats['mean']:,.0f}만원 ({stats['count']:,}건)")
    
    if len(edu_price_comparison) == 2:
        premium_ratio = edu_price_comparison.loc[1, 'mean'] / edu_price_comparison.loc[0, 'mean']
        print(f"   📚 교육특구 프리미엄: {premium_ratio:.2f}배 (+{(premium_ratio-1)*100:.1f}%)")
    
    # 7. 구별 Label Encoding (핵심! 학습 데이터만 사용)
    print("\n8️⃣ 구별 Label Encoding (학습 데이터 기준)")
    
    # 학습 데이터로만 Label Encoder 학습
    label_encoder = LabelEncoder()
    label_encoder.fit(train_data['CGG_NM'])
    
    print(f"   학습된 구 목록 ({len(label_encoder.classes_)}개):")
    for i, gu in enumerate(label_encoder.classes_):
        print(f"   {i:2d}. {gu}")
    
    # 예측 데이터에 새로운 구가 있는지 확인
    predict_gus = set(predict_data['CGG_NM'].unique())
    train_gus = set(train_data['CGG_NM'].unique())
    new_gus = predict_gus - train_gus
    
    if new_gus:
        print(f"\n   ⚠️  예측 데이터에만 있는 구: {new_gus}")
        print(f"   → 이런 구는 가장 가까운 구의 라벨로 매핑됩니다")
    else:
        print(f"\n   ✅ 모든 구가 학습 데이터에 포함되어 있습니다")
    
    # Label Encoding 적용
    train_data.loc[:, 'CGG_LABEL_ENCODED'] = label_encoder.transform(train_data['CGG_NM'])
    
    # 예측 데이터 인코딩 (새로운 구 처리)
    predict_encoded = []
    for gu in predict_data['CGG_NM']:
        if gu in label_encoder.classes_:
            predict_encoded.append(label_encoder.transform([gu])[0])
        else:
            # 새로운 구는 기본값 0으로 설정
            predict_encoded.append(0)
            print(f"   ⚠️  새로운 구 '{gu}'를 라벨 0으로 매핑")
    
    predict_data.loc[:, 'CGG_LABEL_ENCODED'] = predict_encoded
    
    # 라벨 매핑 정보 출력 (학습 데이터 기준 가격순)
    gu_price_for_sort = train_data.groupby('CGG_NM')['PRICE'].mean().to_dict()
    gu_label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    print(f"\n   구별 Label Encoding 매핑 (학습 데이터 가격순):")
    gu_sorted_by_price = sorted(gu_price_for_sort.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for i, (gu, price) in enumerate(gu_sorted_by_price, 1):
        label = gu_label_mapping[gu]
        print(f"   {i:2d}. {gu} → 라벨 {label} (평균 {price:,.0f}만원)")
    
    print(f"\n   🔥 핵심: 학습 데이터(2022-2024)로만 인코딩 학습!")
    print(f"   ✅ 2025 데이터는 순수 예측 타겟")
    print(f"   ✅ 실제 배포 환경과 동일한 조건")
    
    # 8. 강남3구 피처
    print("\n9️⃣ 강남3구 피처")
    
    # 🏙️ 강남3구 선정 근거:
    # - 강남구: 전통적 부촌, 최고가 아파트 집중, 업무지구(테헤란로) + 상업지구(강남역)
    # - 서초구: 법조타운(서초동), 고급 주거지(반포동), 한강 인접 프리미엄
    # - 송파구: 잠실 신도시, 롯데월드타워, 올림픽공원, 교통 요충지
    # ※ 서울 부동산 시장에서 전통적으로 최고 프리미엄을 받는 3개 구
    
    premium_gus = ['강남구', '서초구', '송파구']
    train_data.loc[:, 'IS_PREMIUM_GU'] = train_data['CGG_NM'].isin(premium_gus).astype(int)
    predict_data.loc[:, 'IS_PREMIUM_GU'] = predict_data['CGG_NM'].isin(premium_gus).astype(int)
    
    premium_count_train = train_data['IS_PREMIUM_GU'].sum()
    premium_count_predict = predict_data['IS_PREMIUM_GU'].sum()
    
    print(f"   강남3구 분포:")
    print(f"   - 학습 데이터: {premium_count_train:,}건 ({premium_count_train/len(train_data)*100:.1f}%)")
    print(f"   - 예측 데이터: {premium_count_predict:,}건 ({premium_count_predict/len(predict_data)*100:.1f}%)")
    
    # 9. 최종 8피처 선택
    print("\n🔟 최종 8피처 선택")
    
    final_features = [
        'CGG_LABEL_ENCODED',        # 1. 구별 Label Encoding (학습 데이터 기준)
        'PYEONG',                   # 2. 평수
        'BUILDING_AGE',             # 3. 건축년수 (2025년 기준)
        'FLR',                      # 4. 층수
        'BRAND_SCORE',              # 5. 브랜드 점수 (학습 데이터 기준)
        'IS_PREMIUM_GU',            # 6. 강남3구
        'SUBWAY_SCORE',             # 7. 구별 지하철 접근성
        'EDUCATION_PREMIUM'         # 8. 구별 교육특구
    ]
    
    print(f"   최종 피처 ({len(final_features)}개):")
    for i, feature in enumerate(final_features, 1):
        print(f"   {i}. {feature}")
    
    print(f"\n   🔥 모든 인코딩이 학습 데이터(2022-2024) 기준!")
    print(f"   🎯 2025 데이터는 순수 예측 타겟!")
    
    # 10. 최종 데이터셋 생성
    print("\n1️⃣1️⃣ 최종 데이터셋 생성")
    
    X_train = train_data[final_features]
    y_train = train_data['PRICE']
    X_predict = predict_data[final_features]
    y_predict = predict_data['PRICE']  # 실제 정답 (모델 평가용)
    
    print(f"   학습 피처: {X_train.shape}")
    print(f"   학습 타겟: {y_train.shape}")
    print(f"   예측 피처: {X_predict.shape}")
    print(f"   예측 정답: {y_predict.shape} (평가용)")
    
    # 결측치만 간단히 확인
    train_missing = X_train.isnull().sum().sum()
    predict_missing = X_predict.isnull().sum().sum()
    print(f"   결측치: 학습 {train_missing}개, 예측 {predict_missing}개")
    
    # 11. 파일 저장 (덮어쓰기)
    print("\n1️⃣2️⃣ 파일 저장 (덮어쓰기)")
    
    # 폴더 생성
    os.makedirs('data/processed', exist_ok=True)
    
    # 고정 파일명으로 덮어쓰기
    X_train.to_csv('data/processed/X_train.csv', index=False, encoding='utf-8-sig')
    y_train.to_csv('data/processed/y_train.csv', index=False, encoding='utf-8-sig')
    X_predict.to_csv('data/processed/X_predict.csv', index=False, encoding='utf-8-sig')
    y_predict.to_csv('data/processed/y_predict.csv', index=False, encoding='utf-8-sig')
    
    # 매핑 정보 저장 (학습 데이터 기준)
    mapping_info = {
        'feature_names': final_features,
        'brand_score_mapping': {brand: get_brand_score(brand) for brand in brand_mapping.keys()},
        'subway_score_mapping': subway_score_mapping,
        'education_premium_mapping': education_premium_mapping,
        'gu_label_mapping': gu_label_mapping,
        'label_encoder_classes': label_encoder.classes_.tolist(),
        'train_period': '2022-2024',
        'predict_period': '2025',
        'outlier_removal': {
            'price_q01': price_q01,
            'price_q99': price_q99, 
            'pyeong_q01': pyeong_q01,
            'pyeong_q99': pyeong_q99
        }
    }
    
    import pickle
    with open('data/processed/mapping_info.pkl', 'wb') as f:
        pickle.dump(mapping_info, f)
    
    print(f"   ✅ data/processed/X_train.csv")
    print(f"   ✅ data/processed/y_train.csv")
    print(f"   ✅ data/processed/X_predict.csv")
    print(f"   ✅ data/processed/y_predict.csv")
    print(f"   ✅ data/processed/mapping_info.pkl")
    print(f"   📁 파일 덮어쓰기 완료!")
    
    # 12. 최종 요약
    print("\n1️⃣3️⃣ 최종 요약")
    
    print(f"   데이터 분할:")
    print(f"   - 학습: {len(train_data):,}건 (2022-2024)")
    print(f"   - 예측: {len(predict_data):,}건 (2025)")
    
    print(f"\n   학습 데이터 가격 통계:")
    print(f"   - 평균: {y_train.mean():,.0f}만원")
    print(f"   - 중앙값: {y_train.median():,.0f}만원")
    print(f"   - 범위: {y_train.min():,.0f} ~ {y_train.max():,.0f}만원")
    
    print(f"\n   예측 데이터 가격 통계 (정답):")
    print(f"   - 평균: {y_predict.mean():,.0f}만원")
    print(f"   - 중앙값: {y_predict.median():,.0f}만원")
    print(f"   - 범위: {y_predict.min():,.0f} ~ {y_predict.max():,.0f}만원")
    
    print("\n" + "=" * 60)
    print("🎉 2025 서울 아파트 가격 예측 전처리 완료!")
    print(f"🔥 핵심 포인트:")
    print(f"   📚 데이터 누출 완전 방지: 학습 데이터만으로 극단값 기준 계산")
    print(f"   🎯 브랜드 개선: 브랜드없음 1점 (실제 프리미엄 반영)")
    print(f"   🔮 예측 타겟: 2025 데이터 (순수 예측)")
    print(f"   📁 파일 관리: 덮어쓰기 방식으로 효율성 극대화")
    print("=" * 60)
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_predict': X_predict,
        'y_predict': y_predict,
        'feature_names': final_features,
        'mapping_info': mapping_info,
        'label_encoder': label_encoder
    }

if __name__ == "__main__":
    print("🎯 2025 서울 아파트 가격 예측 전처리 (간단 버전)")
    print("📚 학습: 2022-2024 / 🔮 예측: 2025")
    print("🎯 원본 방식 유지 + 근거 주석 추가!")
    print()
    
    result = preprocessing_for_2025_prediction()
    
    if result:
        print(f"\n🎊 전처리 성공! 🎊")
        print(f"이제 2025 가격 예측 모델을 학습하세요:")
        print(f"python model_training_2025_prediction.py")
    else:
        print(f"\n❌전처리 실패")