"""
2025 서울 아파트 가격 예측 웹 대시보드 (Random Forest 전용)
기존 UI 스타일 유지 - 최고 성능 모델만 사용
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
from pathlib import Path
from datetime import datetime
import time

# 페이지 설정
st.set_page_config(
    page_title="2025 서울 아파트 가격 예측 모델델",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ApartmentPricePredictor2025:
    def __init__(self):
        self.model = None
        self.mapping_info = {}
        self.model_name = ""
        self.model_loaded = False
        self.load_models()
        
    def load_models(self):
        """Random Forest 모델만 로드"""
        try:
            # models 폴더 확인
            models_dir = Path("models")
            if not models_dir.exists():
                st.error("❌ models 폴더가 없습니다!")
                self.create_dummy_model()
                return False
            
            # 매핑 정보 로드
            mapping_path = Path("data/processed/mapping_info.pkl")
            if mapping_path.exists():
                with open(mapping_path, 'rb') as f:
                    self.mapping_info = pickle.load(f)
                st.success("✅ 매핑 정보 로드 성공!")
            else:
                st.warning("⚠️ 매핑 정보가 없어 기본값을 사용합니다.")
                self.create_default_mapping()
            
            # Random Forest만 로드 (최고 성능)
            rf_path = models_dir / "random_forest_model.pkl"
            if rf_path.exists():
                try:
                    self.model = joblib.load(rf_path)
                    self.model_name = "Random Forest"
                    self.model_loaded = True
                    st.success(f"✅ {self.model_name} 모델 로드 성공! (R² 0.918, MAE 13,561만원)")
                    return True
                except Exception as e:
                    st.error(f"Random Forest 로드 실패: {e}")
            else:
                st.error("❌ random_forest_model.pkl 파일을 찾을 수 없습니다!")
            
            self.create_dummy_model()
            return False
            
        except Exception as e:
            st.error(f"❌ 모델 로드 실패: {e}")
            self.create_dummy_model()
            return False
    
    def create_default_mapping(self):
        """기본 매핑 정보 생성"""
        self.mapping_info = {
            'gu_label_mapping': {
                '강남구': 0, '강동구': 1, '강북구': 2, '강서구': 3, '관악구': 4,
                '광진구': 5, '구로구': 6, '금천구': 7, '노원구': 8, '도봉구': 9,
                '동대문구': 10, '동작구': 11, '마포구': 12, '서대문구': 13, '서초구': 14,
                '성동구': 15, '성북구': 16, '송파구': 17, '양천구': 18, '영등포구': 19,
                '용산구': 20, '은평구': 21, '종로구': 22, '중구': 23, '중랑구': 24
            },
            'subway_score_mapping': {
                '강남구': 5, '서초구': 5, '중구': 5, '종로구': 5,
                '송파구': 4, '마포구': 4, '용산구': 4, '영등포구': 4, '성동구': 4, '서대문구': 4,
                '동작구': 3, '관악구': 3, '양천구': 3, '구로구': 3, '금천구': 3,
                '동대문구': 3, '성북구': 3, '광진구': 3, '중랑구': 3, '강북구': 3,
                '도봉구': 3, '노원구': 3, '은평구': 3, '강서구': 2, '강동구': 2
            },
            'education_premium_mapping': {
                '강남구': 1, '서초구': 1, '송파구': 1, '양천구': 1, '노원구': 1
            }
        }
    
    def create_dummy_model(self):
        """더미 모델 생성 (모델 로드 실패시 대안)"""
        st.warning("⚠️ 저장된 모델을 찾을 수 없어 간단한 추정 모델을 사용합니다.")
        self.model_loaded = False
        self.model_name = "간단 추정 모델"
        self.create_default_mapping()
    
    def get_brand_score(self, brand_name):
        """브랜드별 점수 반환 (전처리 코드와 일치)"""
        brand_scores = {
            '기타 브랜드': 1, '래미안': 5, '자이': 5, 'e편한세상': 4, '힐스테이트': 4,
            '아크로': 4, '더샵': 4, '푸르지오': 4, '롯데캐슬': 4,
            '수자인': 3, '위브': 3, '아이파크': 3, '센트럴': 3,
            '현대': 3, '삼성': 3, '한양': 3, '두산': 3
        }
        return brand_scores.get(brand_name, 1)
    
    def get_subway_score(self, district):
        """구별 지하철 접근성 점수"""
        return self.mapping_info.get('subway_score_mapping', {}).get(district, 3)
    
    def get_education_premium(self, district):
        """구별 교육특구 여부"""
        return self.mapping_info.get('education_premium_mapping', {}).get(district, 0)
    
    def encode_district(self, district):
        """구별 Label Encoding"""
        return self.mapping_info.get('gu_label_mapping', {}).get(district, 0)
    
    def is_premium_gu(self, district):
        """강남3구 여부"""
        premium_gus = ['강남구', '서초구', '송파구']
        return 1 if district in premium_gus else 0
    
    def predict_price_dummy(self, inputs):
        """더미 예측 모델 (ML 모델 없을 때)"""
        # 기본 가격: 평당 4000만원
        base_price_per_pyeong = 4000
        
        # 평수 기반 기본 가격
        base_price = inputs['PYEONG'] * base_price_per_pyeong
        
        # 구별 조정
        district_multipliers = {
            '강남구': 2.8, '서초구': 2.6, '송파구': 2.2, '용산구': 2.4,
            '마포구': 2.1, '성동구': 1.9, '광진구': 1.8, '강동구': 1.8,
            '중구': 1.9, '종로구': 2.0, '영등포구': 1.9, '동작구': 1.7,
            '관악구': 1.6, '양천구': 1.8, '서대문구': 1.8, '성북구': 1.7,
            '강북구': 1.5, '도봉구': 1.4, '노원구': 1.5, '은평구': 1.6,
            '동대문구': 1.6, '중랑구': 1.4, '강서구': 1.5, '구로구': 1.4, '금천구': 1.3
        }
        
        district_multiplier = district_multipliers.get(inputs['CGG_NM'], 1.5)
        price = base_price * district_multiplier
        
        # 브랜드 조정
        brand_multiplier = 1 + (inputs['BRAND_SCORE'] - 2) * 0.1
        price *= brand_multiplier
        
        # 건물나이 조정 (신축일수록 비싸고 30년이상 재건축부터도 비싸짐 (기대심리 반영) - U자형 비선형 관계 )
        age_factor = max(0.7, 1 - (inputs['BUILDING_AGE'] * 0.01))
        price *= age_factor
        
        # 층수 조정 (중층이 선호)
        if 5 <= inputs['FLR'] <= 15:
            floor_factor = 1.1
        elif inputs['FLR'] < 3:
            floor_factor = 0.9
        else:
            floor_factor = 1.0
        price *= floor_factor
        
        # 지하철 접근성 조정
        subway_factor = 1 + (inputs['SUBWAY_SCORE'] - 3) * 0.05
        price *= subway_factor
        
        # 교육특구 프리미엄
        if inputs['EDUCATION_PREMIUM']:
            price *= 1.15
        
        return max(price, 10000)  # 최소 1억원
    
    def predict_price(self, inputs):
        """가격 예측"""
        try:
            if self.model_loaded and self.model:
                # 8피처 구성 (전처리 코드와 완전 일치)
                feature_names = [
                    'CGG_LABEL_ENCODED', 'PYEONG', 'BUILDING_AGE', 'FLR', 
                    'BRAND_SCORE', 'IS_PREMIUM_GU', 'SUBWAY_SCORE', 'EDUCATION_PREMIUM'
                ]
                
                features = [
                    inputs['CGG_LABEL_ENCODED'],    # 구별 Label Encoding
                    inputs['PYEONG'],               # 평수
                    inputs['BUILDING_AGE'],         # 건축년수 (2025년 기준)
                    inputs['FLR'],                  # 층수
                    inputs['BRAND_SCORE'],          # 브랜드 점수
                    inputs['IS_PREMIUM_GU'],        # 강남3구
                    inputs['SUBWAY_SCORE'],         # 지하철 접근성
                    inputs['EDUCATION_PREMIUM']     # 교육특구
                ]
                
                # DataFrame 생성 (컬럼명 포함으로 경고 해결)
                pred_data = pd.DataFrame([features], columns=feature_names)
                
                # Random Forest 예측 (스케일링 불필요)
                prediction = self.model.predict(pred_data)[0]
                
                # 최소값 보정
                prediction = max(prediction, 10000)
                
                return prediction
            else:
                # 더미 모델 사용
                return self.predict_price_dummy(inputs)
                
        except Exception as e:
            st.error(f"예측 오류: {e}")
            # 오류시 더미 모델로 대체
            return self.predict_price_dummy(inputs)

def main():
    """메인 대시보드"""
    
    # 헤더
    st.title("🏠 2025 서울 아파트 가격 예측기")
    st.markdown("**Random Forest 모델 (R² 0.918, MAE 13,561만원) - 최고 성능**")
    st.markdown("---")
    
    # 모델 초기화
    predictor = ApartmentPricePredictor2025()
    
    # 모델 상태 표시
    if predictor.model_loaded:
        st.success(f"✅ {predictor.model_name} 모델 로드 완료!")
    else:
        st.warning("⚠️ 간단한 추정 모델을 사용 중입니다.")
    
    # 사이드바 - 입력 패널
    st.sidebar.header("🔧 예측 조건 설정")
    
    # 1. 위치 정보
    st.sidebar.subheader("📍 자치구 정보")
    district = st.sidebar.selectbox(
        "자치구 선택",
        options=['강남구', '강동구', '강북구', '강서구', '관악구', '광진구', '구로구', '금천구',
                '노원구', '도봉구', '동대문구', '동작구', '마포구', '서대문구', '서초구', '성동구',
                '성북구', '송파구', '양천구', '영등포구', '용산구', '은평구', '종로구', '중구', '중랑구'],
        index=0  # 강남구 기본값
    )
    
    # 지하철 접근성 및 교육특구 정보 표시
    subway_score = predictor.get_subway_score(district)
    education_premium = predictor.get_education_premium(district)
    is_premium = predictor.is_premium_gu(district)
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if subway_score == 5:
            st.write("🚇 **최고 접근성**")
        elif subway_score == 4:
            st.write("🚇 **우수 접근성**")
        elif subway_score == 3:
            st.write("🚇 **보통 접근성**")
        else:
            st.write("🚇 **기본 접근성**")
    
    with col2:
        if education_premium:
            st.write("📚 **교육특구**")
        else:
            st.write("📚 일반지역")
    
    if is_premium:
        st.sidebar.success("⭐ 강남3구 프리미엄 지역")
    
    # 2. 아파트 정보
    st.sidebar.subheader("🏠 아파트 정보")
    
    # 평수 입력
    pyeong = st.sidebar.slider(
        "평수",
        min_value=10.0,
        max_value=100.0,
        value=32.0,
        step=0.5
    )
    
    # 건축면적 자동 계산
    area = pyeong / 0.3025
    st.sidebar.write(f"건축면적: **{area:.1f}㎡**")
    
    # 층수
    floor = st.sidebar.slider(
        "층수",
        min_value=1,
        max_value=70,
        value=10
    )
    
    # 건축년도
    build_year = st.sidebar.slider(
        "건축년도",
        min_value=1980,
        max_value=2024,
        value=2015
    )
    
    # 2025년 기준 건물나이 (전처리 코드와 일치)
    building_age = 2025 - build_year
    st.sidebar.write(f"건물나이: **{building_age}년** (2025년 기준)")
    
    # 3. 브랜드 정보
    st.sidebar.subheader("🏢 아파트 브랜드 정보")
    
    brand_name = st.sidebar.selectbox(
        "브랜드 선택",
        options=['기타 브랜드', '래미안', '자이', 'e편한세상', '힐스테이트', '아크로', '더샵',
                '푸르지오', '롯데캐슬', '수자인', '위브', '아이파크', '센트럴',
                '현대', '삼성', '한양', '두산'],
        index=0  # 브랜드없음 기본값
    )
    
    brand_score = predictor.get_brand_score(brand_name)
    
    # 브랜드 등급 표시
    if brand_score == 5:
        st.sidebar.success("⭐⭐⭐⭐⭐ 프리미엄 브랜드")
    elif brand_score == 4:
        st.sidebar.info("⭐⭐⭐⭐ 고급 브랜드")
    elif brand_score == 3:
        st.sidebar.warning("⭐⭐⭐ 중급 브랜드")
    elif brand_score == 2:
        st.sidebar.warning("⭐⭐ 일반 브랜드")    
    else:
        st.sidebar.error("⭐ 기타 브랜드")
    
    # 예측 버튼
    st.sidebar.markdown("---")
    if st.sidebar.button("🔮 가격 예측하기", type="primary", use_container_width=True):
        
        # 입력값 정리
        inputs = {
            'CGG_NM': district,
            'PYEONG': pyeong,
            'FLR': floor,
            'BUILDING_AGE': building_age,
            'BRAND_SCORE': brand_score,
            'SUBWAY_SCORE': subway_score,
            'EDUCATION_PREMIUM': education_premium,
            'IS_PREMIUM_GU': is_premium,
            'CGG_LABEL_ENCODED': predictor.encode_district(district)
        }
        
        # 예측 실행
        prediction = predictor.predict_price(inputs)
        
        if prediction:
            # 메인 화면 - 예측 결과
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown("## 🎯 예측 결과")
                
                # 파란색 테마 예측 가격 표시
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #2196F3 0%, #1565C0 50%, #0D47A1 100%);
                    padding: 2.5rem;
                    border-radius: 20px;
                    text-align: center;
                    color: white;
                    margin: 1.5rem 0;
                    box-shadow: 0 8px 32px rgba(33, 150, 243, 0.4);
                    border: 1px solid rgba(255, 255, 255, 0.2);
                ">
                    <h1 style="margin: 0; font-size: 3.5rem; font-weight: 700; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                        {prediction:,.0f}만원
                    </h1>
                    <h3 style="margin: 1rem 0 0 0; font-size: 1.5rem; opacity: 0.9;">
                        약 {prediction/10000:.1f}억원
                    </h3>
                </div>
                """, unsafe_allow_html=True)
                
                # 평당 가격
                price_per_pyeong = prediction / pyeong
                st.metric(
                    label="🏷️ 평당 가격",
                    value=f"{price_per_pyeong:,.0f}만원",
                    delta=f"{pyeong:.1f}평 기준"
                )
            
            # 상세 정보
            st.markdown("---")
            st.markdown("### 📋 입력 정보 요약")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📍 위치", district)
                st.metric("🏠 평수", f"{pyeong:.1f}평")
            
            with col2:
                st.metric("🏢 층수", f"{floor}층")
                st.metric("🏗️ 건물나이", f"{building_age}년")
            
            with col3:
                st.metric("🏢 브랜드", brand_name)
                st.metric("⭐ 브랜드 점수", f"{brand_score}점")
            
            with col4:
                st.metric("🚇 지하철 점수", f"{subway_score}점")
                st.metric("📚 교육특구", "Yes" if education_premium else "No")
            
            # 8피처 상세 분석
            st.markdown("---")
            st.markdown("### 🔍 8피처 상세 분석")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 수치형 피처**")
                st.write(f"- 구별 라벨: {inputs['CGG_LABEL_ENCODED']}")
                st.write(f"- 평수: {pyeong:.1f}평")
                st.write(f"- 건축년수: {building_age}년 (2025년 기준)")
                st.write(f"- 층수: {floor}층")
            
            with col2:
                st.markdown("**⭐ 점수형 피처**")
                st.write(f"- 브랜드 점수: {brand_score}점/5점")
                st.write(f"- 지하철 점수: {subway_score}점/5점")
                st.write(f"- 강남3구: {'Yes' if is_premium else 'No'}")
                st.write(f"- 교육특구: {'Yes' if education_premium else 'No'}")
    
    # 하단 정보
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 모델 성능")
        if predictor.model_loaded:
            st.write(f"- **모델**: {predictor.model_name}")
            st.write("- **R² Score**: 0.918")
            st.write("- **MAE**: 13,561만원") 
            st.write("- **MAPE**: 10.1%")
            st.write("- **학습 데이터**: 2022-2024")
        else:
            st.write("- **모델**: 간단한 추정 모델")
            st.write("- **기준**: 구별 평당가 × 조정계수")
            st.write("- **정확도**: 참고용")
    
    with col2:
        st.markdown("### 🔥 8피처 구성")
        st.write("1. **구별 라벨 인코딩** (과적합 방지)")
        st.write("2. **평수** (핵심 가격 결정 요인)")
        st.write("3. **건축년수** (2025년 기준)")
        st.write("4. **층수** (중층 선호)")
        st.write("5. **브랜드 점수** (1-5점, 기타 브랜드=1점)")
        st.write("6. **강남3구** (프리미엄 지역)")
        st.write("7. **지하철 접근성** (2-5점)")
        st.write("8. **교육특구** (학군 프리미엄)")


    # 주의사항
    st.markdown("---")
    st.info("""
    💡 **안내사항**
    - 이 예측서비스는 실제 서울 아파트 3년(2022~2024) 거래 데이터(103,251건)를 학습된 머신러닝 모델입니다.
    - 브랜드, 지하철 접근성, 교육특구 등 실제 부동산 가격에 영향을 미치는 요인들을 반영했습니다.
    - 실제 거래가격은 아파트의 개별 아파트의 특성에 따라 달라질 수 있습니다. (고가 아파트 일수록 실거래가와 차이가 많이 날 수 있습니다.)
    - 투자 결정은 반드시 전문가와 상담하시기 바랍니다.
    """)

if __name__ == "__main__":
    main()