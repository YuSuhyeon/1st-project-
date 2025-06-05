"""
서울 아파트 가격 예측 웹 대시보드 (파란색 테마)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path
import os

# 페이지 설정
st.set_page_config(
    page_title="서울 아파트 가격 예측기",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ApartmentPricePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoders = {}
        self.feature_columns = []
        self.model_loaded = False
        self.load_models()
        
    def load_models(self):
        """저장된 모델 로드 (개선된 버전)"""
        try:
            # models 폴더 확인
            models_dir = Path("models")
            if not models_dir.exists():
                st.error("❌ models 폴더가 없습니다!")
                self.create_dummy_model()
                return False
            
            # 모델 파일들 확인
            rf_path = models_dir / "random_forest_model.pkl"
            xgb_path = models_dir / "xgboost_model.pkl"
            lr_path = models_dir / "linear_regression_model.pkl"
            
            model_loaded = False
            
            # Random Forest 시도
            if rf_path.exists():
                try:
                    self.model = joblib.load(rf_path)
                    model_name = "Random Forest"
                    model_loaded = True
                    st.success(f"✅ {model_name} 모델 로드 성공!")
                except Exception as e:
                    st.warning(f"Random Forest 로드 실패: {e}")
            
            # XGBoost 시도 (Random Forest 실패시)
            elif xgb_path.exists():
                try:
                    self.model = joblib.load(xgb_path)
                    model_name = "XGBoost"
                    model_loaded = True
                    st.success(f"✅ {model_name} 모델 로드 성공!")
                except Exception as e:
                    st.warning(f"XGBoost 로드 실패: {e}")
            
            # Linear Regression 시도 (마지막 대안)
            elif lr_path.exists():
                try:
                    self.model = joblib.load(lr_path)
                    # Scaler도 로드
                    scaler_path = models_dir / "scaler.pkl"
                    if scaler_path.exists():
                        self.scaler = joblib.load(scaler_path)
                    model_name = "Linear Regression"
                    model_loaded = True
                    st.success(f"✅ {model_name} 모델 로드 성공!")
                except Exception as e:
                    st.warning(f"Linear Regression 로드 실패: {e}")
            
            if not model_loaded:
                st.error("❌ 모든 모델 로드 실패!")
                self.create_dummy_model()
                return False
            
            # 피처 컬럼 설정
            self.feature_columns = [
                'CGG_NM', 'ARCH_AREA', 'PYEONG', 'FLR', 'BUILDING_AGE',
                'YEAR', 'MONTH', 'PYEONG_GROUP', 'SEASON', 'IS_DIRECT_TRADE'
            ]
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            st.error(f"❌ 모델 로드 실패: {e}")
            self.create_dummy_model()
            return False
    
    def create_dummy_model(self):
        """더미 모델 생성 (모델 로드 실패시 대안)"""
        st.warning("⚠️ 저장된 모델을 찾을 수 없어 간단한 추정 모델을 사용합니다.")
        self.model_loaded = False
    
    def get_district_mapping(self):
        """서울 자치구 매핑"""
        return {
            '강남구': 0, '강동구': 1, '강북구': 2, '강서구': 3, '관악구': 4,
            '광진구': 5, '구로구': 6, '금천구': 7, '노원구': 8, '도봉구': 9,
            '동대문구': 10, '동작구': 11, '마포구': 12, '서대문구': 13, '서초구': 14,
            '성동구': 15, '성북구': 16, '송파구': 17, '양천구': 18, '영등포구': 19,
            '용산구': 20, '은평구': 21, '종로구': 22, '중구': 23, '중랑구': 24
        }
    
    def get_district_multiplier(self, district):
        """구별 가격 배수 (더미 모델용)"""
        multipliers = {
            '강남구': 2.8, '서초구': 2.6, '송파구': 2.2, '강동구': 1.8,
            '마포구': 2.1, '용산구': 2.4, '성동구': 1.9, '광진구': 1.8,
            '동대문구': 1.6, '중랑구': 1.4, '성북구': 1.7, '강북구': 1.5,
            '도봉구': 1.4, '노원구': 1.5, '은평구': 1.6, '서대문구': 1.8,
            '종로구': 2.0, '중구': 1.9, '영등포구': 1.9, '동작구': 1.7,
            '관악구': 1.6, '양천구': 1.8, '강서구': 1.5, '구로구': 1.4, '금천구': 1.3
        }
        return multipliers.get(district, 1.5)
    
    def encode_inputs(self, inputs):
        """입력값 인코딩"""
        encoded = inputs.copy()
        
        # 자치구 인코딩
        district_map = self.get_district_mapping()
        encoded['CGG_NM'] = district_map.get(inputs['CGG_NM'], 0)
        
        # 평형대 그룹 인코딩
        pyeong_map = {'소형': 0, '중소형': 1, '중형': 2, '대형': 3, '초대형': 4}
        encoded['PYEONG_GROUP'] = pyeong_map.get(inputs['PYEONG_GROUP'], 0)
        
        # 계절 인코딩
        season_map = {'봄': 0, '여름': 1, '가을': 2, '겨울': 3}
        encoded['SEASON'] = season_map.get(inputs['SEASON'], 0)
        
        return encoded
    
    def predict_price_dummy(self, inputs):
        """더미 예측 모델 (ML 모델 없을 때)"""
        # 기본 가격: 평당 4000만원
        base_price_per_pyeong = 4000
        
        # 평수 기반 기본 가격
        base_price = inputs['PYEONG'] * base_price_per_pyeong
        
        # 구별 조정
        district_multiplier = self.get_district_multiplier(inputs['CGG_NM'])
        price = base_price * district_multiplier
        
        # 건물나이 조정 (신축일수록 비쌈)
        age_factor = max(0.7, 1 - (inputs['BUILDING_AGE'] * 0.01))
        price *= age_factor
        
        # 층수 조정 (10층 근처가 좋음)
        if 5 <= inputs['FLR'] <= 15:
            floor_factor = 1.1
        elif inputs['FLR'] < 3:
            floor_factor = 0.9
        else:
            floor_factor = 1.0
        price *= floor_factor
        
        # 직거래 할인
        if inputs['IS_DIRECT_TRADE']:
            price *= 0.97
        
        return max(price, 10000)  # 최소 1억원
    
    def predict_price(self, inputs):
        """가격 예측"""
        try:
            if self.model_loaded and self.model:
                # 머신러닝 모델 사용
                encoded_inputs = self.encode_inputs(inputs)
                
                # 예측용 DataFrame 생성
                pred_data = pd.DataFrame([encoded_inputs], columns=self.feature_columns)
                
                # Linear Regression인 경우 스케일링 적용
                if self.scaler:
                    pred_data_scaled = self.scaler.transform(pred_data)
                    prediction = self.model.predict(pred_data_scaled)[0]
                else:
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
    st.title("🏠 서울 아파트 가격 예측기")
    st.markdown("**머신러닝 기반 실시간 아파트 가격 예측 서비스**")
    st.markdown("---")
    
    # 모델 초기화
    predictor = ApartmentPricePredictor()
    
    # 모델 상태 표시
    if predictor.model_loaded:
        st.success("✅ 머신러닝 모델 로드 완료!")
    else:
        st.warning("⚠️ 간단한 추정 모델을 사용 중입니다.")
    
    # 사이드바 - 입력 패널
    st.sidebar.header("🔧 예측 조건 설정")
    
    # 1. 위치 정보
    st.sidebar.subheader("📍 위치")
    district = st.sidebar.selectbox(
        "자치구 선택",
        options=['강남구', '서초구', '송파구', '강동구', '마포구', '용산구', '성동구', 
                '광진구', '동대문구', '중랑구', '성북구', '강북구', '도봉구', '노원구',
                '은평구', '서대문구', '종로구', '중구', '영등포구', '동작구', '관악구',
                '양천구', '강서구', '구로구', '금천구'],
        index=0  # 강남구 기본값
    )
    
    # 2. 아파트 정보
    st.sidebar.subheader("🏠 아파트 정보")
    
    # 면적 입력
    area = st.sidebar.slider(
        "전용면적 (㎡)",
        min_value=20.0,
        max_value=200.0,
        value=84.0,
        step=1.0
    )
    
    # 평수 자동 계산
    pyeong = area * 0.3025
    st.sidebar.write(f"평수: **{pyeong:.1f}평**")
    
    # 평형대 자동 결정
    if pyeong < 15:
        pyeong_group = '소형'
    elif pyeong < 25:
        pyeong_group = '중소형'
    elif pyeong < 35:
        pyeong_group = '중형'
    elif pyeong < 50:
        pyeong_group = '대형'
    else:
        pyeong_group = '초대형'
    
    st.sidebar.write(f"평형대: **{pyeong_group}**")
    
    # 층수
    floor = st.sidebar.slider(
        "층수",
        min_value=1,
        max_value=50,
        value=10
    )
    
    # 건축년도
    current_year = datetime.now().year
    build_year = st.sidebar.slider(
        "건축년도",
        min_value=1980,
        max_value=current_year,
        value=2015
    )
    
    building_age = current_year - build_year
    st.sidebar.write(f"건물나이: **{building_age}년**")
    
    # 3. 거래 정보
    st.sidebar.subheader("💼 거래 정보")
    
    # 거래월
    month = st.sidebar.selectbox(
        "거래월",
        options=list(range(1, 13)),
        index=9  # 10월 기본값
    )
    
    # 계절 자동 결정
    if month in [3, 4, 5]:
        season = '봄'
    elif month in [6, 7, 8]:
        season = '여름'
    elif month in [9, 10, 11]:
        season = '가을'
    else:
        season = '겨울'
    
    st.sidebar.write(f"계절: **{season}**")
    
    # 직거래 여부
    is_direct = st.sidebar.checkbox("직거래", value=False)
    
    # 예측 버튼 (파란색으로 강조)
    st.sidebar.markdown("---")
    if st.sidebar.button("🔮 가격 예측하기", type="primary", use_container_width=True):
        
        # 입력값 정리
        inputs = {
            'CGG_NM': district,
            'ARCH_AREA': area,
            'PYEONG': pyeong,
            'FLR': floor,
            'BUILDING_AGE': building_age,
            'YEAR': current_year,
            'MONTH': month,
            'PYEONG_GROUP': pyeong_group,
            'SEASON': season,
            'IS_DIRECT_TRADE': 1 if is_direct else 0
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
                    delta=f"전용 {pyeong:.1f}평 기준"
                )
            
            # 상세 정보
            st.markdown("---")
            st.markdown("### 📋 입력 정보 요약")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📍 위치", district)
                st.metric("🏠 평수", f"{pyeong:.1f}평")
            
            with col2:
                st.metric("📐 전용면적", f"{area:.0f}㎡")
                st.metric("🏗️ 건물나이", f"{building_age}년")
            
            with col3:
                st.metric("🏢 층수", f"{floor}층")
                st.metric("📅 거래월", f"{month}월")
            
            with col4:
                st.metric("🌟 평형대", pyeong_group)
                st.metric("💼 거래방식", "직거래" if is_direct else "중개거래")
    
    # 하단 정보
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 모델 정보")
        if predictor.model_loaded:
            st.write("- **모델**: Random Forest")
            st.write("- **MAPE**: 13.09%")
            st.write("- **R²**: 0.882")
            st.write("- **학습 데이터**: 103,251건")
        else:
            st.write("- **모델**: 간단한 추정 모델")
            st.write("- **기준**: 구별 평당가 × 조정계수")
            st.write("- **정확도**: 참고용")
    
    with col2:
        st.markdown("### 📈 주요 가격 결정 요인")
        st.write("1. **자치구** (36.5%)")
        st.write("2. **평수** (21.6%)")
        st.write("3. **전용면적** (19.4%)")
        st.write("4. **건물나이** (13.5%)")
        st.write("5. **기타** (9.0%)")
    
    # 주의사항
    st.markdown("---")
    st.info("""
    💡 **안내사항**
    - 이 예측은 2022-2024년 3개년의 약 103,251건의 실거래 데이터를 기반으로 한 통계적 추정값입니다.
    - 실제 거래가격은 개별 아파트의 특성(향, 층, 단지 규모 등)에 따라 달라질 수 있습니다.
    - 투자 결정시에는 반드시 전문가와 상담하시기 바랍니다.
    """)

if __name__ == "__main__":
    main()
