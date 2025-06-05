"""
Seoul Apartment Price Prediction - EDA 실행 스크립트
간편하게 EDA를 실행할 수 있는 스크립트
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 전처리 모듈 임포트
try:
    from src.data.preprocessor import SeoulApartmentPreprocessor
except ImportError as e:
    print(f"❌ 모듈 임포트 실패: {e}")
    print("📁 현재 폴더 구조를 확인해주세요.")
    print("  - src/data/preprocessor.py 파일이 있는지 확인")
    print("  - __init__.py 파일들이 있는지 확인")
    sys.exit(1)

def main():
    """EDA 메인 실행 함수"""
    print("🏠 Seoul Apartment Price Prediction")
    print("📊 탐색적 데이터 분석(EDA) 시작")
    print("=" * 50)
    
    # 데이터 파일 경로 확인
    data_path = "data/raw/20250604_182224_seoul_real_estate.csv"
    
    if not Path(data_path).exists():
        print(f"❌ 데이터 파일을 찾을 수 없습니다: {data_path}")
        print("📁 data/raw/ 폴더에 CSV 파일이 있는지 확인해주세요.")
        return
    
    try:
        # 전처리기 생성 및 실행
        preprocessor = SeoulApartmentPreprocessor(data_path)
        
        # 1. 데이터 로드
        print("\n1️⃣ 데이터 로드")
        if not preprocessor.load_data():
            return
        
        # 2. 기본 정보 확인
        print("\n2️⃣ 기본 정보 분석")
        preprocessor.basic_info()
        
        # 3. 데이터 품질 체크
        print("\n3️⃣ 데이터 품질 체크")
        missing_df, outliers = preprocessor.data_quality_check()
        
        # 4. 피처 엔지니어링
        print("\n4️⃣ 피처 엔지니어링")
        processed_df = preprocessor.feature_engineering()
        
        # 5. 시각화
        print("\n5️⃣ 데이터 시각화")
        try:
            preprocessor.visualize_trends()
        except Exception as e:
            print(f"⚠️ 시각화 중 오류 발생: {e}")
            print("   matplotlib 설정 문제일 수 있습니다. 계속 진행합니다.")
        
        # 6. 상관관계 분석
        print("\n6️⃣ 상관관계 분석")
        try:
            correlation_matrix = preprocessor.correlation_analysis()
        except Exception as e:
            print(f"⚠️ 상관관계 분석 중 오류 발생: {e}")
            print("   시각화 설정 문제일 수 있습니다. 계속 진행합니다.")
        
        # 7. 모델링 데이터 준비
        print("\n7️⃣ 모델링 데이터 준비")
        train_data, test_data, feature_columns = preprocessor.prepare_modeling_data()
        
        # 8. 요약 보고서
        print("\n8️⃣ 요약 보고서")
        preprocessor.generate_summary_report()
        
        print(f"\n🎉 EDA 완료!")
        print(f"📁 결과 파일:")
        print(f"  • data/processed/train_data_2022_2024.csv")
        print(f"  • data/processed/test_data_2025.csv")
        print(f"  • data/processed/feature_info.json")
        
        print(f"\n🚀 다음 단계: 모델 학습")
        print(f"  python scripts/train_models.py")
        
        return preprocessor
        
    except Exception as e:
        print(f"❌ EDA 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_requirements():
    """필요한 라이브러리 체크"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ 누락된 패키지: {', '.join(missing_packages)}")
        print(f"💡 설치 명령어: pip install {' '.join(missing_packages)}")
        return False
    
    return True

if __name__ == "__main__":
    # 라이브러리 체크
    if not check_requirements():
        print("\n📦 필요한 패키지를 먼저 설치해주세요:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # EDA 실행
    result = main()
    
    if result:
        print("\n✅ EDA 성공적으로 완료!")
    else:
        print("\n❌ EDA 실행 실패")
        sys.exit(1)