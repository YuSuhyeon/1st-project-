"""
프로젝트 설정 파일
"""

import os
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent

# 데이터 경로
DATA_PATHS = {
    "raw": PROJECT_ROOT / "data" / "raw",
    "processed": PROJECT_ROOT / "data" / "processed", 
    "predictions": PROJECT_ROOT / "data" / "predictions"
}

# 모델 경로
MODEL_PATH = PROJECT_ROOT / "models"

# 출력 경로
OUTPUT_PATHS = {
    "figures": PROJECT_ROOT / "outputs" / "figures",
    "reports": PROJECT_ROOT / "outputs" / "reports"
}

# API 설정
SEOUL_API_KEY = "646e476d7a62757235397547714b41"

# 모델 설정
MODEL_CONFIG = {
    "target_column": "THING_AMT",
    "test_size": 0.2,
    "random_state": 42,
    "models_to_train": ["xgboost", "random_forest", "linear_regression"]
}

# 성능 목표
PERFORMANCE_TARGETS = {
    "mape": 15.0,
    "r2": 0.65,
    "accuracy_range": 0.8
}

# 피처 설정
FEATURE_COLUMNS = [
    'CGG_CD', 'STDG_CD', 'ARCH_AREA', 'LAND_AREA', 'FLR',
    'ARCH_YR', 'BUILDING_AGE', 'PYEONG', 'YEAR', 'MONTH', 'QUARTER'
]

CATEGORICAL_COLUMNS = ['CGG_CD', 'STDG_CD']
