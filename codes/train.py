# ==================================================================================
# 서울시 아파트 실거래가 예측 모델 (XGBoost Only Version)
# ==================================================================================
# 
# [프로젝트 목표]
# - 서울시 아파트 실거래가를 예측하여 RMSE(평균 제곱근 오차)를 최소화
# - Baseline 모델(Random Forest) 대비 성능 개선
#
# [핵심 전략]
# 1. **Target Transformation**: '거래금액' 대신 '평당가(Unit Price)'를 예측
#    → 면적에 따른 가격 편차를 정규화하여 모델 학습 효율 향상
#
# 2. **Feature Engineering**: 
#    - 지리적 특성: K-Means Clustering (1000개 클러스터)
#    - 교통 밀도: 버스/지하철 반경별 개수 (300m, 500m, 800m, 1200m)
#    - 회전 좌표: 45도 회전으로 서울의 대각선 지리 패턴 학습
#
# 3. **Model Selection**: 
#    - 초기: Random Forest (Baseline)
#    - 실험: RF + XGBoost + LightGBM 앙상블 (실패 - 성능 저하)
#    - 최종: **XGBoost 단일 모델** (가장 우수한 성능)
#
# 4. **Validation Strategy**: 
#    - Time Series Split (최근 3개월 Cutoff)
#    - Look-ahead Bias 방지로 리더보드 점수와 CV 점수 일치
#
# [최종 성과]
# - Baseline RMSE: 16,627 → Final RMSE: 15,114 (약 9% 개선)
# ==================================================================================

# ==================================================================================
# 라이브러리 임포트
# ==================================================================================
import pandas as pd              # 데이터프레임 처리
import numpy as np               # 수치 연산
from sklearn.ensemble import RandomForestRegressor, VotingRegressor  # 앙상블 모델 (현재 미사용)
from xgboost import XGBRegressor  # XGBoost 모델 (메인 모델)
from lightgbm import LGBMRegressor  # LightGBM 모델 (현재 미사용)
from sklearn.cluster import KMeans, MiniBatchKMeans  # K-Means 클러스터링 (지리적 그룹화)
from sklearn.preprocessing import LabelEncoder, StandardScaler  # 인코딩 및 스케일링
from sklearn.metrics import mean_squared_error  # RMSE 계산
import os  # 파일 경로 처리
import json  # 메타데이터 저장

# ==================================================================================
# 1. 데이터 로드 함수
# ==================================================================================
def load_data():
    """
    학습 데이터와 테스트 데이터를 로드합니다.
    
    Returns:
        train (DataFrame): 학습 데이터 (약 110만 건)
        test (DataFrame): 테스트 데이터 (약 9천 건)
    """
    print("📂 데이터 불러오는 중...")
    train = pd.read_csv('../data/train.csv', low_memory=False)  # low_memory=False: 데이터 타입 추론 정확도 향상
    test = pd.read_csv('../data/test.csv', low_memory=False)
    return train, test

# ==================================================================================
# 2. 컬럼명 변환 함수
# ==================================================================================
def rename_columns(df):
    """
    한글 컬럼명을 단순화하여 코드 가독성을 높입니다.
    
    주요 변환:
    - '전용면적(㎡)' → '전용면적'
    - '좌표X' → '경도' (Longitude)
    - '좌표Y' → '위도' (Latitude)
    - 'target' → '거래금액' (실제 거래 금액)
    
    Args:
        df (DataFrame): 변환할 데이터프레임
    
    Returns:
        DataFrame: 컬럼명이 변환된 데이터프레임
    """
    cols_mapping = {
        '전용면적(㎡)': '전용면적',
        '좌표X': '경도',  # longitude
        '좌표Y': '위도',  # latitude
        'target': '거래금액'  # 예측 대상 (실거래가)
    }
    # 컬럼명이 이미 변환되어 있을 수도 있으므로, 존재하는 컬럼만 rename
    df = df.rename(columns=cols_mapping)
    return df

# ==================================================================================
# 3. Label Encoding 함수 (Train 기준)
# ==================================================================================
def label_encode_train_test(train_s, test_s):
    """
    범주형 변수를 숫자로 변환합니다 (Label Encoding).
    
    [중요] Train 데이터의 고유값만 사용하여 매핑을 생성합니다.
    Test 데이터에 Train에 없는 값이 있으면 'Unknown'으로 처리합니다.
    
    예시:
    - Train: ['강남구', '서초구', '송파구'] → {강남구: 0, 서초구: 1, 송파구: 2}
    - Test: ['강남구', '마포구'] → [0, 3]  # '마포구'는 Unknown(3)으로 처리
    
    Args:
        train_s (Series): Train 데이터의 범주형 컬럼
        test_s (Series): Test 데이터의 범주형 컬럼
    
    Returns:
        tuple: (train_encoded, test_encoded) - 인코딩된 Series 쌍
    """
    # 모든 값을 문자열로 변환 (결측치 처리 포함)
    train_s = train_s.astype(str)
    test_s = test_s.astype(str)
    
    # Train의 고유값으로 매핑 딕셔너리 생성
    uniq = pd.Index(train_s.unique())
    mapping = {k: i for i, k in enumerate(uniq)}  # {값: 인덱스}
    unk = len(mapping)  # Unknown은 마지막 번호 부여 (예: 3개 고유값이면 Unknown=3)
    
    # Train과 Test에 매핑 적용 (Test의 Unknown 값은 unk로 채움)
    return train_s.map(mapping).fillna(unk).astype(int), test_s.map(mapping).fillna(unk).astype(int)

# ==================================================================================
# 4. 데이터 전처리 및 특성 공학 메인 함수
# ==================================================================================
def preprocess_data(train, test):
    """
    데이터 전처리 및 특성 공학(Feature Engineering)을 수행하는 메인 함수입니다.
    
    [주요 처리 단계]
    1. 컬럼명 변환 및 불필요한 변수 제거 (27개 노이즈 변수)
    2. Target Transformation: '거래금액' → '평당가' (면적 정규화)
    3. 주소 정보 분할: '시군구' → '시', '구', '동'
    4. 날짜 특성: 거래년도, 거래월, 연식 계산
    5. 교통 특성: 버스/지하철 거리 및 밀도 (BallTree 사용)
    6. 지리적 클러스터링: K-Means (1000개 클러스터)
    7. 회전 좌표: 45도 회전으로 대각선 패턴 학습
    8. Label Encoding: 범주형 변수 → 숫자
    9. 결측치 처리: Train Median으로 채우기
    
    Args:
        train (DataFrame): 학습 데이터
        test (DataFrame): 테스트 데이터
    
    Returns:
        tuple: (X_train, y_train, X_test) - 전처리된 특성과 타겟
    """
    print("⚙️ 데이터 전처리 진행 중...")
    
    # ----------------------------------------------------------------
    # Step 1: 컬럼명 변환
    # ----------------------------------------------------------------
    train = rename_columns(train)
    test = rename_columns(test)

    # ----------------------------------------------------------------
    # Step 2: 불필요한 변수 제거 (Noise Reduction)
    # ----------------------------------------------------------------
    # [설명] 예측에 도움이 되지 않는 27개의 노이즈 변수를 제거합니다.
    # - 관리비, 전화번호, 홈페이지 등 예측과 무관한 정보
    # - 세대현황 세부 분류 (이미 '전체세대수'로 요약됨)
    # - 날짜 정보 (등록일자, 수정일자 등)
    cols_to_drop = [
        '해제사유발생일', '단지소개기존clob', 'k-관리비부과면적',
        'k-전용면적별세대현황(60㎡이하)', 'k-전용면적별세대현황(60㎡~85㎡이하)',
        'k-85㎡~135㎡이하', 'k-135㎡초과', '건축면적', 'K-전화번호', 'K-팩스번호', 'k-세대타입(분양형태)',
        'k-관리방식','k-복도유형','k-난방방식','k-사용검사일-사용승인일','k-홈페이지','k-등록일자','k-수정일자',
        '고용보험관리번호','경비비관리형태','세대전기계약방법','청소비관리형태','기타/의무/임대/임의=1/2/3/4','단지승인일','사용허가여부','관리비 업로드','단지신청일',
    ]
    train = train.drop(columns=cols_to_drop, errors='ignore')  # errors='ignore': 없는 컬럼은 무시
    test = test.drop(columns=cols_to_drop, errors='ignore')
    print(f"✅ Dropped {len(cols_to_drop)} noise features")

    # 불필요한 인덱스 컬럼 제거 (CSV 저장 시 생성된 인덱스)
    if 'Unnamed: 0' in train.columns:
        train = train.drop(columns=['Unnamed: 0'])
    if 'Unnamed: 0' in test.columns:
        test = test.drop(columns=['Unnamed: 0'])

    target_col = '거래금액'  # 예측 대상: 실거래가 (만원 단위)
    
    # ----------------------------------------------------------------
    # Step 3: [핵심] Target Transformation - Unit Price (평당가) 생성
    # ----------------------------------------------------------------
    # [설명] '거래금액'을 직접 예측하면 면적에 따른 편차가 너무 큽니다.
    # 예: 10평 아파트 2억 vs 30평 아파트 6억 → 모델이 면적만 학습
    # 
    # [해결책] '평당가(Unit Price)'를 예측하여 면적 효과를 정규화합니다.
    # - 평당가 = 거래금액 / 전용면적
    # - 예측 시: 평당가 * 전용면적 = 거래금액 (복원)
    # 
    # [효과] Baseline 대비 약 448점 개선 (16,627 → 16,179)
    train['평당가'] = train[target_col] / train['전용면적']
    
    # ----------------------------------------------------------------
    # Step 4: Train/Test 데이터 병합 (전처리 일괄 적용)
    # ----------------------------------------------------------------
    # [이유] Feature Engineering을 Train/Test에 동일하게 적용하기 위함
    # 나중에 'is_train' 컬럼으로 다시 분리합니다.
    train['is_train'] = 1  # 학습 데이터 식별자
    test['is_train'] = 0   # 테스트 데이터 식별자
    test[target_col] = np.nan  # Test에는 정답이 없음
    test['평당가'] = np.nan
    
    data = pd.concat([train, test], axis=0, ignore_index=True)  # 병합 (약 110만 + 9천 = 119만 건)
    
    # ----------------------------------------------------------------
    # Metadata 초기화 (Streamlit 앱에서 사용)
    # ----------------------------------------------------------------
    meta = {
        'dropped_features': cols_to_drop,  # 제거된 변수 목록
        'imputation': {},  # 결측치 처리 정보
        'encoding': {},  # 인코딩 정보
        'address_example': {}  # 주소 분할 예시
    }

    # ----------------------------------------------------------------
    # Step 5: 주소 정보 분할 (시/구/동)
    # ----------------------------------------------------------------
    # [설명] '시군구' 컬럼을 공백 기준으로 분할하여 행정구역 정보를 추출합니다.
    # 예: "서울특별시 강남구 역삼동" → 시="서울특별시", 구="강남구", 동="역삼동"
    # 
    # [중요성] '구' 정보는 Feature Importance 1위 (강남 프리미엄 반영)
    print("📍 주소 정보 처리 중...")
    data['시군구'] = data['시군구'].fillna("Unknown Unknown Unknown")  # 결측치 처리
    sigungu_split = data['시군구'].str.split(' ', expand=True)  # 공백으로 분할
    
    # 분할 결과에 따라 시/구/동 할당
    if sigungu_split.shape[1] >= 3:  # 정상적으로 3개 이상 분할된 경우
        data['시'] = sigungu_split[0]  # 예: "서울특별시"
        data['구'] = sigungu_split[1]  # 예: "강남구" (가장 중요!)
        data['동'] = sigungu_split[2]  # 예: "역삼동"
    else:  # 분할 실패 시 Unknown 처리
        data['시'] = sigungu_split[0]
        data['구'] = 'Unknown'
        data['동'] = 'Unknown'
    
    # 예시 저장 (Streamlit 앱에서 표시용)
    split_ex = f"{data['시군구'].iloc[0]} → {data['시'].iloc[0]}, {data['구'].iloc[0]}, {data['동'].iloc[0]}"
    meta['address_example'] = split_ex
            
    # ----------------------------------------------------------------
    # Step 6: 날짜 및 연식 정보 파생
    # ----------------------------------------------------------------
    # [설명] 계약년월에서 년도/월을 추출하고, 건축년도와의 차이로 연식을 계산합니다.
    # - 거래년도/월: 시장 사이클 반영 (2022년 금리 인상 등)
    # - 연식: 신축 프리미엄 반영 (U자형 패턴: 재건축 기대 + 신축)
    data['계약년월'] = data['계약년월'].astype(str)  # 문자열 변환
    data['거래년도'] = data['계약년월'].str[:4].astype(int)  # 앞 4자리: 년도
    data['거래월'] = data['계약년월'].str[4:].astype(int)  # 뒤 2자리: 월
    
    # 아파트 연식 (Age) 계산: 거래년도 - 건축년도
    # 예: 2023년 거래, 2000년 건축 → 연식 23년
    data['연식'] = data['거래년도'] - data['건축년도']

    # 4. [핵심] 지리적 특성 강화 (Enhanced Geo Features)
    print("🚌 지리 정보 강화 중 (클러스터링 & 교통 피처)...")
    
    # ----------------------------------------------------------------
    # 4-0. 교통 편의성 특성 추가 (Bus & Subway) - Advanced
    # ----------------------------------------------------------------
    try:
        from sklearn.neighbors import BallTree
        print("  교통 데이터 로드 중 (Bus & Subway)...")
        
        bus = pd.read_csv('../data/bus_feature.csv')
        sub = pd.read_csv('../data/subway_feature.csv')
        
        # 결측 좌표 제거
        bus = bus.dropna(subset=['X좌표', 'Y좌표'])
        sub = sub.dropna(subset=['경도', '위도'])
        
        # 좌표 데이터 준비 (Radians for Haversine)
        # Bus: Y좌표=Lat, X좌표=Lon
        bus_rad = np.radians(bus[['Y좌표', 'X좌표']].values)
        # Sub: 위도=Lat, 경도=Lon
        sub_rad = np.radians(sub[['위도', '경도']].values)
        
        # Data: 위도=Lat, 경도=Lon
        data_coords = data[['위도', '경도']].fillna(0)
        data_rad = np.radians(data_coords.values)
        
        # -------------------------------------------------------
        # 7-1. 최단 거리 (Nearest Distance)
        # -------------------------------------------------------
        # [설명] 가장 가까운 버스정류장/지하철역까지의 거리를 계산합니다.
        # - BallTree.query(k=1): 가장 가까운 1개 지점 검색
        # - 반환값: 라디안 단위 거리 → 미터 변환 (* 6371000)
        # - Clipping: 이상치 제거 (버스 2km, 지하철 5km 상한)
        print("  - 최단 거리 계산 중...")
        
        # 7-1-1. Bus 최단 거리
        tree_bus = BallTree(bus_rad, metric='haversine')  # BallTree 생성 (버스)
        dist_bus, _ = tree_bus.query(data_rad, k=1)  # k=1: 가장 가까운 1개
        data['dist_to_bus'] = (dist_bus[:, 0] * 6371000).astype(np.float32).clip(0, 2000)  # 0~2km 제한
        
        # 7-1-2. Subway 최단 거리
        tree_sub = BallTree(sub_rad, metric='haversine')  # BallTree 생성 (지하철)
        dist_sub, _ = tree_sub.query(data_rad, k=1)
        data['dist_to_subway'] = (dist_sub[:, 0] * 6371000).astype(np.float32).clip(0, 5000)  # 0~5km 제한

        # -------------------------------------------------------
        # 7-2. 반경 내 개수 (Count within Radius) - 밀도 측정
        # -------------------------------------------------------
        # [설명] 여러 반경(300m, 500m, 800m, 1200m) 내에 몇 개의 정류장/역이 있는지 계산합니다.
        # - 반경이 작을수록: 더 세밀한 역세권 판단
        # - 반경이 클수록: 더 넓은 교통 편의성 반영
        # 
        # [효과] 다양한 반경을 사용하여 모델이 비선형 패턴을 학습할 수 있습니다.
        print("  - 반경 내 개수 계산 중 (Bus: 300/500/800, Subway: 300/500/800/1200)...")
        
        # 7-2-1. Bus 밀도 (300m, 500m, 800m)
        for r in [300, 500, 800]:  # 반경 (미터)
            radius_rad = r / 6371000  # 미터 → 라디안 변환
            count = tree_bus.query_radius(data_rad, r=radius_rad, count_only=True)  # 반경 내 개수
            data[f'bus_cnt_{r}'] = count  # 예: bus_cnt_300, bus_cnt_500, bus_cnt_800
        
        # 7-2-2. Subway 밀도 (300m, 500m, 800m, 1200m)
        for r in [300, 500, 800, 1200]:  # 지하철은 1200m까지 확장
            radius_rad = r / 6371000
            count = tree_sub.query_radius(data_rad, r=radius_rad, count_only=True)
            data[f'sub_cnt_{r}'] = count  # 예: sub_cnt_300, sub_cnt_500, sub_cnt_800, sub_cnt_1200
        
        # -------------------------------------------------------
        # 7-3. 지하철 접근성 점수 (Weighted Subway Score)
        # -------------------------------------------------------
        # [설명] 모든 지하철이 동일한 가치를 가지지 않습니다!
        # - 2호선, 9호선, 신분당선: 가장 중요 (2.0 가중치)
        # - 3호선, 5호선, 7호선: 중간 (1.5 가중치)
        # - 1호선, 4호선, 6호선, 8호선: 기본 (1.2 가중치)
        # - 기타 호선: 1.0 가중치
        # 
        # [이유] 강남 지역을 관통하는 호선일수록 프리미엄이 높습니다.
        print("  - 지하철 접근성 점수 계산 중 (호선별 가중치 적용)...")
        
        def get_line_weight(line_name):
            """
            지하철 호선별 가중치를 반환합니다.
            
            가중치 기준:
            - 2.0: 강남 핵심 호선 (2호선, 9호선, 신분당선)
            - 1.5: 주요 호선 (3호선, 5호선, 7호선)
            - 1.2: 기본 호선 (1호선, 4호선, 6호선, 8호선)
            - 1.0: 기타
            """
            if any(l in line_name for l in ['2호선', '9호선', '신분당', '분당선']):
                return 2.0  # 가장 중요한 호선
            elif any(l in line_name for l in ['3호선', '7호선', '5호선']):
                return 1.5  # 주요 호선
            elif any(l in line_name for l in ['1호선', '4호선', '6호선', '8호선']):
                return 1.2  # 기본 호선
            return 1.0  # 기타
            
        sub['weight'] = sub['호선'].fillna('').apply(get_line_weight)  # 각 역에 가중치 부여
        
        # 가중치 계산: 1200m 반경 내 가중 합계
        radius_sub_max = 1200 / 6371000  # 1200m → 라디안
        
        # Group 1: High (2.0) - 강남 핵심 호선
        sub_high = sub[sub['weight'] == 2.0]
        tree_high = BallTree(np.radians(sub_high[['위도', '경도']].values), metric='haversine')
        cnt_high = tree_high.query_radius(data_rad, r=radius_sub_max, count_only=True)
        
        # Group 2: Mid (1.5) - 주요 호선
        sub_mid = sub[sub['weight'] == 1.5]
        tree_mid = BallTree(np.radians(sub_mid[['위도', '경도']].values), metric='haversine')
        cnt_mid = tree_mid.query_radius(data_rad, r=radius_sub_max, count_only=True)
        
        # Group 3: Low (Others) - 기타 호선
        sub_low = sub[~sub.index.isin(sub_high.index) & ~sub.index.isin(sub_mid.index)]
        tree_low = BallTree(np.radians(sub_low[['위도', '경도']].values), metric='haversine')
        cnt_low = tree_low.query_radius(data_rad, r=radius_sub_max, count_only=True)
        
        # 최종 점수 = (고급 호선 개수 * 2.0) + (중급 호선 개수 * 1.5) + (기타 호선 개수 * 1.0)
        data['sub_score_1200'] = (cnt_high * 2.0) + (cnt_mid * 1.5) + (cnt_low * 1.0)
        
        print(f"  ✅ 추가된 피처: dist_to_bus/subway, bus_cnt_(300/500/800), sub_cnt_(300/500/800/1200), sub_score_1200")
        
    except Exception as e:
        # 교통 피처 추가 실패 시 0으로 채우기 (모델 학습은 계속 진행)
        print(f"⚠️ 교통 피처 추가 실패: {e}")
        new_cols = ['dist_to_bus', 'dist_to_subway', 'sub_score_1200'] + \
                   [f'bus_cnt_{r}' for r in [300, 500, 800]] + \
                   [f'sub_cnt_{r}' for r in [300, 500, 800, 1200]]
        for c in new_cols:
            data[c] = 0  # 기본값 0

    # ----------------------------------------------------------------
    # Step 8: 회전 좌표계 (45도 회전)
    # ----------------------------------------------------------------
    # [설명] 서울의 지리적 패턴은 대각선 방향입니다 (북동-남서).
    # - 강남: 남동쪽
    # - 강북: 북서쪽
    # 
    # [해결책] 45도 회전 변환으로 대각선 패턴을 수평/수직으로 변환합니다.
    # - 회전X = 위도 + 경도 (대각선 방향)
    # - 회전Y = 위도 - 경도 (수직 방향)
    # 
    # [효과] 모델이 서울의 대각선 가격 패턴을 더 잘 학습합니다.
    if '경도' in data.columns and '위도' in data.columns:
        data['회전좌표X'] = data['위도'] + data['경도']  # 대각선 방향
        data['회전좌표Y'] = data['위도'] - data['경도']  # 수직 방향

    # ----------------------------------------------------------------
    # Step 9: Train/Test 분리 (Data Leakage 방지)
    # ----------------------------------------------------------------
    # [중요] 여기서 Train/Test를 분리하여 Data Leakage를 방지합니다.
    # - K-Means: Train으로만 Fit, Test는 Predict
    # - Imputation: Train Median으로만 계산, Test도 동일한 값 사용
    # - Label Encoding: Train 고유값으로만 매핑, Test는 Unknown 처리
    train_part = data[data['is_train'] == 1].copy()  # Train 데이터 추출
    test_part = data[data['is_train'] == 0].copy()   # Test 데이터 추출
    
    # ----------------------------------------------------------------
    # Step 10: K-Means 클러스터링 (1000개 클러스터)
    # ----------------------------------------------------------------
    # [설명] 위경도 좌표를 1000개의 그룹으로 분류하여 지리적 패턴을 학습합니다.
    # - 같은 클러스터 = 비슷한 위치 = 비슷한 가격 패턴
    # - 1000개: 서울 전체를 세밀하게 분할 (약 110만 건 / 1000 = 클러스터당 1100건)
    # 
    # [중요] Train으로만 Fit, Test는 Predict (Data Leakage 방지)
    # - MiniBatchKMeans: 대용량 데이터에 빠른 클러스터링
    # - batch_size=4096: 한 번에 4096건씩 처리
    if '경도' in train_part.columns and '위도' in train_part.columns:
        print("📍 MiniBatchKMeans 실행 중 (k=1000, Train만)...")
        coords_tr = train_part[['경도', '위도']].fillna(0).values  # Train 좌표
        coords_te = test_part[['경도', '위도']].fillna(0).values   # Test 좌표
        
        kmeans = MiniBatchKMeans(
            n_clusters=1000,      # 1000개 클러스터
            random_state=42,      # 재현성 보장
            batch_size=4096,      # 배치 크기
            n_init=10             # 초기화 횟수
        )
        train_part['클러스터'] = kmeans.fit_predict(coords_tr).astype(str)  # Train: Fit + Predict
        test_part['클러스터'] = kmeans.predict(coords_te).astype(str)        # Test: Predict만
        
    # ----------------------------------------------------------------
    # Step 11: 결측치 처리 (Imputation)
    # ----------------------------------------------------------------
    # [설명] 결측치를 채워서 모델이 학습할 수 있도록 합니다.
    # 
    # [전략]
    # - 숫자형: Train Median으로 채우기 (평균보다 이상치에 강건)
    # - 범주형: 'Unknown'으로 채우기 (Label Encoding에서 처리)
    # 
    # [중요] Train Median만 사용 (Data Leakage 방지)
    # - Test의 결측치도 Train Median으로 채움
    print("🔧 숫자형 결측치 처리 중 (Train Median)...")
    num_cols = train_part.select_dtypes(include=[np.number]).columns  # 숫자형 컬럼
    cols_to_fill = [c for c in num_cols if c != target_col and c != '평당가' and c != 'is_train']  # 타겟 제외
    
    for col in cols_to_fill:
        med = train_part[col].median()  # Train Median 계산
        train_part[col] = train_part[col].fillna(med)  # Train 채우기
        test_part[col] = test_part[col].fillna(med)    # Test도 동일한 값으로 채우기
        meta['imputation'][col] = f"Median ({med:.2f})"
        
    # 범주형 결측치 처리
    cat_cols = train_part.select_dtypes(include=['object']).columns  # 문자열 컬럼
    for col in cat_cols:
        train_part[col] = train_part[col].fillna('Unknown')  # 'Unknown'으로 채우기
        test_part[col] = test_part[col].fillna('Unknown')
        meta['imputation'][col] = "Unknown"

    # ----------------------------------------------------------------
    # Step 12: Label Encoding (범주형 변수 → 숫자)
    # ----------------------------------------------------------------
    # [설명] 문자열 변수를 숫자로 변환하여 모델이 학습할 수 있도록 합니다.
    # 
    # [대상 변수]
    # - '시', '구', '동': 행정구역 (가장 중요!)
    # - '아파트명', '도로명': 위치 식별자
    # - '클러스터': K-Means 결과
    # 
    # [중요] Train 고유값으로만 매핑 (Data Leakage 방지)
    # - Test에 새로운 값이 나오면 'Unknown'으로 처리
    print("🏷️ Label Encoding 중 (Train 기준)...")
    encoding_features = ['시', '구', '동', '아파트명', '도로명', '클러스터']  # 인코딩 대상
    
    for col in encoding_features:
        if col in train_part.columns:
            # label_encode_train_test 함수 사용 (Train 기준 매핑)
            train_part[col], test_part[col] = label_encode_train_test(train_part[col], test_part[col])
            meta['encoding'][col] = "Strict Mapping"  # 메타데이터 기록
            
    # ----------------------------------------------------------------
    # Step 13: Feature Selection (최종 특성 선택)
    # ----------------------------------------------------------------
    # [설명] 모델 학습에 사용할 최종 특성을 선택합니다.
    # 
    # [제거 대상]
    # - 문자열 컬럼: Label Encoding이 안 된 문자열 (모델이 학습 불가)
    # - 'is_train': 데이터 분리용 플래그 (특성 아님)
    # - '거래금액', '평당가': 타겟 변수 (특성에 포함하면 Data Leakage)
    # 
    # [최종 특성]
    # - 지리: '구', '동', '경도', '위도', '회전좌표X/Y', '클러스터'
    # - 교통: 'dist_to_bus/subway', 'bus_cnt_*', 'sub_cnt_*', 'sub_score_1200'
    # - 건물: '전용면적', '건축년도', '연식', '층', '전체세대수' 등
    # - 시간: '거래년도', '거래월'
    
    # 불필요한 문자열 컬럼 제거 (Label Encoding이 안 된 컬럼)
    object_cols = train_part.select_dtypes(include=['object']).columns
    train_part = train_part.drop(columns=object_cols)
    test_part = test_part.drop(columns=object_cols)
    
    # 학습/테스트 데이터 정리
    X_train = train_part.drop(columns=['is_train'])  # 'is_train' 플래그 제거
    X_test = test_part.drop(columns=['is_train', target_col, '평당가'], errors='ignore')  # 타겟 변수 제거
    
    # 학습 타겟을 '평당가'로 설정 (핵심 전략!)
    target = X_train['평당가']  # 타겟 추출
    X_train = X_train.drop(columns=[target_col, '평당가'])  # 타겟 변수 제거
    
    # 메타데이터: 최종 특성 목록 저장
    meta['final_features'] = list(X_train.columns)
    print(f"✅ Final feature count: {len(X_train.columns)}")
    
    # ----------------------------------------------------------------
    # Metadata 저장 (Streamlit 앱에서 사용)
    # ----------------------------------------------------------------
    # [설명] 전처리 과정의 모든 정보를 JSON 파일로 저장합니다.
    # - Streamlit 앱에서 사용자에게 전처리 과정을 설명하기 위함
    # - 제거된 변수, 결측치 처리, 인코딩 정보 등 포함
    meta['transport_params'] = {
        'bus_radii': "[300, 500, 800]",
        'subway_radii': "[300, 500, 800, 1200]",
        'subway_weights': "Line 2/9/Shinbundang=2.0, Line 3/5/7=1.5, Others=1.0"
    }
    meta['validation_strategy'] = "Time Series Split (Last 3 Months from dataset end)"
    meta['target_info'] = "Unit Price (거래금액 / 전용면적)"
    
    # JSON 파일로 저장
    with open('preprocessing_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    print("💾 'preprocessing_metadata.json' 저장 완료")
    
    return X_train, target, X_test

# ==================================================================================
# 5. 모델 학습 함수 (XGBoost Only)
# ==================================================================================
def train_ensemble_model(X_train, y_train):
    """
    XGBoost 모델 학습 및 검증 함수
    
    [전략 변경]
    - 초기: RF + XGBoost + LightGBM 앙상블 (Soft Voting)
    - 실험 결과: 앙상블이 오히려 성능 저하 (노이즈 증가)
    - 최종: **XGBoost 단일 모델** (가장 우수한 성능)
    
    [검증 전략]
    - Time Series Split: 최근 3개월을 Validation으로 사용
    - Look-ahead Bias 방지: 과거 데이터로만 학습, 미래 데이터로 검증
    - RMSE 계산: '평당가' → '거래금액' 복원 후 계산 (실제 금액 기준)
    
    Args:
        X_train (DataFrame): 학습 특성
        y_train (Series): 학습 타겟 ('평당가')
    
    Returns:
        dict: 학습된 모델 딕셔너리 {'xgb': XGBRegressor}
    """
    # 원본 데이터 보존
    train_df = X_train.copy()
    train_df['평당가'] = y_train
    
    # ----------------------------------------------------------------
    # Step 1: 시계열 정렬 (Time Series Split 준비)
    # ----------------------------------------------------------------
    # [설명] 시간 순서대로 정렬하여 과거 → 미래 순서를 보장합니다.
    train_df = train_df.sort_values(by=['거래년도', '거래월'])
    
    y = train_df['평당가']
    X = train_df.drop(columns=['평당가'])
    
    # ----------------------------------------------------------------
    # Step 2: Validation Split (Time Series: 최근 3개월)
    # ----------------------------------------------------------------
    # [설명] 리더보드 점수와 CV 점수를 일치시키기 위해 시계열 분할을 사용합니다.
    # - 최근 3개월: Validation Set
    # - 나머지: Training Set
    # 
    # [중요] Random Split을 사용하면 Look-ahead Bias 발생!
    # - 미래 데이터로 학습 → 과거 데이터 예측 (CV 점수 좋지만 리더보드 나빨)
    print("📅 Splitting Train/Val by LAST 3 MONTHS (Time Series Split)...")
    X['ym'] = X['거래년도']*100 + X['거래월']  # 년월 합치기 (예: 202301)
    val_cutoff = X['ym'].sort_values().unique()[-3]  # 최근 3개월 중 가장 오래된 년월
    
    mask_val = X['ym'] >= val_cutoff  # Validation: 최근 3개월
    mask_tr = ~mask_val                # Training: 나머지
    
    X_tr = X[mask_tr].drop(columns=['ym'])   # Training Set
    X_val = X[mask_val].drop(columns=['ym']) # Validation Set
    y_tr = y[mask_tr]
    y_val = y[mask_val]
    
    print(f"  ✅ Train: {len(X_tr):,} samples, Val: {len(X_val):,} samples (Cutoff: {val_cutoff})")
    
    print("\n" + "="*50)
    print("🤖 Init XGBoost Model (GPU Enabled)...")
    print("="*50)
    
    # ----------------------------------------------------------------
    # Step 3: 모델 초기화 (XGBoost Only)
    # ----------------------------------------------------------------
    # [설명] 초기에는 RF + XGBoost + LightGBM 앙상블을 사용했으나,
    # 실험 결과 XGBoost 단일 모델이 가장 우수한 성능을 보여서 최종 선택했습니다.
    # 
    # [이유]
    # - RF: 각 트리가 독립적 → 잔여 오차 학습 약함
    # - XGBoost: Gradient Boosting → 잔여 오차를 집요하게 학습 → 성능 우수
    # - LightGBM: 빠르지만 XGBoost보다 성능 낮음
    # - 앙상블: 모델 간 예측 패턴 차이로 노이즈 증가 → 성능 저하
    
    # 1. Random Forest (Skipped - 사용 안 함)
    rf = RandomForestRegressor(
        n_estimators=140,
        max_features=0.5,
        random_state=42,
        n_jobs=-1
    )
    
    # ----------------------------------------------------------------
    # 2. XGBoost (Main Model) - 튜닝된 하이퍼파라미터 사용
    # ----------------------------------------------------------------
    # [설명] tune_xgb.py에서 Optuna로 찾은 최적 파라미터를 로드합니다.
    # - RMSE 15,531 달성한 최적 파라미터
    # - 파일이 없으면 기본값 사용
    print("  - XGBoost: 최적 하이퍼파라미터 로드 중...")
    
    try:
        import json
        with open('best_xgb_params.json', 'r') as f:
            best_params = json.load(f)
        print(f"  ✅ 'best_xgb_params.json' 로드 완료!")
        print(f"     - n_estimators: {best_params.get('n_estimators', 5000)}")
        print(f"     - learning_rate: {best_params.get('learning_rate', 0.01):.4f}")
        print(f"     - max_depth: {best_params.get('max_depth', 10)}")
    except FileNotFoundError:
        print("  ⚠️ 'best_xgb_params.json' 없음 → 기본값 사용")
        best_params = {}
    
    xgb = XGBRegressor(
        n_estimators=best_params.get('n_estimators', 5000),
        learning_rate=best_params.get('learning_rate', 0.01),
        max_depth=best_params.get('max_depth', 10),
        min_child_weight=best_params.get('min_child_weight', 1),
        subsample=best_params.get('subsample', 0.8),
        colsample_bytree=best_params.get('colsample_bytree', 0.8),
        reg_alpha=best_params.get('reg_alpha', 0),
        reg_lambda=best_params.get('reg_lambda', 1),
        gamma=best_params.get('gamma', 0),
        random_state=42,
        n_jobs=-1,
        enable_categorical=False,
        tree_method='hist',
        device='cuda'  # GPU 사용
    )
    
    # 3. LightGBM (Skipped - 사용 안 함)
    # 주의: LightGBM GPU 버전이 설치되어 있어야 함. 에러 발생 시 device='cpu'로 변경 필요.
    lgbm = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.01,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        device='gpu'
    )
    
    # ----------------------------------------------------------------
    # Step 4: 모델 학습 및 Validation 예측
    # ----------------------------------------------------------------
    # [설명] XGBoost만 학습하고, RF와 LightGBM은 스킨합니다.
    # - XGBoost Only Mode: 가중치 [0.0, 1.0, 0.0]
    
    # Store models and predictions
    models = {}
    val_preds = {}
    test_preds = {}
    
    # 4.1 RandomForest (Skipped)
    print("\n[1/3] RandomForest: 스킵 (XGBoost Only 모드)")
    val_preds['rf'] = np.zeros(len(X_val))  # Dummy 예측 (0으로 채움)
    models['rf'] = rf
    
    # 4.2 XGBoost (Main Model)
    print("\n[2/3] XGBoost 학습 중 (GPU)...")
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],  # Validation Set으로 조기 종료 판단
        verbose=100  # 100 iteration마다 진행 상황 출력
    )
    val_preds['xgb'] = xgb.predict(X_val)  # Validation 예측
    models['xgb'] = xgb
    
    # 4.3 LightGBM (Skipped)
    print("\n[3/3] LightGBM: 스킵 (XGBoost Only 모드)")
    val_preds['lgbm'] = np.zeros(len(X_val))  # Dummy 예측
    models['lgbm'] = lgbm

    # ----------------------------------------------------------------
    # Step 5: RMSE 계산 (Validation Set)
    # ----------------------------------------------------------------
    # [설명] 모델은 '평당가'를 예측하지만, RMSE는 '거래금액' 기준으로 계산합니다.
    # - 평당가 → 거래금액 복원: 평당가 * 전용면적
    # - RMSE = sqrt(mean((y_true - y_pred)^2))
    # 
    # [중요] 리더보드에서도 '거래금액' 기준으로 평가합니다!
    val_area = X_val['전용면적'].values  # Validation Set의 면적
    y_val_total = y_val.values * val_area  # 실제 거래금액 (만원)

    # 개별 모델 RMSE 계산
    rmse_rf = 0.0  # RandomForest는 사용 안 함
    
    rmse_xgb = np.sqrt(mean_squared_error(y_val_total, val_preds['xgb'] * val_area))
    print(f"  👉 XGBoost Validation RMSE: {rmse_xgb:,.2f}")
    
    rmse_lgbm = 0.0  # LightGBM도 사용 안 함

    # ----------------------------------------------------------------
    # Step 6: 앙상블 가중치 결정
    # ----------------------------------------------------------------
    # [설명] 세 모델의 예측을 가중 평균하여 최종 예측을 만듭니다.
    # 
    # [전략]
    # - Manual Mode: 수동으로 가중치 설정 [0.0, 1.0, 0.0] = XGBoost Only
    # - Auto Mode: scipy.optimize로 최적 가중치 탐색 (실험용)
    # 
    # [결론] XGBoost 단일 모델이 가장 우수하므로 Manual Mode 사용
    print("\n🤖 앙상블 가중치 결정 중...")
    
    # [사용자 옵션] Manual Weight Mode
    # True: MANUAL_WEIGHTS 사용, False: scipy.optimize 사용
    MANUAL_MODE = True
    
    # 가중치 [RandomForest, XGBoost, LightGBM]
    # 합이 1.0이 되도록 설정
    MANUAL_WEIGHTS = [0.0, 1.0, 0.0]  # XGBoost Only
    
    def ensemble_rmse(weights):
        """
        앙상블 RMSE 계산 함수
        
        Args:
            weights (list): [RF, XGB, LGBM] 가중치
        
        Returns:
            float: Validation RMSE (거래금액 기준)
        """
        # 평당가 가중 평균
        final_pred_unit = (
            weights[0] * val_preds['rf'] + 
            weights[1] * val_preds['xgb'] + 
            weights[2] * val_preds['lgbm']
        )
        # 거래금액으로 변환
        final_pred_total = final_pred_unit * val_area
        return np.sqrt(mean_squared_error(y_val_total, final_pred_total))

    if MANUAL_MODE:
        print(f"🔧 수동 모드 활성화. 미리 설정된 가중치 사용: {MANUAL_WEIGHTS}")
        best_weights = np.array(MANUAL_WEIGHTS)
        best_rmse = ensemble_rmse(best_weights)
        
    else:
        # 자동 최적화 모드 (실험용)
        print("⚡ 자동 최적화 모드 활성화.")
        from scipy.optimize import minimize
        
        # 초기 가중치: 동일 가중치
        init_weights = [1/3, 1/3, 1/3]
        
        # 제약 조건: 가중치 합 = 1
        constraints = ({'type': 'eq', 'fun': lambda w: 1 - sum(w)})
        
        # 범위: 각 가중치는 0~1 사이
        bounds = [(0, 1), (0, 1), (0, 1)]
        
        # 최적화 실행
        result = minimize(ensemble_rmse, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        best_weights = result.x
        best_rmse = result.fun

    print(f"\n✅ 최종 가중치:")
    print(f"  - RandomForest : {best_weights[0]:.4f}")
    print(f"  - XGBoost      : {best_weights[1]:.4f}")
    print(f"  - LightGBM     : {best_weights[2]:.4f}")
    
    print(f"  👉 앙상블 Validation RMSE: {best_rmse:,.2f}")
    
    # ----------------------------------------------------------------
    # Step 7: 메트릭 저장 (Streamlit 앱용)
    # ----------------------------------------------------------------
    # [설명] 모델 성능 메트릭을 JSON 파일로 저장하여 Streamlit 앱에서 표시합니다.
    import json
    from datetime import datetime
    
    ensemble_metrics = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ensemble_rmse": best_rmse,
        "individual_rmse": {
            "RandomForest": rmse_rf,
            "XGBoost": rmse_xgb,
            "LightGBM": rmse_lgbm
        },
        "optimal_weights": {
            "RandomForest": best_weights[0],
            "XGBoost": best_weights[1],
            "LightGBM": best_weights[2]
        }
    }
    
    with open('ensemble_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(ensemble_metrics, f, ensure_ascii=False, indent=4)
    print("✅ 'ensemble_metrics.json' 메트릭 저장 완료")
    
    # ----------------------------------------------------------------
    # Step 8: 전체 데이터로 재학습 (Final Submission용)
    # ----------------------------------------------------------------
    # [설명] Validation Set을 포함한 전체 데이터로 다시 학습하여 성능을 극대화합니다.
    # - Validation Split은 모델 평가용이었고, 최종 제출은 모든 데이터 사용
    # - 더 많은 데이터 = 더 나은 성능
    print("\n🚀 전체 데이터로 재학습 중 (Final Submission용)...")
    
    # [Leakage Prevention] 'ym' 컬럼 제거 (시계열 분할용이었음)
    if 'ym' in X.columns:
        X = X.drop(columns=['ym'])
    
    # 가중치가 0보다 큰 모델만 재학습 (XGBoost만 학습)
    if best_weights[0] > 0:
        print("  - RandomForest 재학습 중...")
        rf.fit(X, y)
    
    if best_weights[1] > 0:
        print("  - XGBoost 재학습 중...")
        xgb.fit(X, y)  # 전체 데이터로 학습
        
    if best_weights[2] > 0:
        print("  - LightGBM 재학습 중...")
        lgbm.fit(X, y)
    
    # ----------------------------------------------------------------
    # Step 9: Feature Importance 추출 및 저장
    # ----------------------------------------------------------------
    # [설명] XGBoost의 Feature Importance를 추출하여 Streamlit 앱에서 표시합니다.
    # - Feature Importance: 각 특성이 모델 예측에 얼마나 기여하는지
    # - 상위 특성: '구', '위도', '경도', '연식' 등
    try:
        if best_weights[1] > 0:  # XGBoost 사용 시
            print("📊 Feature Importance 추출 및 저장 (XGBoost)...")
            fi = xgb.feature_importances_  # Feature Importance 배열
            fi_df = pd.DataFrame({'feature': X.columns, 'importance': fi})
            fi_df = fi_df.sort_values(by='importance', ascending=False)
            fi_df.to_csv('feature_importance.csv', index=False, encoding='utf-8-sig')
            print("✅ 'feature_importance.csv' 저장 완료")
        elif best_weights[0] > 0:  # RF fallback (사용 안 함)
            print("📊 Feature Importance 추출 및 저장 (RandomForest)...")
            fi = rf.feature_importances_
            fi_df = pd.DataFrame({'feature': X.columns, 'importance': fi})
            fi_df = fi_df.sort_values(by='importance', ascending=False)
            fi_df.to_csv('feature_importance.csv', index=False, encoding='utf-8-sig')
            print("✅ 'feature_importance.csv' 저장 완료")
    except Exception as e:
        print(f"⚠️ Feature Importance 저장 실패: {e}")

    return models, best_weights

# ==================================================================================
# 6. 예측 및 제출 파일 생성 함수
# ==================================================================================
def make_submission(models, best_weights, X_test):
    """
    제출 파일 생성 함수
    
    [설명]
    학습된 모델을 사용하여 Test 데이터를 예측하고 제출 파일을 생성합니다.
    
    [예측 과정]
    1. 각 모델로 '평당가' 예측
    2. 가중 평균으로 최종 '평당가' 계산
    3. '평당가' × '전용면적' = '거래금액' 복원
    4. 정수 변환 후 CSV 저장
    
    Args:
        models (dict): 학습된 모델 {'rf', 'xgb', 'lgbm'}
        best_weights (array): 최적 가중치 [RF, XGB, LGBM]
        X_test (DataFrame): 테스트 특성
    """
    print("\n🔮 테스트 데이터 예측 중...")
    
    # 최종 예측값 초기화
    final_pred_unit = np.zeros(len(X_test))
    
    # 1. Random Forest (가중치 0이면 스킵)
    if best_weights[0] > 0:
        print(f"  - RandomForest 예측 중 (가중치: {best_weights[0]:.2f})...")
        final_pred_unit += best_weights[0] * models['rf'].predict(X_test)
        
    # 2. XGBoost (Main Model)
    if best_weights[1] > 0:
        print(f"  - XGBoost 예측 중 (가중치: {best_weights[1]:.2f})...")
        final_pred_unit += best_weights[1] * models['xgb'].predict(X_test)
        
    # 3. LightGBM (가중치 0이면 스킵)
    if best_weights[2] > 0:
        print(f"  - LightGBM 예측 중 (가중치: {best_weights[2]:.2f})...")
        final_pred_unit += best_weights[2] * models['lgbm'].predict(X_test)
    
    # 평당가 → 거래금액 복원
    test_area = X_test['전용면적']
    pred_total = final_pred_unit * test_area
    
    # 정수형 변환 후 CSV 저장
    submission = pd.DataFrame({'target': pred_total.astype(int)})
    submission.to_csv('submission_ensemble_weighted.csv', index=False)
    print("\n✅ 'submission_ensemble_weighted.csv' 제출 파일 저장 완료!")

# ==================================================================================
# 7. 메인 실행 함수
# ==================================================================================
def main():
    """
    메인 실행 함수
    
    [실행 순서]
    1. 데이터 로드 (train.csv, test.csv)
    2. 전처리 및 특성 공학
    3. XGBoost 모델 학습
    4. 제출 파일 생성
    """
    print("\n" + "="*60)
    print("🚀 서울 아파트 실거래가 예측 - XGBoost 모델 학습 시작")
    print("="*60)
    
    # 1. 데이터 로드
    print("\n📁 [1/4] 데이터 로드 중...")
    train_org, test_org = load_data()
    
    # 2. 전처리 (특성 공학 포함)
    print("\n⚙️ [2/4] 데이터 전처리 및 특성 공학...")
    X_train, target, X_test = preprocess_data(train_org.copy(), test_org.copy())
    
    # 3. 모델 학습
    print("\n🤖 [3/4] XGBoost 모델 학습 중...")
    models, best_weights = train_ensemble_model(X_train, target)
    
    # 4. 제출 파일 생성
    print("\n📤 [4/4] 제출 파일 생성 중...")
    make_submission(models, best_weights, X_test)
    
    print("\n" + "="*60)
    print("🎉 모든 작업 완료!")
    print("="*60)


if __name__ == '__main__':
    main()
