
import pandas as pd
import numpy as np
import optuna
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import BallTree
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------
# [1] Preprocessing Functions (Reused from train_ensemble.py)
# --------------------------------------------------------------------------------

def label_encode_train_test(train, test, col):
    """
    Train 데이터로 fit 하고, Train/Test 모두 transform.
    Test에만 있는 새로운 라벨은 -1로 처리.
    """
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    
    # 1. Fit on Train
    # train[col]의 고유값들을 정렬하여 fitting (결정론적 동작 보장)
    unique_vals = sorted(list(train[col].astype(str).unique()))
    le.fit(unique_vals)
    
    # 2. Transform Train
    train[col] = le.transform(train[col].astype(str))
    
    # 3. Transform Test (Handle UnseenLabels -> -1)
    # le.classes_ 에 없는 값은 -1로 매핑하기 위해 map 사용
    le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
    test[col] = test[col].astype(str).map(le_dict).fillna(-1).astype(int)
    
    return train, test, le

def preprocess_data(train, test):
    print("Preprocessing data...")
    
    # 1. Column Renaming to English
    cols_mapping = {
        '시군구': 'sigungu',
        '아파트명': 'apt',
        '전용면적(㎡)': '전용면적',
        '계약년월': '계약년월',
        '계약일': '계약일',
        '층': '층',
        '건축년도': '건축년도',
        '도로명': 'road_name',
        'target': '거래금액',  # Keep as '거래금액' for '평당가' calculation
        '좌표X': '경도', # d:/bootcamp2/code/train_ensemble.py aligns with this
        '좌표Y': '위도'
    }
    train = train.rename(columns=cols_mapping)
    test = test.rename(columns=cols_mapping)
    
    # 2. Drop High Cardinality / Unused Columns
    cols_to_drop = [
        '해제사유발생일', '단지소개기존clob', 'k-관리비부과면적', 'k-전용면적별세대현황(60㎡이하)',
        'k-전용면적별세대현황(60㎡~85㎡이하)', 'k-85㎡~135㎡이하', 'k-135㎡초과', '건축면적',
        'K-전화번호', 'K-팩스번호', 'k-세대타입(분양형태)', 'k-관리방식', 'k-복도유형',
        'k-난방방식', 'k-사용검사일-사용승인일', 'k-홈페이지', 'k-등록일자', 'k-수정일자',
        '고용보험관리번호', '경비비관리형태', '세대전기계약방법', '청소비관리형태',
        '기타/의무/임대/임의=1/2/3/4', '단지승인일', '사용허가여부', '관리비 업로드', '단지신청일'
    ]
    train = train.drop(columns=cols_to_drop, errors='ignore')
    test = test.drop(columns=cols_to_drop, errors='ignore')
    
    # 3. Geo & Address Features on Full Data (Row-Independent)
    # Combine for non-leakage operations
    train['is_train'] = 1
    test['is_train'] = 0
    target_col = '거래금액'  # Updated to match renamed column
    
    # Sort columns to align
    common_cols = [c for c in train.columns if c in test.columns and c != target_col and c != 'is_train']
    data = pd.concat([train[common_cols + ['is_train', target_col]], test[common_cols + ['is_train']]], axis=0).reset_index(drop=True)
    
    # 3-1. Address Parsings
    print("Processing address info...")
    data['gu'] = data['sigungu'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else 'Unknown')
    data['dong'] = data['sigungu'].apply(lambda x: x.split()[2] if len(x.split()) > 2 else 'Unknown')
    data['contract_ym'] = data['계약년월'].astype(str)
    
    # 3-2. Date Features
    data['거래년도'] = data['contract_ym'].str[:4].astype(int)
    data['거래월'] = data['contract_ym'].str[4:].astype(int)
    data['건축년도'] = data['건축년도'].fillna(data['건축년도'].median())
    data['age'] = data['거래년도'] - data['건축년도']
    
    # --------------------------------------------------------------------------------
    # 4. Transport Features (Using BallTree) - [UPDATED with Multi-Radius & Clipping]
    # --------------------------------------------------------------------------------
    try:
        print("Generating Enhanced Geo Features...")
        from sklearn.neighbors import BallTree
        print("  Loading Transport Data (Bus & Subway)...")
        
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
        
        # 1. 거리 (Nearest Distance)
        print("  - Calculating Nearest Distances...")
        
        # 1-1. Bus
        tree_bus = BallTree(bus_rad, metric='haversine')
        dist_bus, _ = tree_bus.query(data_rad, k=1)
        data['dist_to_bus'] = (dist_bus[:, 0] * 6371000).astype(np.float32).clip(0, 2000)
        
        # 1-2. Subway
        tree_sub = BallTree(sub_rad, metric='haversine')
        dist_sub, _ = tree_sub.query(data_rad, k=1)
        data['dist_to_subway'] = (dist_sub[:, 0] * 6371000).astype(np.float32).clip(0, 5000)

        # 2. 밀도 (Count within Radius) - [Refined] Multiple Radii
        print("  - Calculating Counts within Radius (Bus: 300/500/800, Sub: 300/500/800/1200)...")
        
        # 2-1. Bus (300, 500, 800)
        for r in [300, 500, 800]:
            radius_rad = r / 6371000
            count = tree_bus.query_radius(data_rad, r=radius_rad, count_only=True)
            data[f'bus_cnt_{r}'] = count
        
        # 2-2. Subway (300, 500, 800, 1200)
        for r in [300, 500, 800, 1200]:
            radius_rad = r / 6371000
            count = tree_sub.query_radius(data_rad, r=radius_rad, count_only=True)
            data[f'sub_cnt_{r}'] = count
        
        # 3. 접근성 점수 (Weighted Score)
        print("  - Calculating Subway Accessibility Score (Line Weights)...")
        
        def get_line_weight(line_name):
            if any(l in line_name for l in ['2호선', '9호선', '신분당', '분당선']):
                return 2.0
            elif any(l in line_name for l in ['3호선', '7호선', '5호선']):
                return 1.5
            elif any(l in line_name for l in ['1호선', '4호선', '6호선', '8호선']):
                return 1.2
            return 1.0
            
        sub['weight'] = sub['호선'].fillna('').apply(get_line_weight)
        
        # Weight Calculation using 1200m radius (largest)
        radius_sub_max = 1200 / 6371000
        
        # Group 1: High (2.0)
        sub_high = sub[sub['weight'] == 2.0]
        tree_high = BallTree(np.radians(sub_high[['위도', '경도']].values), metric='haversine')
        cnt_high = tree_high.query_radius(data_rad, r=radius_sub_max, count_only=True)
        
        # Group 2: Mid (1.5)
        sub_mid = sub[sub['weight'] == 1.5]
        tree_mid = BallTree(np.radians(sub_mid[['위도', '경도']].values), metric='haversine')
        cnt_mid = tree_mid.query_radius(data_rad, r=radius_sub_max, count_only=True)
        
        # Group 3: Low (Others)
        sub_low = sub[~sub.index.isin(sub_high.index) & ~sub.index.isin(sub_mid.index)]
        tree_low = BallTree(np.radians(sub_low[['위도', '경도']].values), metric='haversine')
        cnt_low = tree_low.query_radius(data_rad, r=radius_sub_max, count_only=True)
        
        # Final Score
        data['sub_score_1200'] = (cnt_high * 2.0) + (cnt_mid * 1.5) + (cnt_low * 1.0)
        
        print(f"  > Added features: dist_to_bus/subway (clipped), bus_cnt_(300/500/800), sub_cnt_(300/500/800/1200), sub_score_1200")
        
    except Exception as e:
        print(f"⚠️ Failed to add transport features: {e}")
        new_cols = ['dist_to_bus', 'dist_to_subway', 'sub_score_1200'] + \
                   [f'bus_cnt_{r}' for r in [300, 500, 800]] + \
                   [f'sub_cnt_{r}' for r in [300, 500, 800, 1200]]
        for c in new_cols:
            data[c] = 0

    if '경도' in data.columns and '위도' in data.columns:
        # 4-1. 회전 좌표계 (Row Independent)
        data['회전좌표X'] = data['위도'] + data['경도']
        data['회전좌표Y'] = data['위도'] - data['경도']
        
    # --------------------------------------------------------------------------------
    # [Leakage Prevention] Split Train/Test HERE for Data-Dependent Operations
    # --------------------------------------------------------------------------------
    train_part = data[data['is_train'] == 1].copy()
    test_part = data[data['is_train'] == 0].copy()
    
    # K-Means (Fit Train -> Predict Test)
    if '경도' in train_part.columns and '위도' in train_part.columns:
        print("  Running MiniBatchKMeans (k=1000) on Train Only...")
        coords_tr = train_part[['경도', '위도']].fillna(0).values
        coords_te = test_part[['경도', '위도']].fillna(0).values
        
        kmeans = MiniBatchKMeans(n_clusters=1000, random_state=42, batch_size=4096, n_init=10)
        train_part['클러스터'] = kmeans.fit_predict(coords_tr).astype(str)
        test_part['클러스터'] = kmeans.predict(coords_te).astype(str)
        
    # Imputation (Fit Train -> Fill Both)
    print("Imputing numeric missing values (Train Median)...")
    num_cols = train_part.select_dtypes(include=[np.number]).columns
    cols_to_fill = [c for c in num_cols if c != target_col and c != '평당가' and c != 'is_train']
    
    for col in cols_to_fill:
        med = train_part[col].median()
        train_part[col] = train_part[col].fillna(med)
        test_part[col] = test_part[col].fillna(med)
        
    # Target (Unit Price) - Debug and Create
    print(f"DEBUG: Checking for '평당가' creation...")
    print(f"  '평당가' in columns: {'평당가' in train_part.columns}")
    print(f"  '거래금액' in columns: {'거래금액' in train_part.columns}")
    print(f"  '전용면적' in columns: {'전용면적' in train_part.columns}")
    
    if '거래금액' in train_part.columns and '전용면적' in train_part.columns:
        print(f"  Creating '평당가' = '거래금액' / '전용면적'")
        train_part['평당가'] = train_part['거래금액'] / train_part['전용면적']
        print(f"  '평당가' created successfully. Sample values: {train_part['평당가'].head(3).tolist()}")
    else:
        print(f"  ERROR: Cannot create '평당가' - missing required columns!")
        print(f"  Available columns: {train_part.columns.tolist()}")


    # Label Encoding (Strict Fit on Train)
    st_cols = ['sigungu', 'dong', 'apt', 'road_name', '클러스터', 'gu']
    print("Label Encoding (Strict Fit on Train)...")
    
    for c in st_cols:
        if c in train_part.columns:
            train_part, test_part, _ = label_encode_train_test(train_part, test_part, c)

    # 4. Extract Target BEFORE dropping columns
    y_train = train_part['평당가']
    
    # 5. Feature Selection
    # Drop strings, helper columns, and target-related columns
    # First, drop known helper columns
    drop_final = ['sigungu', 'dong', 'apt', 'road_name', '계약년월', '계약일', 'contract_ym', 'is_train', '거래금액', '평당가', 'gu']
    
    X_train = train_part.drop(columns=drop_final, errors='ignore')
    X_test = test_part.drop(columns=drop_final, errors='ignore')
    
    # Then, automatically drop ALL remaining object-type columns (strings not encoded)
    object_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        print(f"  Dropping {len(object_cols)} remaining object columns: {object_cols[:5]}...")  # Show first 5
        X_train = X_train.drop(columns=object_cols)
        X_test = X_test.drop(columns=object_cols)
    
    print(f"  Final feature count: {X_train.shape[1]}")
    print(f"  Final dtypes: {X_train.dtypes.value_counts().to_dict()}")
    
    # 6. Log Transform
    y_train = np.log1p(y_train)
    
    return X_train, y_train, X_test

# --------------------------------------------------------------------------------
# [2] Optuna Objective Function
# --------------------------------------------------------------------------------

def objective(trial, X, y):
    # 1. Suggest Parameters
    param = {
        'tree_method': 'hist',
        'device': 'cuda',  # Use GPU
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'n_estimators': trial.suggest_int('n_estimators', 2000, 6000, step=500),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 6, 14),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 20.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0)
    }
    
    # 2. Time Series Split (Last 3 Months)
    # Reconstruct 'ym' for split if not present in X (it was likely dropped)
    # Wait, X might not have '거래년도' if dropped? 
    # Check if '거래년도', '거래월' are in X. They should be if not dropped.
    
    # X should have '거래년도', '거래월'.
    ym = X['거래년도']*100 + X['거래월']
    val_cutoff = ym.sort_values().unique()[-3]
    
    mask_val = ym >= val_cutoff
    mask_tr = ~mask_val
    
    X_tr = X[mask_tr] #.drop(columns=['거래년도', '거래월']) # Keep features if useful
    y_tr = y[mask_tr]
    X_val = X[mask_val]
    y_val = y[mask_val]
    
    # 3. Model Training
    model = XGBRegressor(**param, early_stopping_rounds=100)
    
    # Modern XGBoost API: early_stopping_rounds is now a constructor parameter
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=0
    )
    
    # 4. Evaluation
    preds = model.predict(X_val)
    
    # Inverse Log for RMSE Calculation (Real Scale)
    # Note: y_tr, y_val are log1p transformed.
    # preds is log scale.
    # To compare with Leaderboard, we need Real Scale Total Price RMSE.
    # But for optimization stability, minimizing Log Scale RMSE is often enough.
    # However, user cares about the 15,114 number (Total Price RMSE).
    # Let's metric on Total Price RMSE to be aligned with Leaderboard.
    
    # Recover Area
    val_area = X_val['전용면적'].values
    true_price = np.expm1(y_val) * val_area
    pred_price = np.expm1(preds) * val_area
    
    rmse = np.sqrt(mean_squared_error(true_price, pred_price))
    
    return rmse

# --------------------------------------------------------------------------------
# [3] Main Execution
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    # 1. Load Data
    print("Loading data...")
    train = pd.read_csv('../data/train.csv', low_memory=False)
    test = pd.read_csv('../data/test.csv', low_memory=False)
    
    # 2. Preprocess
    X, y, _ = preprocess_data(train, test)
    
    print(f"\n[Optuna] Starting Hyperparameter Tuning...")
    print(f"  - Target: Minimize Validation RMSE (Last 3 Months Split)")
    print(f"  - Search Space: n_est(2k-6k), lr(0.005-0.05), depth(6-14), etc.")
    
    # 3. Create Study
    study = optuna.create_study(direction='minimize')
    
    # 4. Optimize
    # N_TRIALS: Adjust based on time. 20 trials * 1 min each ~ 20 min.
    # User might want to see results faster? Let's try 10-20.
    n_trials = 20 
    
    try:
        study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
        
        print("\n==================================================")
        print("🎉 Optimization Completed!")
        print("==================================================")
        print(f"🏆 Best Mean RMSE: {study.best_value:,.0f}")
        print("🧩 Best Params:")
        for k, v in study.best_params.items():
            print(f"  - {k}: {v}")
            
        # Save Best Params
        import json
        with open('best_xgb_params.json', 'w') as f:
            json.dump(study.best_params, f, indent=4)
        print("💾 Best params saved to 'best_xgb_params.json'")
            
    except KeyboardInterrupt:
        print("\n⚠️ Tuning interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during tuning: {e}")
