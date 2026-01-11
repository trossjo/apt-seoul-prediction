
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Basic Configuration
st.set_page_config(
    page_title="아파트 실거래가 예측 모델 발표",
    page_icon="🏢",
    layout="wide"
)

# Korean Font Support for Matplotlib (Windows)
if os.name == 'nt':
    plt.rc('font', family='Malgun Gothic')
plt.rc('axes', unicode_minus=False)

# Custom Style
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    h1, h2, h3 {
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("🏙️ 서울시 아파트 실거래가 예측 모델 Optimization")
st.markdown("**목표**: RMSE(평균 제곱근 오차) 최소화 (Target: < 14,000)")

# -------------------------------------------------------------------------------------------
# [Load Data Function]
# -------------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    import gdown
    import tempfile
    import os as _os
    
    # Google Drive에서 데이터 로드 (Streamlit Cloud 배포용)
    # train.csv 파일 ID: 1yYgA8I-0VuQhdTAi1hQZVJ_zK9dy4Har
    train_file_id = "1yYgA8I-0VuQhdTAi1hQZVJ_zK9dy4Har"
    train_url = f"https://drive.google.com/uc?id={train_file_id}"
    
    # 임시 파일로 다운로드 후 읽기 (대용량 파일 바이러스 스캔 우회)
    temp_dir = tempfile.gettempdir()
    train_path = _os.path.join(temp_dir, "train.csv")
    
    if not _os.path.exists(train_path):
        gdown.download(train_url, train_path, quiet=False)
    
    train = pd.read_csv(train_path, low_memory=False)
    
    # Simple rename for display (Korean for Presentation)
    cols_mapping = {
        '시군구': '시군구',
        '아파트명': '아파트명',
        '전용면적(㎡)': '전용면적',
        '계약년월': '계약년월',
        '건축년도': '건축년도',
        '좌표X': 'longitude', # Keep for PyDeck
        '좌표Y': 'latitude',  # Keep for PyDeck
        'target': '거래금액'
    }
    train = train.rename(columns=cols_mapping)
    
    # Derived
    train['평당가'] = train['거래금액'] / train['전용면적']

    # Date Derivation (For Plots)
    train['계약년월'] = train['계약년월'].astype(str)
    train['거래년도'] = train['계약년월'].str[:4].astype(int)
    train['거래월'] = train['계약년월'].str[4:].astype(int)
    
    # Coordinates cleaning
    train['latitude'] = pd.to_numeric(train['latitude'], errors='coerce')
    train['longitude'] = pd.to_numeric(train['longitude'], errors='coerce')
    train = train.dropna(subset=['latitude', 'longitude'])
    
    # split sigungu for dong
    if '시군구' in train.columns:
        sigungu_split = train['시군구'].str.split(' ', expand=True)
        if sigungu_split.shape[1] >= 3:
            train['dong'] = sigungu_split[2]
        else:
            train['dong'] = 'Unknown'
            
    # Pre-format price for Tooltip
    train['unit_price_str'] = train['평당가'].apply(lambda x: f"{x:,.0f}")
    
    return train

with st.spinner('데이터 로딩 중... (처음 실행 시 시간이 소요됩니다)'):
    df = load_data()

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 프로젝트 개요", "📈 탐색적 데이터 분석 (EDA)", "🛠️ 특성 공학 (Feature Eng)", "🤖 모델링 전략", "🚀 최종 결과"])

# -------------------------------------------------------------------------------------------
# [1] Project Overview
# -------------------------------------------------------------------------------------------
with tab1:
    st.header("1. 프로젝트 개요 및 성과")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🎯 핵심 목표")
        st.markdown("""
        - **과제**: 서울시 아파트 실거래가 예측
        - **평가 지표**: RMSE (Root Mean Squared Error)
        - **데이터**: 
            - `train.csv` (1,118,822개 행)
            - `test.csv` (9,272개 행)
        """)
            
    st.divider()
    
    st.subheader("🔎 데이터 품질 점검 (Missing Value Analysis)")
    data = df # Use global loaded data
    
    col_mq1, col_mq2 = st.columns([1, 2])
    
    with col_mq1:
        st.markdown("**결측치 현황 (Missing Values)**")
        st.markdown("""
        - **초기 데이터**: 좌표(X, Y) 및 일부 아파트 정보 결측 존재
        - **조치**: 
            - 좌표 결측(X, Y): **Kakao API Geocoding**으로 100% 복원
            - 범주형(아파트명 등): **'Unknown'**으로 대체 (결측 자체를 하나의 정보로 활용)
            - 수치형(건축년도 등): **중앙값(Median)**으로 대체하여 분포 왜곡 방지
        """)
        
    with col_mq2:
        # Visualize Missing Values (Original) - Assuming we want to show 'What it was like' or 'Current State'
        # Since we load cleaned data in load_data, let's pretend or create a dummy series for demonstration if needed, 
        # or better, just show the cleanliness of 'load_data' output or raw checks.
        # However, presentation usually shows "Problem -> Solution". 
        # For now, let's show the columns that *had* issues.
        
        # Checking actual nulls in loaded data (which should be clean now)
        nulls = data.isnull().sum()
        nulls = nulls[nulls > 0]
        
        if len(nulls) > 0:
            st.warning("⚠️ 현재 데이터에 아직 결측치가 남아있습니다.")
            st.bar_chart(nulls)
        else:
            st.success("✅ 모든 중요 데이터(좌표 포함) 결측치 제거/보완 완료!")
            
            # Show what WAS missing (Simulation for presentation flow)
            example_nulls = pd.Series({
                '좌표X': 0, # Fixed
                '좌표Y': 0, # Fixed
                '도로명': 0, # Fixed
                '유형': 0
            })
            # st.bar_chart(example_nulls)
    
    with col2:
        st.markdown("### 🏆 Leaderboard Score 진행")
        scores = pd.DataFrame({
            '단계': [
                '(RF) 기본 모델 + 위치 군집화 (k=1000)', 
                '(RF) 타겟 변경(평당가) + 지리 정보', 
                '(RF) 교통 피처 추가 (버스/지하철)',
                '(XGB) 모델 변경 (1k/0.03)', 
                '(XGB) 파라미터 (3k/0.02)',
                '(XGB) 파라미터 (5k/0.02)',
                '(XGB) 파라미터 (5k/0.01)',
                '(XGB) 교통 피처 세분화'
            ],
            'Score (RMSE)': [
                "16,627", 
                "16,179", 
                "16,283", 
                "16,013", 
                "15,403",
                "15,469",
                "15,322",
                "🚀 15,114"
            ],
            '변화': [
                '-', 
                '▼ 448 (성능 향상)', 
                '▲ 104 (오히려 하락)', 
                '▼ 270 (XGB 전환 효과)', 
                '▼ 610 (최고 성능 달성)',
                '▲ (학습률 0.02 -> 0.01이 더 유리)',
                '▼ 147 (최고 성능 갱신)',
                '▼ 208 (교통 밀도, 거리 Clip 효과)'
            ]
        })
        st.dataframe(scores, use_container_width=True)
        
        st.info("💡 **Insight**: **교통 밀도 세분화(300/500/800m)**와 **거리 Clipping**이 모델의 과적합을 막고 일반화 성능을 크게 높였습니다.")



# -------------------------------------------------------------------------------------------
# [2] Preprocessing
# -------------------------------------------------------------------------------------------
with tab2:
    st.header("2. 탐색적 데이터 분석 (EDA)")
    
    st.markdown("데이터의 주요 패턴과 상관관계를 분석하여 모델링 전략을 수립했습니다.")
    st.divider()
    
    # -----------------------------------
    # 1. Price Analysis
    # -----------------------------------
    st.markdown("### 📈 가격 변동 및 추세 분석")
    col_p1, col_p2 = st.columns(2)
    
    # Price vs Age
    with col_p1:
        st.markdown("#### 🏗️ 건축년도별 평균 평당가")
        if '건축년도' in df.columns:
            age_price = df.groupby('건축년도')['평당가'].mean().reset_index()
            age_price = age_price[age_price['건축년도'] > 1900]
            
            fig_age, ax_age = plt.subplots(figsize=(6, 4))
            sns.lineplot(data=age_price, x='건축년도', y='평당가', ax=ax_age, marker='o', color='#2ca02c')
            ax_age.set_title("건축년도에 따른 평당가 (U자형 패턴)")
            ax_age.set_ylabel("평균 평당가 (만원)")
            ax_age.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig_age)
        else:
            st.warning("'건축년도' 데이터 없음")
            
    # Price vs Time (Transaction)
    with col_p2:
        st.markdown("#### 📅 거래 시점별 가격 추이")
        if '거래년도' in df.columns and '거래월' in df.columns:
            time_df = df.groupby(['거래년도', '거래월'])['평당가'].mean().reset_index()
            time_df['거래일자'] = pd.to_datetime(time_df['거래년도'].astype(str) + '-' + time_df['거래월'].astype(str) + '-01')
            
            fig_time, ax_time = plt.subplots(figsize=(6, 4))
            sns.lineplot(data=time_df, x='거래일자', y='평당가', ax=ax_time, marker='o', color='#d62728')
            ax_time.set_title("시기별 평당가 변동 (Time Series)")
            ax_time.set_ylabel("평균 평당가 (만원)")
            ax_time.grid(True, linestyle='--', alpha=0.6)
            plt.xticks(rotation=45)
            st.pyplot(fig_time)
        else:
            st.warning("'거래시점' 데이터 없음")
            
    st.info("💡 **Insight**: 구축(재건축)과 신축의 가격이 높고, 2022년 이후 하락 후 반등하는 추세가 뚜렷합니다.")

    st.divider()

    # -----------------------------------
    # 2. Correlation
    # -----------------------------------
    st.markdown("### 🔥 주요 변수 상관관계 (Correlation)")
    st.caption("모델 학습에 실제 기여도가 높은 핵심 변수와 타겟(평당가) 간의 관계를 분석합니다.")
    
    # 1. Base Dataframe
    df_corr = df.copy()
    if 'latitude' in df_corr.columns:
        df_corr = df_corr.rename(columns={'latitude': '위도', 'longitude': '경도'})
    
    # 2. Select Columns Strategy
    target_cols = ['평당가', '거래금액']
    selected_cols = []
    
    # Strategy A: Use Feature Importance if available
    fi_path = 'codes/feature_importance.csv'
    if os.path.exists(fi_path):
        try:
            fi_df = pd.read_csv(fi_path)
            # Get top 20 numeric features
            top_features = fi_df['feature'].head(20).tolist()
            # Map fi_df names to df_corr names (e.g., latitude -> 위도)
            name_map = {'latitude': '위도', 'longitude': '경도'}
            top_features = [name_map.get(f, f) for f in top_features]
            
            selected_cols = [c for c in top_features if c in df_corr.columns]
            if selected_cols:
                st.info("💡 **Feature Importance** 상위 변수를 기준으로 필터링했습니다.")
        except:
            pass
            
    # Strategy B: Fallback (Filter manually if A failed or empty)
    if not selected_cols:
        # Get all numeric
        all_numeric = df_corr.select_dtypes(include=[np.number]).columns.tolist()
        # Filter out noisy 'k-' columns and IDs
        selected_cols = [c for c in all_numeric if not c.startswith('k-') and 'id' not in c.lower() and 'unnamed' not in c.lower()]
        # If still too many, prioritize essentials
        if len(selected_cols) > 15:
             essentials = ['전용면적', '건축년도', '거래년도', '거래월', '층', '위도', '경도', '전체동수', '전체세대수', '주차대수']
             # Keep vars that match essentials roughly
             filtered = [c for c in selected_cols if any(x in c for x in essentials)]
             if len(filtered) >= 3:
                 selected_cols = filtered

    # Ensure targets are included
    final_cols = list(set(selected_cols + target_cols))
    
    # [Fix] Filter for numeric columns only (prevent 'ValueError: could not convert string to float')
    # Feature Importance file might imply 'Dong' or 'Apt' are important (Label Encoded in model), 
    # but here they are Strings. We must skip them for Correlation Matrix.
    numeric_cols_in_df = df_corr.select_dtypes(include=[np.number]).columns.tolist()
    final_cols = [c for c in final_cols if c in df_corr.columns and c in numeric_cols_in_df]
    
    # 3. Plot
    if len(final_cols) > 1:
        # [Presentation] Rename columns for cleaner display (Remove 'k-' prefix)
        display_df = df_corr[final_cols].copy()
        display_df.columns = [c.replace('k-', '').replace('K-', '') for c in display_df.columns]
        
        corr_mat = display_df.corr()
        
        # Sort by correlation with '평당가' for better readability
        if '평당가' in corr_mat.index:
            sorted_idx = corr_mat['평당가'].abs().sort_values(ascending=False).index
            corr_mat = corr_mat.loc[sorted_idx, sorted_idx]
            
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        # [Fix] k=1 removes diagonal from mask (Visible Diagonal)
        # Prevents "Empty First Row/Last Col" issue where only upper/lower triangle exists
        mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
        sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='coolwarm', ax=ax, vmin=-1, vmax=1, mask=mask)
        plt.title("핵심 변수 상관관계 분석")
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig_corr)
    else:
        st.warning("분석할 변수가 충분하지 않습니다.")


# -------------------------------------------------------------------------------------------
# [3] Geo Analysis (Visualization)
# -------------------------------------------------------------------------------------------
with tab3:
    st.header("3. 특성 공학 및 지리적 분석 (Feature Eng)")
    
    # ---------------------------------------------------------
    # 1. Preprocessing Section (Moved from Tab 2)
    # ---------------------------------------------------------
    st.markdown("### 🛠️ 데이터 전처리 (Data Preprocessing)")
    
    col_pre1, col_pre2 = st.columns(2)
    with col_pre1:
        with st.expander("1. 불필요한 변수 제거 & 타겟 변환", expanded=True):
            st.markdown("**1) Feature Selection**: 노이즈가 되는 27개 변수 제거 (k-전화번호, 관리비부과면적 등)")
            st.markdown("**2) Target Transformation**: `총 거래금액` -> `평당가`로 변환 (왜곡 방지)")
            st.code("train['평당가'] = train['거래금액'] / train['전용면적']", language='python')
            
    with col_pre2:
        with st.expander("2. 파생 변수 생성 (Derived Features)", expanded=True):
            st.markdown("- **연식(Age)**: `거래년도` - `건축년도` (구축/신축 여부)")
            st.markdown("- **시계열(Time)**: `거래년도`, `거래월` 수치형 변환")
            st.code("""
data['연식'] = data['거래년도'] - data['건축년도']
data['거래월'] = data['계약년월'].str[4:].astype(int)
            """, language='python')
            
    st.divider()
    
    # Preprocessing Metadata Header
    st.markdown("### 📋 상세 전처리 내역 (Metadata)")
    
    import json
    try:
        with open('codes/preprocessing_metadata.json', 'r', encoding='utf-8') as f:
            meta = json.load(f)
            
        c1, c2 = st.columns(2)
        with c1:
            with st.expander("🗑️ 제거된 변수 (Dropped Features)", expanded=False):
                st.write(meta.get('dropped_features', []))
            with st.expander("🔡 인코딩 방식 (Encoding)", expanded=False):
                st.json(meta.get('encoding', {}))
                
        with c2:
            with st.expander("🧩 결측치 처리 (Imputation)", expanded=False):
                st.json(meta.get('imputation', {}))
            with st.expander("🏁 최종 학습 변수 (Final Features)", expanded=False):
                st.write(meta.get('final_features', []))
            
    except Exception as e:
        st.info("⚠️ 전처리 메타데이터가 없습니다. (학습 후 생성됨: preprocessing_metadata.json)")
        
    st.divider()
    
    st.markdown("### 🧐 지역별 시세 분석 (Data Insight)")
    st.markdown("데이터를 분석해보면 **지역(구, 동)에 따른 평당 가격 편차**가 매우 큽니다.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### 🏛️ Top 10 비싼 '구' (Gu)")
        gu_data = pd.DataFrame({
            '구 (Gu)': ['강남구', '서초구', '송파구', '용산구', '성동구', '마포구', '광진구', '동작구', '중구', '강동구'],
            '평당가 (만원/㎡)': [1363, 1186, 1052, 1049, 869, 837, 808, 768, 763, 752]
        })
        st.dataframe(gu_data, use_container_width=True)
    
    with col_b:
        st.markdown("#### 🏘️ Top 10 비싼 '동' (Dong)")
        dong_data = pd.DataFrame({
            '구 (Gu)': ['종로구', '종로구', '종로구', '강남구', '강남구', '서초구', '중구', '종로구', '종로구', '송파구'],
            '동 (Dong)': ['신문로2가', '평동', '홍파동', '압구정동', '개포동', '반포동', '입정동', '교북동', '교남동', '잠실동'],
            '평당가 (만원/㎡)': [2317, 2149, 2039, 1730, 1719, 1633, 1627, 1536, 1500, 1468]
        })
        st.dataframe(dong_data, use_container_width=True)

    st.markdown("---")
    st.subheader("🤔 왜 단순히 '동별 평균 가격'을 학습시키지 않았나?")
    st.error("""
    **❌ 실험 결과: 단순 '동별 평균 가격' 피처 추가 시 과적합 발생**
    - **Validation RMSE**: 25,800 (매우 낮음, 학습 데이터 너무 잘 맞춤)
    - **Leaderboard Score**: 17,414 (오히려 Baseline보다 성능 하락)
    
    **원인**:
    1. **Data Leakage**: 타겟 정보를 그대로 피처로 쓰면 모델이 그 값에만 의존하게 됨.
    2. **행정구역의 한계**: 길 하나 사이로 동이 바뀌지만 생활권은 같은 경우가 많음. 행정구역 이름보다는 **'실제 물리적 위치'**가 중요함.
    """)
    
    st.success("""
    **✅ 해결책: K-Means Geo Clustering**
    - 행정구역 이름 대신 **위도/경도 좌표** 자체를 군집화했습니다.
    - **1000개의 미세 그룹**으로 나누어, 행정구역 경계를 넘어선 **'실질적인 입지 가치'**를 모델이 학습하도록 유도했습니다.
    """)
    
    st.markdown("#### 🗺️ 1000개 클러스터 시각화 (K-Means)")
    st.markdown("서울시 아파트를 1000개의 미세 생활권으로 나누어 시각화한 결과입니다.")
    
    # Run KMeans for Visualization (Cached)
    @st.cache_data
    def run_kmeans(data_df):
        coords = data_df[['longitude', 'latitude']]
        kmeans = KMeans(n_clusters=1000, random_state=42, n_init=10)
        data_df['cluster'] = kmeans.fit_predict(coords)
        return data_df

    with st.spinner('지리적 클러스터링 계산 중...'):
        geo_df = run_kmeans(df.sample(50000, random_state=42).copy()) # Sample for map speed

    # Calculate mean price per cluster for color intensity
    cluster_stats = geo_df.groupby('cluster')['평당가'].mean().reset_index()
    cluster_stats.columns = ['cluster', 'mean_price']
    
    # Normalize price for color mapping (0 to 255)
    cluster_stats['norm_price'] = (cluster_stats['mean_price'] - cluster_stats['mean_price'].min()) / \
                                  (cluster_stats['mean_price'].max() - cluster_stats['mean_price'].min())
    
    geo_df = geo_df.merge(cluster_stats, on='cluster')
    
    # ---------------------------------------------------------------------------------------
    # Map Visualization: Side-by-Side Comparison
    # ---------------------------------------------------------------------------------------
    st.markdown("#### 🗺️ 1000개 클러스터 vs 평당 가격 시각화")
    st.markdown("좌측은 **클러스터 구분**, 우측은 **평당 가격**입니다. 클러스터가 가격 분포를 얼마나 잘 반영하는지 비교해보세요.")

    # 1. Prepare Colors for Cluster Map (Random)
    np.random.seed(42)
    cluster_colors = {c: np.random.randint(0, 255, 3).tolist() for c in geo_df['cluster'].unique()}
    colors = geo_df['cluster'].map(cluster_colors).tolist()
    geo_df['c_r'] = [c[0] for c in colors]
    geo_df['c_g'] = [c[1] for c in colors]
    geo_df['c_b'] = [c[2] for c in colors]

    # 2. Prepare Colors for Price Map (Red-Blue Heatmap)
    geo_df['p_r'] = (geo_df['norm_price'] * 255).astype(int)
    geo_df['p_g'] = 50
    geo_df['p_b'] = ((1 - geo_df['norm_price']) * 255).astype(int)

    # Common View State
    view_state = pdk.ViewState( latitude=37.5665, longitude=126.9780, zoom=10, pitch=45, bearing=0 )

    # Common Tooltip (Apply validation logic inside load_data, assuming passed)
    tooltip = {
        "html": """
        <div style="padding: 10px; color: black; background-color: white; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.3); font-family: sans-serif;">
            <b>🏢 아파트:</b> {아파트명}<br/>
            <b>📍 구역(Dong):</b> {dong}<br/>
            <b>💰 평당가:</b> {unit_price_str} 만원/㎡<br/>
            <b>🧩 클러스터:</b> Group {cluster}
        </div>
        """,
        "style": {"color": "black"}
    }
    
    # Layout Columns
    map_col1, map_col2 = st.columns(2)

    # --- LEFT MAP: Cluster Groups ---
    with map_col1:
        st.subheader("🧩 클러스터 그룹 (Geo Groups)")
        layer_cluster = pdk.Layer(
            "ScatterplotLayer", geo_df,
            get_position=['longitude', 'latitude'],
            get_color=['c_r', 'c_g', 'c_b'],
            get_radius=80, pickable=True, opacity=0.7, stroked=True, filled=True, line_width_min_pixels=0
        )
        r_cluster = pdk.Deck( layers=[layer_cluster], initial_view_state=view_state, tooltip=tooltip, map_style='mapbox://styles/mapbox/light-v9' )
        st.pydeck_chart(r_cluster)
        st.caption("🎨 Random Colors: 1000개의 구역 구분")

    # --- RIGHT MAP: Price Level ---
    with map_col2:
        st.subheader("💸 평당 가격 (Price Level)")
        layer_price = pdk.Layer(
            "ScatterplotLayer", geo_df,
            get_position=['longitude', 'latitude'],
            get_color=['p_r', 'p_g', 'p_b'],
            get_radius=80, pickable=True, opacity=0.7, stroked=True, filled=True, line_width_min_pixels=0
        )
        r_price = pdk.Deck( layers=[layer_price], initial_view_state=view_state, tooltip=tooltip, map_style='mapbox://styles/mapbox/light-v9' )
        st.pydeck_chart(r_price)
        st.caption("🔴 Red: High Price, 🔵 Blue: Low Price")
    
    st.markdown("### 📐 회전 좌표계 (Rotated Coordinates)")
    st.markdown("트리 모델이 대각선 경계를 더 잘 학습하기 위해 좌표계를 45도 회전시킨 특성을 추가했습니다.")
    st.latex(r"x_{new} = lat + lon, \quad y_{new} = lat - lon")

    # ---------------------------------------------------------------------------------------
    # ---------------------------------------------------------------------------------------
    # 3. Transportation Strategy (Storytelling)
    # ---------------------------------------------------------------------------------------
    st.markdown("---")
    st.header("3. Feature Engineering: 교통 접근성 강화 전략")
    
    st.markdown("""
    초기 모델 분석 결과, 단순한 **'가장 가까운 역까지의 거리'**만으로는 서울의 복잡한 교통 입지를 설명하기 어려웠습니다.
    이에 따라 **'반경 내 밀도(Density)'**와 **'거리 제한(Clipping)'** 기법을 도입하여 성능을 극대화했습니다.
    """)
    
    col_str1, col_str2 = st.columns(2)
    
    with col_str1:
        st.error("❌ 기존 접근 방식 (Baseline)")
        st.markdown("""
        - **단순 거리 (Nearest Distance)**: 
            - 단순히 가장 가까운 역까지의 거리만 계산
            - **문제점**: 10km 떨어진 역도 '가장 가까운 역'으로 인식되어 집값에 불필요한 노이즈 발생 (Outlier)
        - **단순 개수**:
            - 반경 고려 없이 역세권 유무만 판단
        """)
        
    with col_str2:
        st.success("✅ 개선된 접근 방식 (Advanced)")
        st.markdown("""
        - **거리 Clipping (Distance Clipping)**:
            - 버스: `2km`, 지하철: `5km` 이상은 **동일하게 먼 것으로 간주** (영향력 차단)
        - **멀티 반경 밀도 (Multi-Radius Density)**:
            - **300m** (초역세권), **500m** (역세권), **800m** (도보권) 내 개수를 각각 산출하여 입지 가치 세분화
        - **가중치 점수 (Weighted Score)**:
            - 주요 노선(2호선, 9호선 등)에 가중치를 부여해 단순 개수보다 질적인 가치를 반영
        """)

    # Load Transport Data for Visualization (Google Drive)
    try:
        import gdown
        import tempfile
        import os as _os
        temp_dir = tempfile.gettempdir()
        
        # 버스 파일 ID: 1kObluIdbX0MnaWhoWn_i6PWRcEojf5id
        bus_path = _os.path.join(temp_dir, "bus_feature.csv")
        if not _os.path.exists(bus_path):
            gdown.download("https://drive.google.com/uc?id=1kObluIdbX0MnaWhoWn_i6PWRcEojf5id", bus_path, quiet=False)
        bus_df = pd.read_csv(bus_path)
        
        # 지하철 파일 ID: 15w1lH8jb1xtlT-qmn5CIEc3xIfDwFkmH
        sub_path = _os.path.join(temp_dir, "subway_feature.csv")
        if not _os.path.exists(sub_path):
            gdown.download("https://drive.google.com/uc?id=15w1lH8jb1xtlT-qmn5CIEc3xIfDwFkmH", sub_path, quiet=False)
        sub_df = pd.read_csv(sub_path)
        
        # Clean Coords
        bus_df = bus_df.rename(columns={'X좌표': 'lon', 'Y좌표': 'lat'})
        bus_df = bus_df.dropna(subset=['lat', 'lon'])
        sub_df = sub_df.rename(columns={'경도': 'lon', '위도': 'lat'})
        sub_df = sub_df.dropna(subset=['lat', 'lon'])

        st.markdown("### 🗺️ 교통 인프라 시각화 (Infrastructure Map)")
        
        col_t1, col_t2 = st.columns([3, 1])
        with col_t1:
            view_state_trans = pdk.ViewState(latitude=37.5665, longitude=126.9780, zoom=10.5, pitch=0)
            
            # Layer: Subway (Blue)
            layer_sub = pdk.Layer(
                "ScatterplotLayer", sub_df,
                get_position=['lon', 'lat'],
                get_color=[0, 0, 255, 180],
                get_radius=150,
                pickable=True
            )
            
            # Layer: Bus (Green, smaller)
            layer_bus = pdk.Layer(
                "ScatterplotLayer", bus_df,
                get_position=['lon', 'lat'],
                get_color=[0, 255, 100, 80],
                get_radius=30,
                pickable=True
            )
            
            r_trans = pdk.Deck(
                layers=[layer_bus, layer_sub],
                initial_view_state=view_state_trans,
                map_style='mapbox://styles/mapbox/light-v9',
                tooltip={"html": "<b>{역사명}</b><br/>{호선}" if '역사명' in sub_df.columns else "Transport"}
            )
            st.pydeck_chart(r_trans)
            
        with col_t2:
            st.info("데이터 분포 확인")
            st.caption(f"🚇 지하철역: {len(sub_df):,}개")
            st.caption(f"🚌 버스정류장: {len(bus_df):,}개")
            st.markdown("---")
            st.write("서울 전역에 촘촘히 분포된 버스 정류장과 주요 거점인 지하철역의 분포를 확인했습니다.")

    except Exception as e:
        st.warning(f"교통 데이터 로드 실패: {e}")

# ---------------------------------------------------------------------------------------


# -------------------------------------------------------------------------------------------
# [5] Modeling Strategy
# -------------------------------------------------------------------------------------------
with tab4:
    st.header("4. 모델링 및 타겟 최적화 전략")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💡 Unit Price (평당 단가) 전략")
        st.markdown("**'총 거래금액' 대신 '전용면적당 단가'를 예측**합니다.")
        

        st.markdown("""
        - 평형별 가격 차이가 크므로, **단위 면적당 가격(Target)을 예측**하고,
        - 나중에 `전용면적`을 곱해 `총 거래금액`을 복원하는 전략이 오차를 줄입니다.
        """)
        
    with col2:
        st.subheader("📅 검증 전략 (Time Series Split)")
        st.error("❌ 기존: Random Split (20%)")
        st.caption("과거와 미래 데이터가 섞여 시계열 특성 반영 불가 (Leakage 위험)")
        
        st.success("✅ 변경: 최근 3개월 분리 (Time Cutoff)")
        st.markdown("""
        ```text
        [학습 데이터 (Train)] (~ 2023.06)      | [검증 (Val)] (2023.07 ~ 09)
        ███████████████████████████████████ | ░░░░░░
                                            ▲ Cutoff Point
        ```
        """)
        st.caption("미래를 예측하는 과제 특성에 맞춰 **'과거 데이터로 학습 -> 미래 데이터로 검증'**하는 파이프라인 구축")

    st.divider()
    
    st.markdown("### 🤖 모델 선택 (Model Selection)")
    
    st.info("""초기에는 Baseline의 Random Forest를 사용했으나, 앙상블 실험(RF + XGBoost + LightGBM) 결과 
    오히려 성능이 저하되어 **XGBoost 단일 모델**에 집중하는 전략으로 전환했습니다.""")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    
    with col_m1:
        st.markdown("#### 1. Random Forest")
        st.markdown("**역할**: Baseline (초기 접근)")
        st.markdown("""
        **장점**:
        - 과적합에 강함 (분산↓)
        - 안정적인 성능 보장
        - 여러 트리의 평균으로 예측
        
        **단점**:
        - 각 트리가 독립적으로 학습
        - **잔여 오차(Residual)를 집요하게 학습하지 못함**
        - 성능 한계 명확
        """)
        
    with col_m2:
        st.markdown("#### 2. XGBoost (Main) 🏆")
        st.markdown("**역할**: 최종 선택 모델")
        st.markdown("""
        **강점**:
        - **Gradient Boosting**: 이전 트리의 오차를 다음 트리가 학습 → 잔여 오차 집요하게 감소
        - **정규화 (L1/L2 + Pruning)**: 과적합 방지
        - **GPU 가속**: 110만 건 대용량 데이터 빠른 학습
        - **비선형 패턴**: U자형 건축년도, 대각선 지리 패턴 효과적 학습
        
        **결과**:
        - 3개 모델 중 **압도적 성능**
        - 앙상블보다 단일 모델이 더 우수
        """)

    with col_m3:
        st.markdown("#### 3. LightGBM (Sub)")
        st.markdown("**역할**: 앙상블 실험용")
        st.markdown("""
        **장점**:
        - 대용량 데이터 학습 속도 압도적
        - 빠른 실험 반복 가능
        
        **앙상블 실험 결과**:
        - RF + XGB + LGBM (1/3 가중평균)
        - **예상**: 각 모델 약점 상쇄
        - **실제**: 오히려 성능 저하 ❌
        - **원인**: 모델 간 예측 패턴 차이로 노이즈 증가
        
        **최종 결정**: XGBoost Only
        """)
    
    st.warning("""💡 **교훈**: 많은 모델을 섞는다고 무조건 좋은 것이 아닙니다. 
    **가장 강력한 하나를 극대화**하는 것이 더 효과적일 수 있습니다.""")

    # -----------------------------------------------------------------
    # (New) Ensemble Performance Section (Displayed if available)
    # -----------------------------------------------------------------
    st.markdown("---")
    st.header("🏆 앙상블 모델 성능 (RF + XGB + LGBM)")
    
    metric_path = 'codes/ensemble_metrics.json'
    if os.path.exists(metric_path):
        with open(metric_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        
        # 1. Summary Metrics
        st.info(f"📅 학습 완료 시간: {metrics.get('timestamp', 'N/A')}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="🔥 최종 앙상블 RMSE", value=f"{metrics['ensemble_rmse']:,.0f}")
        with col2:
            best_single_model = min(metrics['individual_rmse'], key=metrics['individual_rmse'].get)
            best_single_rmse = metrics['individual_rmse'][best_single_model]
            st.metric(label=f"🥇 최고 단일 모델 ({best_single_model})", value=f"{best_single_rmse:,.0f}", 
                      delta=f"{metrics['ensemble_rmse'] - best_single_rmse:,.0f} (Improvement)")
        
        # 2. Individual Model Performance Chart
        st.subheader("📊 모델별 성능 비교 (RMSE 낮을수록 좋음)")
        rmses = metrics['individual_rmse']
        rmses['Ensemble (Weighted)'] = metrics['ensemble_rmse']
        
        rmse_df = pd.DataFrame(list(rmses.items()), columns=['Model', 'RMSE'])
        rmse_df = rmse_df.sort_values('RMSE', ascending=False)
        
        # Color highlight for Ensemble
        colors = ['#d3d3d3'] * len(rmse_df)
        rmse_df = rmse_df.reset_index(drop=True)
        try:
            ens_idx = rmse_df[rmse_df['Model'] == 'Ensemble (Weighted)'].index[0]
            colors[ens_idx] = '#ff4b4b' # Red for Ensemble
        except:
            pass
        
        fig_rmse, ax_rmse = plt.subplots(figsize=(10, 4))
        sns.barplot(data=rmse_df, x='RMSE', y='Model', palette=colors, ax=ax_rmse)
        ax_rmse.set_xlabel("Validation RMSE (Total Price)")
        for i, v in enumerate(rmse_df['RMSE']):
            ax_rmse.text(v, i, f" {v:,.0f}", va='center', fontweight='bold')
        st.pyplot(fig_rmse)

        # 3. Optimal Weights Chart
        st.subheader("⚖️ 최적 앙상블 가중치 (Optimal Weights)")
        weights = metrics['optimal_weights']
        weights = {k: v for k, v in weights.items() if v > 0.001}
        
        if weights:
            fig_w, ax_w = plt.subplots(figsize=(6, 6))
            ax_w.pie(weights.values(), labels=weights.keys(), autopct='%1.1f%%', 
                     startangle=140, colors=['#66b3ff','#99ff99','#ffcc99'])
            ax_w.set_title("Model Contribution Weights")
            st.pyplot(fig_w)
        else:
            st.warning("가중치 정보를 불러올 수 없습니다.")

        # ---------------------------------------------------------
        # Experiment History Log
        # ---------------------------------------------------------
        st.markdown("---")
        st.subheader("📉 실험 이력 (Experiment Log)")
        st.markdown("다양한 하이퍼파라미터 조합에 따른 성능 변화 기록입니다.")
        
        exp_data = {
            "Model": ["XGBoost Only", "XGBoost Only", "XGBoost Only", "Ensemble (Mix)"],
            "Params": ["n_est=5000, lr=0.01", "n_est=5000, lr=0.02", "n_est=3000, lr=0.02", "RF+XGB+LGBM"],
            "RMSE (LB/Val)": ["🚀 15,114 (New Best!)", "15,469", "15,403", "❌ 17,500+"],
            "Note": ["Transport Refinement + Clip", "Learning Rate 0.02 too high", "Good Baseline", "Overfitting"]
        }
        exp_df = pd.DataFrame(exp_data)
        st.table(exp_df)

        st.markdown("---")
        st.info("""
        **💡 참고: 이 점수는 어떻게 나오나요? (Validation RMSE)**
        
        이 점수는 **'모의고사 점수'**입니다. 
        단, 시계열 데이터의 특성을 반영하기 위해 무작위 분할이 아닌 **'마지막 3개월 (Time Series Split)'** 데이터를 검증 셋으로 사용했습니다.
        
        - **Validation Set (Last 3 Months)**: 가장 최근 경향을 테스트 (미래 예측 시뮬레이션)
        - **Test Set**: 리더보드 제출용
        
        따라서 이 점수가 잘 나오면, 실제 리더보드(미래 데이터) 성적도 좋을 가능성이 높습니다.
        """)
        
    else:
        st.warning("⚠️ 앙상블 학습 결과 파일 ('ensemble_metrics.json')이 없습니다. train_ensemble.py를 먼저 실행해주세요.")

# -------------------------------------------------------------------------------------------
# [6] Final Results
# -------------------------------------------------------------------------------------------
with tab5:
    st.header("5. 최종 결과 및 제언")
    
    st.balloons()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Final Leaderboard", "15,114", "Best Score")
    col2.metric("Validation RMSE", "12,200", "Last 3 Months")
    col3.metric("Improvement", "1,513 ▼", "from Baseline(16,627)")
    
    st.divider()
    st.subheader("📊 모델이 주목한 핵심 변수 (Top Features)")
    try:
        fi_df = pd.read_csv('codes/feature_importance.csv')
        fig_fi = plt.figure(figsize=(10, 5))
        sns.barplot(data=fi_df.head(15), x='importance', y='feature', palette='viridis')
        plt.title("Top 15 Feature Importance")
        st.pyplot(fig_fi)
    except:
        st.info("특성 중요도 데이터가 아직 없습니다. (학습 후 생성됨)")
    
    st.markdown("### ✅ 결론 (Conclusion)")
    st.markdown("""
    1. **Unit Price Target**: 단순 가격 예측보다 평당 단가 예측이 훨씬 효과적이었습니다.
    2. **Refined Geo Features**: 단순 거리뿐만 아니라 '300m/500m/800m' 등 세분화된 밀도 피처가 성능 향상의 핵심이었습니다.
    3. **Future Work**: 이제 **Bayesian Optimization (Optuna)**를 활용해 XGBoost의 하이퍼파라미터를 정밀하게 튜닝하면 RMSE 14,000점대 진입도 가능할 것입니다.
    """)

# Footer
st.sidebar.markdown("---")
