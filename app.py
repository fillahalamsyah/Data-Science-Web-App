import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from utils.data_loader import DataLoader
from utils.data_analyzer import DataAnalyzer
from utils.preprocessor import DataPreprocessor
from utils.model_manager import ModelManager
from home import show_home

st.set_page_config(
    page_title="Data Science Web App",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .step-header {
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        padding: 0.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Navigation
st.sidebar.title("🧭 Navigasi")
st.sidebar.markdown("---")

# Main navigation
nav_option = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Home", "📊 Data Science Pipeline", "📚 Artikel Data Science"],
    index=0
)

if nav_option == "🏠 Home":
    show_home()

elif nav_option == "📚 Artikel Data Science":
    st.title("📚 Artikel Data Science")
    st.info("🚧 Halaman artikel sedang dalam pengembangan. Coming soon!")
    
    # Placeholder for articles
    st.markdown("""
    ### 📖 Artikel yang Akan Tersedia:
    
    1. **Pengantar Data Science** - Dasar-dasar dan overview
    2. **Data Collection & Cleaning** - Teknik pengumpulan dan pembersihan data
    3. **Exploratory Data Analysis** - Analisis eksploratori mendalam
    4. **Data Preprocessing** - Persiapan data untuk modeling
    5. **Feature Engineering** - Teknik rekayasa fitur
    6. **Machine Learning Fundamentals** - Dasar-dasar ML
    7. **Model Evaluation** - Evaluasi dan validasi model
    8. **Hyperparameter Tuning** - Optimisasi parameter
    9. **Deep Learning Basics** - Pengenalan deep learning
    10. **Data Visualization** - Teknik visualisasi data
    11. **Ethics in Data Science** - Etika dalam data science
    """)

elif nav_option == "📊 Data Science Pipeline":
    st.markdown('<h1 class="main-header">🔬 Data Science Pipeline</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.1rem; color: #666;'>
            Platform lengkap untuk pembelajaran dan implementasi Data Science - 
            Dari loading data hingga deployment model
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline steps navigation
    st.sidebar.markdown("### 🔄 Pipeline Steps")
    steps = [
        "1. 📊 Load/Generate/Upload Dataset",
        "2. 🔍 Exploratory Data Analysis (EDA)", 
        "3. 🛠️ Preprocessing",
        "4. 📐 Dimensionality Reduction",
        "5. 🤖 Model Training & Evaluation",
        "6. 💾 Export Dataset/Model"
    ]
    step = st.sidebar.radio("Pilih tahap:", steps)

    # Session state initialization
    session_vars = ["df", "df_processed", "target", "model", "X_train", "X_test", "y_train", "y_test", "model_name", "best_params"]
    for var in session_vars:
        if var not in st.session_state:
            st.session_state[var] = None

    # Helper function
    def st_input(key, func, *args, **kwargs):
        return func(*args, key=key, **kwargs)

    # Step 1: Load/Generate/Upload Dataset
    if step == steps[0]:
        st.markdown('<div class="step-header"><h2>📊 Step 1: Load/Generate/Upload Dataset</h2></div>', unsafe_allow_html=True)
        
        # Dataset Information Panel
        with st.expander("📚 Informasi Dataset Yang Tersedia"):
            dataset_info = DataLoader.get_dataset_info()
            for name, info in dataset_info.items():
                st.write(f"**{name.upper()}**: {info['deskripsi']}")
                st.write(f"- Fitur: {info['fitur']}, Sampel: {info['sampel']}, Kelas: {info['kelas']}")
                st.write(f"- Use Case: {info['use_case']}")
                st.write("")

        data_source = st.radio("Pilih sumber data:", ["Dataset scikit-learn", "Generate dataset", "Upload file"])
        
        if data_source == "Dataset scikit-learn":
            dataset_name = st.selectbox("Pilih dataset", ["iris", "wine", "breast_cancer", "digits"])
            if st.button("Load Dataset"):
                try:
                    data = DataLoader.load_sklearn_dataset(dataset_name)
                    df = data.frame
                    st.session_state.df = df
                    st.session_state.target = data.target.name if hasattr(data.target, 'name') else "target"
                    st.success("Dataset berhasil dimuat!")
                    
                    # Enhanced dataset information
                    summary = DataAnalyzer.get_data_summary(df)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Jumlah Sampel", summary["shape"][0])
                    with col2:
                        st.metric("Jumlah Fitur", summary["shape"][1] - 1)
                    with col3:
                        st.metric("Jumlah Kelas", df[st.session_state.target].nunique())
                    
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
        
        elif data_source == "Generate dataset":
            with st.expander("📖 Penjelasan Jenis Dataset Generated"):
                st.write("""
                **Classification**: Dataset untuk masalah klasifikasi dengan fitur yang dapat memisahkan kelas
                **Regression**: Dataset untuk masalah regresi dengan hubungan linear/non-linear
                **Blobs**: Dataset clustering dengan grup-grup data yang terpisah jelas
                """)
            
            col1, col2 = st.columns(2)
            with col1:
                gen_type = st.selectbox("Jenis dataset", ["Classification", "Regression", "Blobs"])
                n_samples = st.number_input("Jumlah sampel", 100, 10000, 200)
                n_features = st.number_input("Jumlah fitur", 2, 50, 4)
            
            with col2:
                kwargs = {}
                if gen_type == "Classification":
                    kwargs['n_classes'] = st.number_input("Jumlah kelas", 2, 10, 3)
                    kwargs['n_redundant'] = st.number_input("Fitur redundan", 0, n_features//2, 0)
                    kwargs['n_informative'] = st.number_input("Fitur informatif", 1, n_features, min(n_features, 3))
                elif gen_type == "Blobs":
                    kwargs['n_centers'] = st.number_input("Jumlah cluster", 2, 10, 3)
                    kwargs['cluster_std'] = st.number_input("Standar deviasi cluster", 0.1, 5.0, 1.0)
            
            if st.button("Generate Dataset"):
                try:
                    df = DataLoader.generate_dataset(gen_type, n_samples, n_features, **kwargs)
                    st.session_state.df = df
                    st.session_state.target = "target"
                    st.success("Dataset berhasil digenerate!")
                    
                    # Show generation summary
                    summary = DataAnalyzer.get_data_summary(df)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Sampel", n_samples)
                    with col2:
                        st.metric("Fitur", n_features)
                    with col3:
                        if gen_type in ["Classification", "Blobs"]:
                            st.metric("Kelas/Cluster", df["target"].nunique())
                        else:
                            st.metric("Target Range", f"{df['target'].min():.2f} - {df['target'].max():.2f}")
                        
                except Exception as e:
                    st.error(f"Error generating dataset: {e}")
        
        elif data_source == "Upload file":
            uploaded = st.file_uploader("Upload file (csv, xlsx, parquet)", type=["csv", "xlsx", "parquet"])
            if uploaded:
                try:
                    df = DataLoader.load_uploaded_file(uploaded)
                    st.session_state.df = df
                    st.session_state.target = st.selectbox("Pilih kolom target", df.columns)
                    st.success("Dataset berhasil diupload!")
                except Exception as e:
                    st.error(f"Error loading file: {e}")

        # Display dataset preview and manipulation tools
        if st.session_state.df is not None:
            st.write("Preview dataset:")
            st.dataframe(st.session_state.df.head())
            
            # Data quality assessment
            st.subheader("🔍 Data Quality Assessment")
            summary = DataAnalyzer.get_data_summary(st.session_state.df)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Missing Values", summary["missing_values"])
            with col2:
                st.metric("Duplicates", summary["duplicates"])
            with col3:
                st.metric("Memory Usage (KB)", f"{summary['memory_usage']:.2f}")

    # Step 2: EDA & Visualisasi
    elif step == steps[1]:
        st.markdown('<div class="step-header"><h2>🔍 Step 2: Exploratory Data Analysis (EDA)</h2></div>', unsafe_allow_html=True)
        if st.session_state.df is not None:
            df = st.session_state.df
            analyzer = DataAnalyzer()
            
            # Enhanced EDA with tabs
            tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                "📊 Overview", "📈 Distribusi", "🔗 Korelasi", "📉 Boxplot", "📊 Pairplot", "🧮 Value Counts", "🔬 Scatter 2 Fitur"])
            with tab7:
                st.subheader("Scatterplot Dua Fitur Numerik")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        col_x = st.selectbox("Fitur X", numeric_cols, key="scatter_x")
                    with col2:
                        col_y = st.selectbox("Fitur Y", [c for c in numeric_cols if c != col_x], key="scatter_y")
                # Batasi jumlah titik yang diplot
                max_points = 1000
                if len(df) > max_points:
                    st.warning(f"Data terlalu banyak (> {max_points} baris), hanya menampilkan sample {max_points} data.")
                    plot_df = df[[col_x, col_y]].sample(max_points, random_state=42)
                else:
                    plot_df = df[[col_x, col_y]]
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots()
                ax.scatter(plot_df[col_x], plot_df[col_y], alpha=0.7, s=10)
                ax.set_xlabel(col_x)
                ax.set_ylabel(col_y)
                ax.set_title(f"Scatterplot: {col_x} vs {col_y}")
                st.pyplot(fig)
                # Tampilkan nilai korelasi
                corr_val = plot_df[[col_x, col_y]].corr().iloc[0,1]
                st.info(f"Korelasi Pearson: {corr_val:.3f}")
            
            with tab1:
                st.subheader("Dataset Overview")
                summary = analyzer.get_data_summary(df)
                
                # Display summary metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Ringkasan Dataset:**")
                    summary_df = pd.DataFrame({
                        "Metrik": ["Jumlah Baris", "Jumlah Kolom", "Fitur Numerik", "Fitur Kategorikal", "Missing Values", "Duplikat"],
                        "Nilai": [
                            int(summary["shape"][0]), 
                            int(summary["shape"][1]), 
                            int(summary["numeric_cols"]), 
                            int(summary["categorical_cols"]), 
                            int(summary["missing_values"]), 
                            int(summary["duplicates"])
                        ]
                    })
                    st.dataframe(summary_df.set_index("Metrik"))
                
                with col2:
                    st.write("**Statistik Deskriptif:**")
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        st.dataframe(df[numeric_cols].describe().style.background_gradient(axis=1))
                    else:
                        st.info("Tidak ada fitur numerik")

                # Tabel kolom, tipe data, dan info lainnya
                st.write("**Tabel Kolom & Tipe Data:**")
                col_info = pd.DataFrame({
                    'Kolom': df.columns,
                    'Tipe Data': df.dtypes.astype(str).values,
                    'Jumlah Unik': [df[c].nunique() for c in df.columns],
                    'Jumlah Missing': [df[c].isnull().sum() for c in df.columns],
                    'Contoh Nilai': [df[c].dropna().unique()[:3] for c in df.columns]
                })
                st.dataframe(col_info.set_index('Kolom'))
                
                # Visualisasi tambahan: Bar Plot statistik setiap fitur numerik
                st.subheader("📊 Visualisasi Statistik Fitur Numerik")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    # Pilihan statistik
                    stat_option = st.selectbox(
                        "Pilih statistik untuk dibandingkan:",
                        ('mean', 'median', 'std', 'min', 'max'),
                        key='stat_comparison'
                    )

                    # Kalkulasi statistik yang dipilih
                    stats_data = df[numeric_cols].agg(stat_option)

                    # Plotting
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(figsize=(10, 4))
                    stats_data.plot(kind='bar', ax=ax, color='skyblue')
                    
                    # Judul dan label dinamis
                    title_map = {
                        'mean': 'Rata-rata',
                        'median': 'Median',
                        'std': 'Standar Deviasi',
                        'min': 'Nilai Minimum',
                        'max': 'Nilai Maksimum'
                    }
                    ax.set_title(f"{title_map.get(stat_option, stat_option.capitalize())} Setiap Fitur Numerik")
                    ax.set_ylabel(title_map.get(stat_option, stat_option.capitalize()))
                    ax.tick_params(axis='x', rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Tidak ada fitur numerik untuk divisualisasikan.")
                
            
            with tab2:
                st.subheader("Analisis Distribusi")
                feature_to_analyze = st.selectbox("Pilih fitur untuk distribusi", df.columns)
                if feature_to_analyze:
                    fig = analyzer.plot_distribution(df, feature_to_analyze)
                    st.pyplot(fig)
                    # Outlier analysis
                    if pd.api.types.is_numeric_dtype(df[feature_to_analyze]):
                        try:
                            outlier_analysis = analyzer.get_outlier_analysis(df, feature_to_analyze)
                            if outlier_analysis:
                                st.write("**Analisis Outlier:**")
                                outlier_df = pd.DataFrame({
                                    "Metrik": ["Q1", "Q3", "IQR", "Lower Bound", "Upper Bound", "Outlier Count", "Outlier %"],
                                    "Nilai": [
                                        float(outlier_analysis["Q1"]), 
                                        float(outlier_analysis["Q3"]), 
                                        float(outlier_analysis["IQR"]), 
                                        float(outlier_analysis["lower_bound"]), 
                                        float(outlier_analysis["upper_bound"]),
                                        int(outlier_analysis["outlier_count"]), 
                                        f"{outlier_analysis['outlier_percentage']:.2f}%"
                                    ]
                                })
                                st.dataframe(outlier_df.set_index("Metrik"))
                        except Exception as e:
                            st.warning(f"Tidak dapat menganalisis outlier: {e}")

            with tab3:
                st.subheader("Analisis Korelasi")
                
                # Correlation matrix
                fig = analyzer.plot_correlation_matrix(df)
                if fig:
                    st.pyplot(fig)
                    
                    # Strong correlations
                    try:
                        strong_corr = analyzer.find_strong_correlations(df)
                        if strong_corr:
                            st.write("**Korelasi Kuat (|r| > 0.7):**")
                            strong_corr_df = pd.DataFrame(strong_corr)
                            # Ensure proper data types
                            strong_corr_df['Correlation'] = strong_corr_df['Correlation'].astype(float)
                            st.dataframe(strong_corr_df.style.background_gradient(subset=['Correlation']))
                        else:
                            st.info("Tidak ada korelasi kuat ditemukan")
                    except Exception as e:
                        st.warning(f"Tidak dapat menganalisis korelasi: {e}")

            with tab4:
                st.subheader("Boxplot Fitur Numerik")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    box_col = st.selectbox("Pilih fitur numerik untuk boxplot", numeric_cols)
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    ax.boxplot(df[box_col].dropna(), vert=False)
                    ax.set_title(f"Boxplot: {box_col}")
                    st.pyplot(fig)
                else:
                    st.info("Tidak ada fitur numerik untuk boxplot.")

            with tab5:
                st.subheader("Pairplot (Scatterplot Matrix)")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    import seaborn as sns
                    import matplotlib.pyplot as plt
                    pairplot_cols = st.multiselect("Pilih fitur untuk pairplot", numeric_cols, default=list(numeric_cols)[:4])
                    if len(pairplot_cols) > 1:
                        fig = sns.pairplot(df[pairplot_cols].dropna())
                        st.pyplot(fig)
                    else:
                        st.info("Pilih minimal 2 fitur untuk pairplot.")
                else:
                    st.info("Tidak cukup fitur numerik untuk pairplot.")

            with tab6:
                st.subheader("Value Counts Fitur Kategorikal")
                cat_cols = df.select_dtypes(include=["object", "category"]).columns
                if len(cat_cols) > 0:
                    cat_col = st.selectbox("Pilih fitur kategorikal", cat_cols)
                    value_counts = df[cat_col].value_counts(dropna=False)
                    st.bar_chart(value_counts)
                    st.dataframe(value_counts.rename("Jumlah").reset_index().rename(columns={"index": cat_col}))
                else:
                    st.info("Tidak ada fitur kategorikal.")

        else:
            st.warning("Silakan load/generate/upload dataset terlebih dahulu.")

    # Step 3: Preprocessing  
    elif step == steps[2]:
        st.markdown('<div class="step-header"><h2>🛠️ Step 3: Preprocessing</h2></div>', unsafe_allow_html=True)
        if st.session_state.df is not None:
            df = st.session_state.df.copy()
            preprocessor = DataPreprocessor()
            
            st.write("Preview data sebelum preprocessing:")
            st.dataframe(df.head())
            
            # Preprocessing tabs
            tabs = st.tabs(["🔧 Imputasi", "🏷️ Encoding", "📏 Scaling", "📊 Hasil"])
            
            with tabs[0]:
                st.subheader("Imputasi Nilai Hilang")
                missing_analysis = DataAnalyzer.analyze_missing_values(df)
                
                if len(missing_analysis) > 0:
                    st.write("**Analisis Missing Values:**")
                    st.dataframe(missing_analysis.style.background_gradient(subset=['Missing %']))
                    
                    # Show recommendations
                    recommendations = preprocessor.get_imputation_recommendations(df)
                    st.write("**Rekomendasi Strategi:**")
                    st.dataframe(recommendations.set_index('Kolom'))
                    
                    # Imputation interface
                    target_col = st.session_state.target
                    impute_cols = st.multiselect("Kolom untuk imputasi", missing_analysis['Kolom'].tolist())
                    
                    if impute_cols:
                        strategy = st.selectbox("Strategi imputasi", ["mean", "median", "most_frequent", "constant"])
                        fill_value = st.text_input("Nilai pengganti (untuk constant)", "0") if strategy == "constant" else None
                        
                        if st.button("Lakukan Imputasi"):
                            try:
                                df, _ = preprocessor.impute_missing_values(df, impute_cols, strategy, fill_value)
                                st.success("✅ Imputasi selesai!")
                            except Exception as e:
                                st.error(f"❌ Imputasi gagal: {e}")
                else:
                    st.success("✅ Tidak ada missing values!")
            
            with tabs[1]:
                st.subheader("Encoding Fitur Kategorikal")
                cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
                
                if cat_cols:
                    target_col = st.session_state.target
                    cat_cols_sel = st.multiselect("Kolom untuk encoding", 
                                                [col for col in cat_cols if col != target_col])
                    
                    if cat_cols_sel:
                        encoding_type = st.selectbox("Jenis encoding", ["label", "onehot", "ordinal"])
                        
                        encoding_info = {
                            "label": "Mengkonversi kategori menjadi angka (0,1,2...)",
                            "onehot": "Membuat kolom binary untuk setiap kategori", 
                            "ordinal": "Mirip label encoding dengan urutan tertentu"
                        }
                        st.info(f"ℹ️ {encoding_info[encoding_type]}")
                        
                        if st.button("Lakukan Encoding"):
                            try:
                                df, _ = preprocessor.encode_categorical(df, cat_cols_sel, encoding_type)
                                st.success("✅ Encoding selesai!")
                            except Exception as e:
                                st.error(f"❌ Encoding gagal: {e}")
                else:
                    st.success("✅ Tidak ada fitur kategorikal!")
            
            with tabs[2]:
                st.subheader("Scaling Fitur Numerik")
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                
                if num_cols:
                    target_col = st.session_state.target
                    cols_for_scaling = [col for col in num_cols if col != target_col]
                    num_cols_sel = st.multiselect("Kolom untuk scaling",
                                                cols_for_scaling,
                                                default=cols_for_scaling)
                    
                    if num_cols_sel:
                        scaler_type = st.selectbox("Jenis scaler", ["standard", "minmax", "robust"])
                        
                        scaler_info = {
                            "standard": "Normalisasi dengan mean=0, std=1",
                            "minmax": "Scaling ke range 0-1",
                            "robust": "Menggunakan median dan IQR (tahan outlier)"
                        }
                        st.info(f"ℹ️ {scaler_info[scaler_type]}")
                        
                        if st.button("Lakukan Scaling"):
                            try:
                                df, _ = preprocessor.scale_features(df, num_cols_sel, scaler_type)
                                st.success("✅ Scaling selesai!")
                            except Exception as e:
                                st.error(f"❌ Scaling gagal: {e}")
                else:
                    st.info("Tidak ada fitur numerik untuk scaling")
            
            with tabs[3]:
                st.subheader("📊 Hasil Preprocessing")
                st.session_state.df_processed = df
                
                # Show comparison
                if st.session_state.df is not None:
                    summary = preprocessor.get_preprocessing_summary(st.session_state.df, df)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Dataset Original:**")
                        orig_df = pd.DataFrame({
                            'Metrik': ['Baris', 'Kolom', 'Missing Values', 'Kategorikal'],
                            'Nilai': [summary['original_shape'][0], summary['original_shape'][1],
                                    summary['original_missing'], summary['original_categorical']]
                        })
                        st.dataframe(orig_df.set_index('Metrik'))
                    
                    with col2:
                        st.write("**Dataset Processed:**") 
                        proc_df = pd.DataFrame({
                            'Metrik': ['Baris', 'Kolom', 'Missing Values', 'Kategorikal'],
                            'Nilai': [summary['processed_shape'][0], summary['processed_shape'][1],
                                    summary['processed_missing'], summary['processed_categorical']]
                        })
                        st.dataframe(proc_df.set_index('Metrik'))
                
                st.write("**Preview data setelah preprocessing:**")
                st.dataframe(df.head())

        else:
            st.warning("Silakan load/generate/upload dataset terlebih dahulu.")

    # Step 4: Dimensionality Reduction
    elif step == steps[3]:
        st.markdown('<div class="step-header"><h2>📐 Step 4: Dimensionality Reduction</h2></div>', unsafe_allow_html=True)
        
        # Import the enhanced module
        from utils.dimensionality_reducer import DimensionalityReducer
        
        # Enhanced dimensionality reduction guide
        with st.expander("📖 Panduan Lengkap Reduksi Dimensi"):
            st.write("""
            **Reduksi Dimensi** adalah teknik untuk mengurangi jumlah fitur sambil mempertahankan informasi penting:
            
            **Kapan menggunakan:**
            - Dataset dengan banyak fitur (>10-20)
            - Visualisasi data multidimensi
            - Mengurangi overfitting dan noise
            - Mempercepat training model
            - Mengatasi curse of dimensionality
            
            **Kategori Metode:**
            - **Linear Transformation**: PCA, LDA, Factor Analysis, SVD, ICA
            - **Non-linear Manifold**: t-SNE, Isomap, LLE, MDS, Spectral Embedding
            - **Feature Selection**: Univariate, RFE, Model-based
            """)
        
        if "df_processed" not in st.session_state or st.session_state.df_processed is None:
            st.warning("Silakan lakukan preprocessing pada Step 3 terlebih dahulu.")
        else:
            df = st.session_state.df_processed.copy()
            reducer = DimensionalityReducer()
            
            # Dimensionality analysis
            st.subheader("📊 Analisis Dimensionalitas Dataset")
            target_col = st.session_state.get('target')
            analysis = reducer.analyze_dimensionality(df, target_col)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Fitur", len(df.columns))
            with col2:
                st.metric("Fitur Numerik", analysis["n_features"])
            with col3:
                st.metric("Sampel", analysis["n_samples"])
            with col4:
                st.metric("Rasio Sampel/Fitur", f"{analysis['ratio']:.1f}")
            
            # Show recommendations
            st.write("**Rekomendasi:**")
            for rec in analysis["recommendations"]:
                st.write(f"- {rec}")
            
            # Show high correlations if any
            if "high_correlations" in analysis:
                with st.expander("🔍 Pasangan Fitur Berkorelasi Tinggi"):
                    corr_df = pd.DataFrame(analysis["high_correlations"], 
                                         columns=["Fitur 1", "Fitur 2", "Korelasi"])
                    st.dataframe(corr_df.style.background_gradient(subset=["Korelasi"]))
            
            if analysis["n_features"] < 2:
                st.error("❌ Minimal 2 fitur numerik diperlukan untuk reduksi dimensi.")
            else:
                # Method selection with enhanced interface
                st.subheader("🎛️ Pilih Metode Reduksi Dimensi")
                
                # Organize methods by category
                categories = reducer.get_method_categories()
                method_info = reducer.get_method_info()
                
                # Category selection
                selected_category = st.selectbox("Pilih Kategori Metode:", list(categories.keys()))
                available_methods = categories[selected_category]
                
                # Method selection within category
                method = st.selectbox("Pilih Metode:", available_methods)
                
                # Display method information
                info = method_info[method]
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**{info['type']}**\n\n{info['description']}")
                with col2:
                    st.write(f"**Terbaik untuk:** {info['best_for']}")
                    st.write(f"**Kelebihan:** {info['pros']}")
                    st.write(f"**Kekurangan:** {info['cons']}")
                
                st.success(f"💡 **Kapan menggunakan:** {info['when_use']}")
                
                # Feature selection
                all_numeric_features = [col for col in df.select_dtypes(include=[np.number]).columns if col != target_col]
                features_for_reduction = st.multiselect(
                    "Pilih fitur yang akan direduksi", 
                    all_numeric_features, 
                    default=all_numeric_features,
                    help="Pilih fitur numerik yang ingin direduksi. Default: semua fitur numerik kecuali target."
                )
                if not features_for_reduction:
                    st.error("Pilih minimal satu fitur untuk reduksi dimensi.")
                    st.stop()                
                # Parameter configuration
                st.subheader("⚙️ Konfigurasi Parameter")
                
                # Number of components/features
                if method in ["Univariate Selection", "Recursive Feature Elimination", "Model-based Selection"]:
                    max_features = len(features_for_reduction)
                    n_components = st.number_input("Jumlah fitur yang dipilih", 1, max_features, min(10, max_features))
                else:
                    if method == "LDA" and target_col and target_col in df.columns:
                        n_classes = df[target_col].nunique()
                        max_components = min(len(features_for_reduction), n_classes - 1)
                        if max_components < 1:
                            st.error(f"LDA tidak dapat dijalankan. Jumlah kelas ({n_classes}) tidak mencukupi.")
                            st.stop()
                        n_components = st.number_input("Jumlah komponen", 1, max_components, min(2, max_components))
                    else:
                        max_comp = min(len(features_for_reduction), 10)
                        n_components = st.number_input("Jumlah komponen", 1, max_comp, min(2, max_comp))
                
                # Method-specific parameters
                params = {}
                param_cols = st.columns(3)
                
                if method == "t-SNE":
                    with param_cols[0]:
                        params['perplexity'] = st.slider("Perplexity", 5, 50, 30,
                            help="Balances local vs global aspects. Higher = more neighbors considered")
                    with param_cols[1]:
                        params['learning_rate'] = st.select_slider("Learning Rate", 
                            options=[10, 50, 100, 200, 500, 1000], value=200,
                            help="Controls optimization speed")
                    
                    st.info("**t-SNE Tips:** Perplexity 5-15 untuk dataset kecil, 30-50 untuk dataset sedang")
                    
                elif method in ["Isomap", "LLE", "Spectral Embedding"]:
                    with param_cols[0]:
                        params['n_neighbors'] = st.slider("Number of Neighbors", 2, 50, 5,
                            help="Number of neighbors for local neighborhood graph")
                    st.info("**Manifold Tips:** Fewer neighbors = preserve local structure, More neighbors = global view")
                
                elif method == "Kernel PCA":
                    with param_cols[0]:
                        params['kernel'] = st.selectbox("Kernel", ['rbf', 'linear', 'poly', 'sigmoid'])
                    with param_cols[1]:
                        if params['kernel'] == 'rbf':
                            params['gamma'] = st.select_slider("Gamma", options=[0.001, 0.01, 0.1, 1, 'scale'], value='scale')
                
                elif method == "Model-based Selection":
                    with param_cols[0]:
                        params['threshold'] = st.selectbox("Threshold", ['mean', 'median', '0.1*mean', '0.5*mean', '2*mean'])
                
                # Check if method requires target
                if info['supervised'] and (not target_col or target_col not in df.columns):
                    st.error(f"❌ {method} memerlukan kolom target. Silakan atur di step sebelumnya.")
                else:
                    # Perform dimensionality reduction
                    if st.button("🚀 Jalankan Reduksi Dimensi"):
                        with st.spinner(f"Menjalankan {method}..."):
                            try:
                                result = reducer.perform_dimensionality_reduction(
                                    df, method, features_for_reduction, target_col, n_components, **params
                                )
                            except Exception as e:
                                result = {"success": False, "message": f"Error: {e}"}
                        
                        if result["success"]:
                            st.success(result["message"])
                            
                            # Store results
                            st.session_state["dimred_result"] = result
                            st.session_state["dimred_method"] = method
                            st.session_state["dimred_features"] = features_for_reduction
                            
                            # Display results with tabs
                            result_tabs = st.tabs(["📊 Hasil", "📈 Visualisasi", "🔍 Detail", "💾 Export"])
                            
                            with result_tabs[0]:
                                st.subheader("Ringkasan Hasil")
                                
                                # Basic information
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Dimensi Asli", len(features_for_reduction))
                                with col2:
                                    if result["X_reduced"] is not None:
                                        new_dim = result["X_reduced"].shape[1]
                                        st.metric("Dimensi Baru", new_dim)
                                        reduction_pct = (1 - new_dim/len(features_for_reduction)) * 100
                                with col3:
                                    if result["X_reduced"] is not None:
                                        st.metric("Reduksi", f"{reduction_pct:.1f}%")
                                
                                # Method-specific information
                                additional_info = result["additional_info"]
                                if "explained_variance_ratio" in additional_info:
                                    st.write("**Explained Variance:**")
                                    var_df = pd.DataFrame({
                                        "Component": [f"Comp {i+1}" for i in range(len(additional_info["explained_variance_ratio"]))],
                                        "Individual (%)": (additional_info["explained_variance_ratio"] * 100).round(2),
                                        "Cumulative (%)": (additional_info.get("cumulative_variance", np.cumsum(additional_info["explained_variance_ratio"])) * 100).round(2)
                                    })
                                    st.dataframe(var_df.set_index("Component"))
                                
                                elif "selected_features" in additional_info:
                                    st.write("**Fitur Terpilih:**")
                                    selected_features = additional_info["selected_features"]
                                    st.write(f"Dari {len(features_for_reduction)} fitur, terpilih {len(selected_features)} fitur:")
                                    
                                    if "scores" in additional_info:
                                        # For univariate selection
                                        score_df = pd.DataFrame({
                                            "Fitur": selected_features,
                                            "Score": [additional_info["scores"][features_for_reduction.index(f)] for f in selected_features],
                                            "P-value": [additional_info["pvalues"][features_for_reduction.index(f)] for f in selected_features]
                                        }).sort_values("Score", ascending=False)
                                        st.dataframe(score_df.set_index("Fitur"))
                                    
                                    elif "feature_importances" in additional_info:
                                        # For model-based selection
                                        importance_df = pd.DataFrame({
                                            "Fitur": features_for_reduction,
                                            "Importance": additional_info["feature_importances"]
                                        }).sort_values("Importance", ascending=False)
                                        
                                        # Highlight selected features
                                        def highlight_selected(row):
                                            return ['background-color: lightgreen' if row.name in selected_features else '' for _ in row]
                                        
                                        st.dataframe(importance_df.set_index("Fitur").style.apply(highlight_selected, axis=1))
                                        st.caption("Fitur terpilih ditandai dengan background hijau")
                        
                            with result_tabs[1]:
                                st.subheader("Visualisasi Hasil")
                                
                                if result["X_reduced"] is not None:
                                    # Main visualization
                                    y_for_plot = df[target_col] if target_col and target_col in df.columns else None
                                    fig_main = reducer.plot_reduction_results(
                                        result["X_reduced"], y_for_plot, method, n_components, result["feature_names"]
                                    )
                                    if fig_main:
                                        st.pyplot(fig_main)
                                    
                                    # Explained variance plot
                                    if "explained_variance_ratio" in additional_info:
                                        fig_var = reducer.plot_explained_variance(additional_info, method)
                                        if fig_var:
                                            st.pyplot(fig_var)
                            
                            with result_tabs[2]:
                                st.subheader("Informasi Detail")
                                
                                # Component analysis for linear methods
                                if "components" in additional_info:
                                    fig_heatmap = reducer.plot_component_heatmap(
                                        additional_info, features_for_reduction, method
                                    )
                                    if fig_heatmap:
                                        st.pyplot(fig_heatmap)
                                    
                                    # Component interpretation
                                    st.write("**Interpretasi Komponen:**")
                                    components = additional_info["components"]
                                    for i, component in enumerate(components):
                                        st.write(f"**Component {i+1}:**")
                                        # Find top contributing features
                                        top_features = np.argsort(np.abs(component))[-5:][::-1]
                                        for j, feat_idx in enumerate(top_features):
                                            feature_name = features_for_reduction[feat_idx]
                                            contribution = component[feat_idx]
                                            st.write(f"  {j+1}. {feature_name}: {contribution:.3f}")
                                
                                
                                # Additional method-specific info
                                if method == "t-SNE" and "kl_divergence" in additional_info:
                                    st.write(f"**KL Divergence:** {additional_info['kl_divergence']:.4f}")
                                    st.write(f"**Iterations:** {additional_info['n_iter']}")
                                
                                elif method == "Isomap" and "reconstruction_error" in additional_info:
                                    st.write(f"**Reconstruction Error:** {additional_info['reconstruction_error']:.4f}")
                                
                                elif method == "MDS" and "stress" in additional_info:
                                    st.write(f"**Stress:** {additional_info['stress']:.4f}")
                        
                        with result_tabs[3]:
                            st.subheader("Export Hasil")
                            
                            if result["X_reduced"] is not None:
                                # Create reduced dataset
                                if method in ["Univariate Selection", "Recursive Feature Elimination", "Model-based Selection"]:
                                    # For feature selection, use original feature names
                                    df_reduced = pd.DataFrame(result["X_reduced"], 
                                                            columns=result["feature_names"],
                                                            index=df.index)
                                else:
                                    # For transformation methods, use new component names
                                    df_reduced = pd.DataFrame(result["X_reduced"],
                                                            columns=result["feature_names"],
                                                            index=df.index)
                                
                                # Add back non-numeric columns and target
                                other_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
                                if target_col and target_col not in other_cols:
                                    other_cols.append(target_col)
                                
                                if other_cols:
                                    df_final = pd.concat([df_reduced, df[other_cols]], axis=1)
                                else:
                                    df_final = df_reduced
                                
                                st.write("**Preview data setelah reduksi dimensi:**")
                                st.dataframe(df_final.head(10))
                                
                                # Export options
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    if st.button("💾 Gunakan untuk Training Model"):
                                        st.session_state.df_processed = df_final
                                        st.success("✅ Data berhasil diperbarui! Lanjutkan ke Step 5 untuk training model.")
                                        st.balloons()
                                
                                with col2:
                                    csv_reduced = df_final.to_csv(index=False).encode()
                                    st.download_button(
                                        "📥 Download CSV",
                                        csv_reduced,
                                        f"dataset_{method.lower().replace(' ', '_')}_reduced.csv",
                                        "text/csv"
                                    )

    # Step 5: Model Training & Evaluation
    elif step == steps[4]:
        
        # Enhanced model training guide
        with st.expander("📖 Panduan Model Training & Evaluation"):
            st.write("""
            **Model Training & Evaluation** adalah tahap inti dalam machine learning:
            
            **Tahapan:**
            1. **Pemilihan Model**: Pilih algoritma yang sesuai dengan problem dan data
            2. **Hyperparameter Tuning**: Optimasi parameter untuk performa terbaik
            3. **Training**: Latih model dengan data training
            4. **Evaluation**: Evaluasi performa dengan data testing
            5. **Validation**: Cross-validation untuk estimasi performa yang robust
            
            **Tips:**
            - Mulai dengan model sederhana sebagai baseline
            - Gunakan cross-validation untuk validasi yang lebih robust
            - Perhatikan overfitting vs underfitting
            - Pilih metrik evaluasi yang sesuai dengan business objective
            """)
        
        if st.session_state.df_processed is not None:
            df = st.session_state.df_processed
            model_manager = ModelManager()
            
            # Enhanced feature and target selection
            st.subheader("🎯 Konfigurasi Training")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Pilih Target & Features:**")
                all_cols = list(df.columns)
                target = st.selectbox("Kolom Target", all_cols, 
                                    index=all_cols.index(st.session_state.target) if st.session_state.target in all_cols else 0)
                
            with col2:
                st.write("**Problem Type:**")
                problem_type = st.selectbox("Tipe Problem", ["Klasifikasi", "Regresi", "Klastering"])
            
            # Feature selection with advanced options
            available_features = [col for col in all_cols if col != target]
            
            feature_selection_method = st.radio("Metode Pemilihan Fitur:", 
                                               ["Manual Selection", "Select All", "Top Correlated Features"])
            
            if feature_selection_method == "Manual Selection":
                features = st.multiselect("Pilih Fitur", available_features, default=available_features[:10])
            elif feature_selection_method == "Select All":
                features = available_features
                st.info(f"Menggunakan semua {len(features)} fitur yang tersedia")
            else:  # Top Correlated Features
                if problem_type != "Klastering":
                    n_top_features = st.slider("Jumlah fitur terkorelasi tertinggi", 3, min(20, len(available_features)), 10)
                    # Calculate correlation with target
                    correlations = df[available_features].corrwith(df[target]).abs().sort_values(ascending=False)
                    features = correlations.head(n_top_features).index.tolist()
                    st.info(f"Menggunakan {len(features)} fitur dengan korelasi tertinggi: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
                else:
                    features = available_features
                    st.info("Clustering menggunakan semua fitur numerik")
            
            if not features:
                st.warning("⚠️ Pilih minimal satu fitur untuk training.")
            else:
                # Model selection with enhanced interface
                st.subheader("🤖 Pemilihan Model")
                
                # Show available models
                available_models = model_manager.get_model_list(problem_type)
                
                if not available_models:
                    st.error(f"❌ Tidak ada model yang tersedia untuk problem type '{problem_type}'. Silakan pilih problem type yang berbeda.")
                    st.stop()
                
                # Model selection with description
                model_name = st.selectbox("Pilih Model", available_models)
                
                # Training configuration
                training_tabs = st.tabs(["⚙️ Parameter", "🎯 Training", "📊 Evaluasi", "🔄 Cross Validation"])
                
                with training_tabs[0]:
                    st.subheader("Konfigurasi Parameter Model")
                    model, param_grid = model_manager.create_model_with_params(model_name, problem_type)
                    
                    if model is None:
                        st.stop()  # Stop execution if model creation failed
                
                with training_tabs[1]:
                    st.subheader("Training Setup")
                    
                    if problem_type != "Klastering":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            test_size = st.slider("Test Size (%)", 10, 50, 20)
                        with col2:
                            random_state = st.number_input("Random State", 0, 100, 42)
                        with col3:
                            stratify = st.checkbox("Stratified Split", value=True if problem_type == "Klasifikasi" else False)
                    
                        # Data splitting
                        from sklearn.model_selection import train_test_split
                        X = df[features]
                        y = df[target]
                        
                        try:
                            if stratify and problem_type == "Klasifikasi":
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=test_size/100, random_state=random_state, stratify=y
                                )
                            else:
                                X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=test_size/100, random_state=random_state
                                )
                            
                            # Store in session state
                            st.session_state.X_train = X_train
                            st.session_state.X_test = X_test
                            st.session_state.y_train = y_train
                            st.session_state.y_test = y_test
                            
                            # Show split information
                            split_info = pd.DataFrame({
                                "Dataset": ["Training", "Testing", "Total"],
                                "Samples": [len(X_train), len(X_test), len(X)],
                                "Percentage": [f"{len(X_train)/len(X)*100:.1f}%", f"{len(X_test)/len(X)*100:.1f}%", "100%"]
                            })
                            st.dataframe(split_info.set_index("Dataset"))
                            
                        except Exception as e:
                            st.error(f"Error dalam data splitting: {e}")
                            st.stop()
                    
                    # Training button
                    if st.button("🚀 Train Model", type="primary"):
                        try:
                            with st.spinner(f"Training {model_name}..."):
                                if problem_type == "Klastering":
                                    X = df[features]
                                    model.fit(X)
                                    labels = model.labels_
                                    
                                    st.success("✅ Clustering berhasil!")
                                    st.session_state.model = model
                                    st.session_state.clustering_labels = labels
                                    
                                else:
                                    # Train supervised model
                                    model.fit(X_train, y_train)
                                    y_pred = model.predict(X_test)
                                    
                                    # Get prediction probabilities if available
                                    y_proba = None
                                    if hasattr(model, "predict_proba") and problem_type == "Klasifikasi":
                                        y_proba = model.predict_proba(X_test)
                                    
                                    st.session_state.model = model
                                    st.session_state.y_pred = y_pred
                                    st.session_state.y_proba = y_proba
                                    
                                    st.success("✅ Training berhasil!")
                        
                        except Exception as e:
                            st.error(f"❌ Training gagal: {e}")
                
                with training_tabs[2]:
                    st.subheader("Hasil Evaluasi")
                    
                    if problem_type == "Klastering" and "clustering_labels" in st.session_state:
                        labels = st.session_state.clustering_labels
                        X = df[features]
                        
                        # Evaluate clustering
                        try:
                            if len(set(labels)) > 1:
                                from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
                                sil_score = silhouette_score(X, labels)
                                cal_score = calinski_harabasz_score(X, labels)
                                dav_score = davies_bouldin_score(X, labels)
                                
                                eval_df = pd.DataFrame({
                                    "Metrik": ["Silhouette Score", "Calinski-Harabasz Score", "Davies-Bouldin Score"],
                                    "Skor": [sil_score, cal_score, dav_score]
                                })
                                st.dataframe(eval_df.set_index("Metrik"))
                            else:
                                st.info("Hanya satu cluster ditemukan. Metrik evaluasi tidak dapat dihitung.")
                        
                        # Plot clustering results
                            # Plot clustering results
                            try:
                                if len(features) >= 2:
                                    fig = model_manager.plot_clustering_results(X, labels, model_name)
                                    if fig:
                                        st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error plotting clustering results: {e}")

                        except Exception as e:
                            st.error(f"Error evaluating clustering: {e}")
                    
                    elif "y_pred" in st.session_state and st.session_state.y_pred is not None:
                        y_test = st.session_state.y_test
                        y_pred = st.session_state.y_pred
                        y_proba = st.session_state.get("y_proba")
                        
                        if problem_type == "Klasifikasi":
                            results = model_manager.evaluate_classification(y_test, y_pred, y_proba)
                            
                            # Display metrics
                            eval_df = pd.DataFrame({
                                "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score"],
                                "Skor": [results["accuracy"], results["precision"], results["recall"], results["f1_score"]]
                            })
                            st.dataframe(eval_df.set_index("Metrik"))
                            
                            # Plot results - generate class names from the data
                            class_names = [str(cls) for cls in sorted(y_test.unique())]
                            fig = model_manager.plot_classification_results(y_test, y_pred, y_proba, class_names)
                            st.pyplot(fig)
                            
                            # Detailed classification report
                            with st.expander("📋 Detailed Classification Report"):
                                report_df = pd.DataFrame(results["classification_report"]).transpose()
                                st.dataframe(report_df)
                        else:  # Regression
                            results = model_manager.evaluate_regression(y_test, y_pred)
                            
                            # Display metrics
                            eval_df = pd.DataFrame({
                                "Metrik": ["R² Score", "MAE", "MSE", "RMSE", "Explained Variance", "Max Error"],
                                "Skor": [results["r2_score"], results["mae"], results["mse"], 
                                       results["rmse"], results["explained_variance"], results["max_error"]]
                            })
                            st.dataframe(eval_df.set_index("Metrik"))
                            
                            # Plot results
                            fig = model_manager.plot_regression_results(y_test, y_pred)
                            st.pyplot(fig)
                            
                            # Feature importance if available
                            if hasattr(st.session_state.model, 'feature_importances_'):
                                with st.expander("📊 Feature Importance"):
                                    importance_df = pd.DataFrame({
                                        'Feature': features,
                                        'Importance': st.session_state.model.feature_importances_
                                    }).sort_values('Importance', ascending=False)
                                    
                                    fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
                                    sns.barplot(data=importance_df.head(10), x='Importance', y='Feature', ax=ax_imp)
                                    ax_imp.set_title('Top 10 Feature Importance')
                                    st.pyplot(fig_imp)
                                    st.dataframe(importance_df)
                    else:
                        st.info("👆 Silakan lakukan training terlebih dahulu pada tab Training")
                
                with training_tabs[3]:
                    st.subheader("Cross Validation")
                    
                    if problem_type != "Klastering":
                        col1, col2 = st.columns(2)
                        with col1:
                            cv_folds = st.number_input("Number of CV Folds", 3, 10, 5)
                            cv_scoring = st.selectbox("Scoring Metric", 
                                                    ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'] if problem_type == "Klasifikasi" 
                                                    else ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'])
                        
                        with col2:
                            use_grid_search = st.checkbox("Grid Search CV", value=False)
                            if use_grid_search:
                                search_type = st.selectbox("Search Type", ["GridSearchCV", "RandomizedSearchCV"])
                        
                        if st.button("Run Cross Validation"):
                            try:
                                X = df[features]
                                y = df[target]
                                
                                with st.spinner("Running Cross Validation..."):
                                    if use_grid_search:
                                        # Use the base model for grid search
                                        base_model = model_manager.get_base_model(model_name)
                                        
                                        if search_type == "GridSearchCV":
                                            search = GridSearchCV(base_model, param_grid, cv=cv_folds, 
                                                                scoring=cv_scoring, n_jobs=-1)
                                        else:
                                            search = RandomizedSearchCV(base_model, param_grid, cv=cv_folds,
                                                                      scoring=cv_scoring, n_jobs=-1, n_iter=20)
                                        
                                        search.fit(X, y)
                                        
                                        st.success("✅ Grid Search CV completed!")
                                        st.write("**Best Parameters:**")
                                        st.json(search.best_params_)
                                        st.write(f"**Best CV Score:** {search.best_score_:.4f}")
                                        
                                        # Store best model
                                        st.session_state.model = search.best_estimator_
                                        st.session_state.best_params = search.best_params_
                                        
                                    else:
                                        # Regular cross validation
                                        cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring=cv_scoring)
                                        
                                        st.success("✅ Cross Validation completed!")
                                        
                                        # Display CV results
                                        cv_results = pd.DataFrame({
                                            "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))] + ["Mean", "Std"],
                                            "Score": list(cv_scores) + [cv_scores.mean(), cv_scores.std()]
                                        })
                                        st.dataframe(cv_results.set_index("Fold"))
                                        
                                        # Plot CV scores
                                        fig_cv, ax_cv = plt.subplots(figsize=(10, 6))
                                        ax_cv.plot(range(1, len(cv_scores)+1), cv_scores, 'bo-', linewidth=2, markersize=8)
                                        ax_cv.axhline(y=cv_scores.mean(), color='r', linestyle='--', 
                                                    label=f'Mean: {cv_scores.mean():.4f}')
                                        ax_cv.fill_between(range(1, len(cv_scores)+1), 
                                                          cv_scores.mean() - cv_scores.std(),
                                                          cv_scores.mean() + cv_scores.std(),
                                                          alpha=0.2, color='red')
                                        ax_cv.set_xlabel('Fold')
                                        ax_cv.set_ylabel(f'{cv_scoring} Score')
                                        ax_cv.set_title('Cross Validation Scores')
                                        ax_cv.legend()
                                        ax_cv.grid(True, alpha=0.3)
                                        st.pyplot(fig_cv)
                                
                            except Exception as e:
                                st.error(f"❌ Cross Validation failed: {e}")
                    
                    else:
                        st.info("Cross Validation tidak tersedia untuk clustering")

        else:
            st.warning("Silakan lakukan preprocessing terlebih dahulu.")

    # Step 6: Export Dataset/Model
    elif step == steps[5]:
        st.markdown('<div class="step-header"><h2>💾 Step 6: Export Dataset/Model</h2></div>', unsafe_allow_html=True)
        
        if st.session_state.df is not None:
            st.subheader("Export Dataset Original")
            csv = st.session_state.df.to_csv(index=False).encode()
            st.download_button("Download CSV (Original)", csv, "dataset_original.csv", "text/csv")
        
        if st.session_state.df_processed is not None:
            st.subheader("Export Dataset Setelah Preprocessing")
            csv2 = st.session_state.df_processed.to_csv(index=False).encode()
            st.download_button("Download CSV (Processed)", csv2, "dataset_processed.csv", "text/csv")
        
        if st.session_state.model is not None:
            st.subheader("Export Model")
            try:
                import joblib
                import io
                model_bytes = io.BytesIO()
                joblib.dump(st.session_state.model, model_bytes)
                model_bytes.seek(0)
                st.download_button("Download Model", model_bytes.getvalue(), "model.joblib", "application/octet-stream")
            except Exception as e:
                st.error(f"Error exporting model: {e}")


# --- Progress tracker & reset button: Selalu tampil di sidebar ---
def show_progress_and_reset():
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Progress Tracker")
    progress_data = {
        "Load Data": st.session_state.df is not None,
        "EDA": st.session_state.df is not None,  
        "Preprocessing": st.session_state.df_processed is not None,
        "Model Training": st.session_state.model is not None
    }
    for task, completed in progress_data.items():
        if completed:
            st.sidebar.markdown(f"✅ {task}")
        else:
            st.sidebar.markdown(f"⏳ {task}")
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Reset Semua Proses"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# Panggil di awal script utama (setelah import dan session state init)
show_progress_and_reset()