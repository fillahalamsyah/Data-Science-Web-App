import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime


def show_home():
    """Display the home page content"""
    
    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='color: #1f77b4; font-size: 3rem; margin-bottom: 0;'>
            🔬 Data Science Web App
        </h1>
        <p style='font-size: 1.2rem; color: #666; margin-top: 0.5rem;'>
            Platform pembelajaran dan implementasi Data Science yang komprehensif
        </p>
        <p style='font-size: 1rem; color: #888;'>
            Dari eksplorasi data hingga deployment model - semua dalam satu aplikasi!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Dataset Built-in", "5", help="Iris, Wine, Boston Housing, Diabetes, Breast Cancer")
    with col2:
        st.metric("🤖 ML Algoritma", "12+", help="Classification & Regression algorithms")
    with col3:
        st.metric("📈 Visualisasi", "20+", help="Interactive charts and plots")
    with col4:
        st.metric("🛠️ Tools", "6", help="Complete ML pipeline tools")
    
    st.markdown("---")
    
    # Features Overview
    st.header("🎯 Fitur Utama")
    
    features_tab1, features_tab2, features_tab3 = st.tabs(["📚 Learning", "🔬 Analysis", "🤖 Modeling"])
    
    with features_tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("""
            ### 📖 Artikel Data Science
            - **12+ Topik Lengkap**
            - Pengantar hingga Advanced
            - Teori & Implementasi
            - Best Practices
            - Real-world Examples
            """)
            
            if st.button("🚀 Mulai Belajar", key="learn_btn"):
                st.info("👈 Gunakan sidebar untuk navigasi ke halaman Artikel!")
        
        with col2:
            # Create a sample learning path visualization
            learning_topics = [
                "Data Science Intro", "Data Collection", "EDA", "Preprocessing",
                "Feature Engineering", "ML Fundamentals", "Model Training", 
                "Evaluation", "Hyperparameter Tuning", "Deep Learning",
                "Visualization", "Ethics in DS"
            ]
            
            progress_data = {
                'Topic': learning_topics,
                'Difficulty': [1, 2, 2, 3, 4, 3, 4, 3, 5, 5, 2, 3],
                'Duration (hours)': [2, 3, 4, 3, 5, 4, 6, 3, 4, 8, 3, 2]
            }
            
            fig = px.scatter(progress_data, x='Difficulty', y='Duration (hours)', 
                           hover_name='Topic', size='Duration (hours)',
                           color='Difficulty', color_continuous_scale='viridis',
                           title="Learning Path Overview")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with features_tab2:
        col1, col2 = st.columns([2, 1])
        with col1:
            # Create sample data visualization
            np.random.seed(42)
            sample_data = pd.DataFrame({
                'Feature_A': np.random.normal(50, 15, 100),
                'Feature_B': np.random.normal(30, 10, 100),
                'Feature_C': np.random.normal(70, 20, 100),
                'Target': np.random.choice(['Class_1', 'Class_2', 'Class_3'], 100)
            })
            
            fig = px.scatter_3d(sample_data, x='Feature_A', y='Feature_B', z='Feature_C',
                              color='Target', title="Sample 3D Data Exploration")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            ### 🔍 Eksplorasi Data
            - **5 Dataset Built-in**
            - Upload CSV/Excel
            - Analisis Univariat
            - Analisis Bivariat  
            - Analisis Multivariat
            - Data Quality Check
            - Missing Values Analysis
            - Outlier Detection
            """)
            
            if st.button("🔬 Mulai Eksplorasi", key="explore_btn"):
                st.info("👈 Pilih 'Load/Generate/Upload Dataset' di sidebar!")
    
    with features_tab3:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("""
            ### 🎯 Model Training
            - **12+ Algoritma ML**
            - Classification & Regression
            - Hyperparameter Tuning
            - Cross Validation
            - Model Comparison
            - Performance Metrics
            - Feature Importance
            - Model Export
            """)
            
            if st.button("🤖 Train Model", key="model_btn"):
                st.info("👈 Mulai dari Load Dataset di sidebar!")
        
        with col2:
            # Model performance comparison chart
            models = ['Logistic Regression', 'Random Forest', 'SVM', 'XGBoost', 'Neural Network']
            accuracy = [85.2, 89.7, 87.3, 91.2, 88.9]
            training_time = [0.5, 2.3, 1.8, 3.2, 5.1]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=training_time, y=accuracy, mode='markers+text',
                text=models, textposition="top center",
                marker=dict(size=15, color=accuracy, colorscale='viridis', 
                          colorbar=dict(title="Accuracy %")),
                name="Models"
            ))
            fig.update_layout(
                title="Model Performance vs Training Time",
                xaxis_title="Training Time (seconds)",
                yaxis_title="Accuracy (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Quick Start Guide
    st.header("🚀 Panduan Cepat")
    
    quick_start_col1, quick_start_col2 = st.columns([2, 1])
    
    with quick_start_col1:
        st.markdown("""
        ### 📋 Langkah-langkah Memulai:
        
        **1. 📊 Load Data**
        - Pilih dataset built-in (Iris, Wine, dll.)
        - Generate synthetic data
        - Upload file CSV/Excel Anda
        
        **2. 🔍 Eksplorasi**
        - Analisis distribusi data
        - Cari pola dan korelasi
        - Identifikasi outliers
        
        **3. 🛠️ Preprocessing**
        - Handle missing values
        - Encode categorical features
        - Scale numerical features
        
        **4. 📐 Dimensionality Reduction** *(Opsional)*
        - PCA, t-SNE, LDA
        - Feature selection
        - Visualisasi high-dimensional data
        
        **5. 🤖 Model Training**
        - Pilih algoritma yang sesuai
        - Tune hyperparameters
        - Evaluasi performa
        
        **6. 💾 Export**
        - Download processed dataset
        - Save trained model
        - Export visualizations
        """)
    
    with quick_start_col2:
        # Progress visualization
        steps = ["Load Data", "EDA", "Preprocessing", "Dim. Reduction", "Model Training", "Export"]
        progress = [100, 85, 70, 40, 60, 30]  # Sample progress
        
        fig = go.Figure(go.Bar(
            x=progress,
            y=steps,
            orientation='h',
            marker_color=['#ff7f0e' if p < 100 else '#2ca02c' for p in progress]
        ))
        fig.update_layout(
            title="Typical Workflow Progress",
            xaxis_title="Completion %",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick action buttons
        st.markdown("### ⚡ Quick Actions")
        if st.button("📊 Load Sample Data", key="quick_load"):
            st.info("👈 Gunakan sidebar → '1. Load/Generate/Upload Dataset'")
        
        if st.button("📚 Read Articles", key="quick_articles"):
            st.info("👈 Gunakan sidebar → 'Artikel Data Science'")
        
        if st.button("🔗 View on GitHub", key="github"):
            st.markdown("[🔗 GitHub Repository](https://github.com/)", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Datasets Overview
    st.header("📊 Dataset Built-in")
    
    datasets_info = {
        "Iris": {
            "samples": 150,
            "features": 4,
            "classes": 3,
            "type": "Classification",
            "description": "Classic flower classification dataset",
            "use_case": "Beginner-friendly classification"
        },
        "Wine": {
            "samples": 178,
            "features": 13,
            "classes": 3,
            "type": "Classification", 
            "description": "Wine quality classification",
            "use_case": "Multi-class classification"
        },
        "Breast Cancer": {
            "samples": 569,
            "features": 30,
            "classes": 2,
            "type": "Classification",
            "description": "Medical diagnosis dataset",
            "use_case": "Binary classification, healthcare"
        },
        "Boston Housing": {
            "samples": 506,
            "features": 13,
            "classes": np.nan,
            "type": "Regression",
            "description": "House price prediction",
            "use_case": "Real estate price modeling"
        },
        "Diabetes": {
            "samples": 442,
            "features": 10,
            "classes": np.nan,
            "type": "Regression",
            "description": "Disease progression prediction",
            "use_case": "Healthcare regression analysis"
        }
    }
    
    # Create dataset overview table
    datasets_df = pd.DataFrame(datasets_info).T
    datasets_df = datasets_df.reset_index().rename(columns={'index': 'Dataset'})
    
    # Convert columns to numeric
    datasets_df['samples'] = pd.to_numeric(datasets_df['samples'])
    datasets_df['features'] = pd.to_numeric(datasets_df['features'])
    datasets_df['classes'] = pd.to_numeric(datasets_df['classes'])

    st.dataframe(
        datasets_df.style.format({
            'samples': '{:,}',
            'features': '{:,}'
        }),
        use_container_width=True
    )
    
    # Dataset comparison chart
    fig = px.scatter(datasets_df, x='features', y='samples', 
                    color='type', size='features',
                    hover_name='Dataset',
                    title="Dataset Comparison: Features vs Samples")
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Technologies Used
    st.header("🛠️ Teknologi yang Digunakan")
    
    tech_col1, tech_col2, tech_col3 = st.columns(3)
    
    with tech_col1:
        st.markdown("""
        **🎨 Frontend & UI:**
        - Streamlit
        - Plotly
        - Matplotlib/Seaborn
        """)
    
    with tech_col2:
        st.markdown("""
        **📊 Data Processing:**
        - Pandas
        - NumPy
        - Scikit-learn
        """)
    
    with tech_col3:
        st.markdown("""
        **🤖 Machine Learning:**
        - Classification algorithms
        - Regression algorithms
        - Clustering methods
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>
            Made with ❤️ using Streamlit 
        </p>
        <p style='font-size: 0.9rem;'>
            © 2024 Data Science Web App. Happy Learning! 🚀
        </p>
    </div>
    """, unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    show_home()
