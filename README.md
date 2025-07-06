# 🔬 Data Science Web App

Aplikasi web interaktif berbasis Streamlit untuk pembelajaran dan implementasi Data Science, dilengkapi dengan artikel pembelajaran dan fitur hands-on machine learning.

## 🎯 Fitur Utama

### 📚 **Artikel Data Science**
Koleksi artikel lengkap tentang konsep dan metodologi Data Science:
- Pengantar Data Science
- Data Collection & Cleaning
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Engineering
- Machine Learning Fundamentals
- Supervised & Unsupervised Learning
- Model Evaluation & Selection
- Hyperparameter Tuning
- Deep Learning Basics
- Data Visualization
- Ethics in Data Science

### 📊 **Eksplorasi Data Interaktif**
Tools untuk analisis mendalam dataset:
- **Dataset Built-in**: Iris, Wine, Boston Housing, Diabetes, Breast Cancer
- **Upload Custom Dataset**: Support CSV dan Excel
- **Univariate Analysis**: Distribusi dan statistik per variabel
- **Bivariate Analysis**: Hubungan antar variabel
- **Multivariate Analysis**: Correlation heatmap, pair plot, parallel coordinates
- **Data Quality**: Missing values, duplicates, outliers

### 🤖 **Pelatihan Model Machine Learning**
Implementasi lengkap ML pipeline:
- **Data Preparation**: Feature selection, encoding, scaling
- **Model Selection**: 10+ algoritma classification dan regression
- **Training & Evaluation**: Comprehensive metrics dan visualizations
- **Hyperparameter Tuning**: Grid Search dengan Cross-Validation
- **Model Comparison**: Side-by-side algorithm comparison

## 🛠️ Teknologi yang Digunakan

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Machine Learning**: Scikit-learn
- **File Support**: CSV, Excel (openpyxl, xlrd)

## 📦 Instalasi

### Prerequisites
- Python 3.7 atau lebih tinggi
- pip (Python package manager)

### Langkah Instalasi

1. **Clone repository**
   ```bash
   git clone <repository-url>
   cd Data-Science-Web-App
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi**
   ```bash
   streamlit run app.py
   ```

4. **Buka browser**
   Aplikasi akan otomatis membuka di `http://localhost:8501`

## 🚀 Cara Penggunaan

### 1. Beranda
- Overview fitur aplikasi
- Quick actions untuk navigasi cepat
- Statistik aplikasi

### 2. Artikel Data Science
- Pilih artikel dari sidebar
- Baca teori dan konsep Data Science
- Panduan step-by-step untuk setiap topik

### 3. Eksplorasi Data
- **Built-in Dataset**: Pilih dari 5 dataset populer
- **Upload File**: Upload dataset CSV/Excel Anda
- **Analisis**: Gunakan tabs untuk different types of analysis
- **Export**: Download hasil analisis

### 4. Pelatihan Model
- **Data Prep**: Configure preprocessing steps
- **Model Selection**: Choose algorithm dan set parameters
- **Training**: Train model dan evaluate performance
- **Tuning**: Optimize hyperparameters
- **Comparison**: Compare multiple algorithms

## 📊 Dataset Built-in

| Dataset | Type | Samples | Features | Target |
|---------|------|---------|----------|--------|
| Iris | Classification | 150 | 4 | Species (3 classes) |
| Wine | Classification | 178 | 13 | Wine class (3 classes) |
| Boston Housing | Regression | 506 | 13 | House price |
| Diabetes | Regression | 442 | 10 | Disease progression |
| Breast Cancer | Classification | 569 | 30 | Diagnosis (2 classes) |

## 🤖 Algoritma Machine Learning

### Classification
- Logistic Regression
- Random Forest Classifier
- Support Vector Machine
- Decision Tree Classifier
- K-Nearest Neighbors
- Naive Bayes

### Regression
- Linear Regression
- Random Forest Regressor
- Support Vector Regression
- Decision Tree Regressor
- K-Nearest Neighbors Regressor
- Ridge Regression
- Lasso Regression

## 📈 Evaluation Metrics

### Classification
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Confusion Matrix
- Classification Report

### Regression
- R², RMSE, MAE, MSE
- Residual Analysis
- Prediction vs Actual plots

## 🎨 Visualizations

### Data Exploration
- Histograms, Box plots, Bar charts
- Scatter plots, Correlation heatmaps
- Pair plots, Parallel coordinates
- Missing value patterns

### Model Evaluation
- Confusion Matrix heatmap
- ROC Curves
- Residual plots
- Feature importance charts
- Model comparison plots

## 📁 Struktur Project

```
Data-Science-Web-App/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
└── pages/                # Application pages
    ├── __init__.py
    ├── articles.py       # Data Science articles
    ├── data_exploration.py  # Data exploration tools
    └── model_training.py    # ML model training
```

## 🤝 Kontribusi

Kontribusi sangat welcome! Silakan:

1. Fork repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📋 Roadmap

### v2.0 (Planned)
- [ ] Deep Learning models (TensorFlow/PyTorch integration)
- [ ] Time Series Analysis
- [ ] Natural Language Processing tools
- [ ] Advanced visualization (3D plots, interactive dashboards)
- [ ] Model deployment features
- [ ] Database connectivity
- [ ] User authentication

### v2.1 (Future)
- [ ] AutoML capabilities
- [ ] Model interpretability tools (SHAP, LIME)
- [ ] A/B testing framework
- [ ] Real-time model monitoring
- [ ] API endpoints for model serving

## 🐛 Bug Reports & Feature Requests

Gunakan GitHub Issues untuk:
- Report bugs
- Request new features
- Suggest improvements
- Ask questions

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Scikit-learn team untuk amazing ML library
- Streamlit team untuk incredible web framework
- Plotly team untuk beautiful visualizations
- Open source community untuk inspiration

## 📞 Contact

- **Developer**: [Your Name]
- **Email**: [your.email@example.com]
- **LinkedIn**: [your-linkedin-profile]
- **GitHub**: [your-github-profile]

---

**Happy Data Science Learning! 🚀📊🤖**
