import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

class DataAnalyzer:
    """Module for data analysis and visualization"""
    
    @staticmethod
    def get_data_summary(df):
        """Get comprehensive data summary"""
        return {
            "shape": df.shape,
            "numeric_cols": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_cols": len(df.select_dtypes(include=['object', 'category']).columns),
            "missing_values": df.isnull().sum().sum(),
            "duplicates": df.duplicated().sum(),
            "memory_usage": df.memory_usage(deep=True).sum() / 1024  # KB
        }
    
    @staticmethod
    def analyze_missing_values(df):
        """Analyze missing values in dataset"""
        missing_stats = pd.DataFrame({
            'Kolom': df.columns,
            'Missing Count': df.isnull().sum().values,
            'Missing %': (df.isnull().sum() / len(df) * 100).values,
            'Data Type': [str(df[col].dtype) for col in df.columns]
        })
        return missing_stats[missing_stats['Missing Count'] > 0]
    
    @staticmethod
    def plot_distribution(df, column):
        """Plot distribution for a column"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        if pd.api.types.is_numeric_dtype(df[column]):
            # Histogram with KDE
            sns.histplot(df[column], kde=True, stat="density", ax=axes[0], alpha=0.7)
            mean_val = df[column].mean()
            median_val = df[column].median()
            axes[0].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
            axes[0].axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.2f}')
            axes[0].legend()
            axes[0].set_title(f"Distribusi {column}")
            
            # Box plot
            sns.boxplot(x=df[column], ax=axes[1])
            axes[1].set_title(f"Box Plot - {column}")
        else:
            # Bar chart
            value_counts = df[column].value_counts()
            value_counts.plot(kind='bar', ax=axes[0], color='skyblue')
            axes[0].set_title(f"Distribusi {column}")
            axes[0].set_xlabel("Kategori")
            axes[0].set_ylabel("Frekuensi")
            
            # Pie chart
            value_counts.plot(kind='pie', autopct='%1.1f%%', ax=axes[1])
            axes[1].set_ylabel('')
            axes[1].set_title(f"Proporsi {column}")
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def get_outlier_analysis(df, column):
        """Analyze outliers in numeric column"""
        if not pd.api.types.is_numeric_dtype(df[column]):
            return None
        
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        
        return {
            "Q1": Q1,
            "Q3": Q3, 
            "IQR": IQR,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": len(outliers),
            "outlier_percentage": len(outliers) / len(df) * 100
        }
    
    @staticmethod
    def plot_correlation_matrix(df):
        """Plot correlation matrix for numeric columns"""
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) < 2:
            return None
        
        corr_matrix = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, ax=ax, 
                   cmap="RdBu_r", center=0, fmt=".2f", 
                   square=True, linewidths=0.5)
        ax.set_title("Matrix Korelasi Fitur Numerik")
        return fig
    
    @staticmethod
    def find_strong_correlations(df, threshold=0.7):
        """Find strongly correlated feature pairs"""
        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) < 2:
            return []
        
        corr_matrix = df[num_cols].corr()
        strong_corr = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > threshold:
                    strength = 'Very Strong' if abs(corr_val) > 0.9 else 'Strong'
                    strong_corr.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_val,
                        'Strength': strength
                    })
        
        return strong_corr
