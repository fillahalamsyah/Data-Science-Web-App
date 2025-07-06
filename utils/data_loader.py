import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets

class DataLoader:
    """Module for loading and generating datasets"""
    
    @staticmethod
    def get_dataset_info():
        """Get information about available datasets"""
        return {
            "iris": {
                "deskripsi": "Dataset klasifikasi bunga iris dengan 3 spesies",
                "fitur": 4, "sampel": 150, "kelas": 3,
                "use_case": "Klasifikasi multi-kelas, pembelajaran supervised"
            },
            "wine": {
                "deskripsi": "Dataset klasifikasi wine berdasarkan analisis kimia", 
                "fitur": 13, "sampel": 178, "kelas": 3,
                "use_case": "Klasifikasi, analisis komponen kimia"
            },
            "breast_cancer": {
                "deskripsi": "Dataset diagnosa kanker payudara (malignant/benign)",
                "fitur": 30, "sampel": 569, "kelas": 2,
                "use_case": "Klasifikasi biner, aplikasi medis"
            },
            "digits": {
                "deskripsi": "Dataset pengenalan digit tulisan tangan (0-9)",
                "fitur": 64, "sampel": 1797, "kelas": 10,
                "use_case": "Klasifikasi multi-kelas, computer vision"
            }
        }
    
    @staticmethod
    def load_sklearn_dataset(dataset_name):
        """Load sklearn dataset"""
        if dataset_name == "iris":
            return datasets.load_iris(as_frame=True)
        elif dataset_name == "wine":
            return datasets.load_wine(as_frame=True)
        elif dataset_name == "breast_cancer":
            return datasets.load_breast_cancer(as_frame=True)
        elif dataset_name == "digits":
            return datasets.load_digits(as_frame=True)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    @staticmethod
    def generate_dataset(gen_type, n_samples, n_features, **kwargs):
        """Generate synthetic dataset"""
        if gen_type == "Classification":
            X, y = datasets.make_classification(
                n_samples=n_samples, 
                n_features=n_features, 
                n_classes=kwargs.get('n_classes', 3),
                n_redundant=kwargs.get('n_redundant', 0),
                n_informative=kwargs.get('n_informative', min(n_features, 3)),
                random_state=42
            )
        elif gen_type == "Regression":
            X, y = datasets.make_regression(
                n_samples=n_samples, 
                n_features=n_features, 
                random_state=42
            )
        elif gen_type == "Blobs":
            X, y = datasets.make_blobs(
                n_samples=n_samples, 
                n_features=n_features,
                centers=kwargs.get('n_centers', 3),
                cluster_std=kwargs.get('cluster_std', 1.0),
                random_state=42
            )
        else:
            raise ValueError(f"Unknown generation type: {gen_type}")
        
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        df["target"] = y
        return df
    
    @staticmethod
    def load_uploaded_file(uploaded_file):
        """Load uploaded file"""
        ext = uploaded_file.name.split(".")[-1]
        if ext == "csv":
            return pd.read_csv(uploaded_file)
        elif ext == "xlsx":
            return pd.read_excel(uploaded_file)
        elif ext == "parquet":
            return pd.read_parquet(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
