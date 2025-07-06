import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    """Module for data preprocessing operations"""
    
    @staticmethod
    def get_imputation_recommendations(df):
        """Get imputation strategy recommendations"""
        recommendations = []
        missing_cols = df.columns[df.isnull().any()].tolist()
        
        for col in missing_cols:
            missing_pct = df[col].isnull().sum() / len(df)
            
            if df[col].dtype in ['object', 'category']:
                rec = "most_frequent (mode)"
            elif missing_pct > 0.3:
                rec = "constant atau hapus kolom"
            elif pd.api.types.is_numeric_dtype(df[col]) and abs(df[col].skew()) > 1:
                rec = "median (data skewed)"
            else:
                rec = "mean (data normal)"
            
            recommendations.append({
                'Kolom': col,
                'Missing %': missing_pct * 100,
                'Rekomendasi': rec
            })
        
        return pd.DataFrame(recommendations)
    
    @staticmethod
    def impute_missing_values(df, columns, strategy='mean', fill_value=None):
        """Impute missing values"""
        df_copy = df.copy()
        
        if strategy == 'constant' and fill_value is not None:
            imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
        else:
            imputer = SimpleImputer(strategy=strategy)
        
        df_copy[columns] = imputer.fit_transform(df_copy[columns])
        return df_copy, imputer
    
    @staticmethod
    def encode_categorical(df, columns, encoding_type='label'):
        """Encode categorical variables"""
        df_copy = df.copy()
        encoders = {}
        
        if encoding_type == 'label':
            for col in columns:
                le = LabelEncoder()
                df_copy[col] = le.fit_transform(df_copy[col].astype(str))
                encoders[col] = le
        
        elif encoding_type == 'onehot':
            df_copy = pd.get_dummies(df_copy, columns=columns, prefix=columns)
        
        elif encoding_type == 'ordinal':
            oe = OrdinalEncoder()
            df_copy[columns] = oe.fit_transform(df_copy[columns].astype(str))
            encoders['ordinal'] = oe
        
        return df_copy, encoders
    
    @staticmethod
    def scale_features(df, columns, scaler_type='standard'):
        """Scale numerical features"""
        df_copy = df.copy()
        
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
        
        df_copy[columns] = scaler.fit_transform(df_copy[columns])
        return df_copy, scaler
    
    @staticmethod
    def get_preprocessing_summary(original_df, processed_df):
        """Get preprocessing summary"""
        return {
            'original_shape': original_df.shape,
            'processed_shape': processed_df.shape,
            'original_missing': original_df.isnull().sum().sum(),
            'processed_missing': processed_df.isnull().sum().sum(),
            'original_categorical': len(original_df.select_dtypes(include=['object', 'category']).columns),
            'processed_categorical': len(processed_df.select_dtypes(include=['object', 'category']).columns)
        }
