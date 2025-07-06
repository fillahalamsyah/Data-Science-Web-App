import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch, SpectralClustering
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, precision_score, 
    recall_score, f1_score, roc_auc_score, roc_curve, precision_recall_curve,
    mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, 
    max_error, silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import warnings
warnings.filterwarnings('ignore')

class ModelManager:
    """Enhanced module for comprehensive model creation, training and evaluation"""
    
    @staticmethod
    def get_model_info():
        """Get detailed information about available models"""
        return {
            # Classification Models
            "Logistic Regression": {
                "type": "Linear Classification",
                "description": "Linear model for binary and multiclass classification using logistic function",
                "pros": "Fast, interpretable, no hyperparameter tuning needed, probabilistic output",
                "cons": "Assumes linear relationship, sensitive to outliers",
                "best_for": "Linear separable data, baseline model, interpretability needed",
                "complexity": "Low",
                "category": "Classification"
            },
            "Random Forest": {
                "type": "Ensemble Classification",
                "description": "Ensemble of decision trees with voting mechanism",
                "pros": "Handles overfitting well, feature importance, works with missing values",
                "cons": "Can overfit with very noisy data, less interpretable",
                "best_for": "General purpose, mixed data types, feature selection",
                "complexity": "Medium",
                "category": "Classification"
            },
            "SVM": {
                "type": "Support Vector Classification",
                "description": "Finds optimal hyperplane to separate classes using support vectors",
                "pros": "Effective in high dimensions, memory efficient, versatile kernels",
                "cons": "Slow on large datasets, sensitive to feature scaling",
                "best_for": "High-dimensional data, text classification, small to medium datasets",
                "complexity": "High",
                "category": "Classification"
            },
            "KNN": {
                "type": "Instance-based Classification",
                "description": "Classifies based on majority vote of k nearest neighbors",
                "pros": "Simple, no assumptions about data, works well locally",
                "cons": "Computationally expensive, sensitive to irrelevant features",
                "best_for": "Small datasets, irregular decision boundaries",
                "complexity": "Low",
                "category": "Classification"
            },
            "Decision Tree": {
                "type": "Tree-based Classification",
                "description": "Makes decisions using tree-like model of decisions",
                "pros": "Highly interpretable, handles mixed data types, no scaling needed",
                "cons": "Prone to overfitting, unstable",
                "best_for": "Interpretability, mixed data types, rule extraction",
                "complexity": "Medium",
                "category": "Classification"
            },
            "Gradient Boosting": {
                "type": "Ensemble Classification",
                "description": "Sequential ensemble that corrects previous model errors",
                "pros": "High accuracy, handles mixed data types, feature importance",
                "cons": "Prone to overfitting, requires tuning, slower training",
                "best_for": "High accuracy needed, competitions, structured data",
                "complexity": "High",
                "category": "Classification"
            },
            "AdaBoost": {
                "type": "Ensemble Classification",
                "description": "Adaptive boosting that focuses on misclassified examples",
                "pros": "Good generalization, reduces bias and variance",
                "cons": "Sensitive to noise and outliers, can overfit",
                "best_for": "Binary classification, weak learners combination",
                "complexity": "Medium",
                "category": "Classification"
            },
            "MLP Classifier": {
                "type": "Neural Network Classification",
                "description": "Multi-layer perceptron neural network for classification",
                "pros": "Can learn complex patterns, universal approximator",
                "cons": "Requires large data, many hyperparameters, black box",
                "best_for": "Large datasets, complex patterns, non-linear relationships",
                "complexity": "High",
                "category": "Classification"
            },
            "Naive Bayes": {
                "type": "Probabilistic Classification",
                "description": "Probabilistic classifier based on Bayes theorem with independence assumption",
                "pros": "Fast, works well with small data, handles multiple classes well",
                "cons": "Strong independence assumption, categorical inputs need smoothing",
                "best_for": "Text classification, small datasets, baseline model",
                "complexity": "Low",
                "category": "Classification"
            },
            "Extra Trees": {
                "type": "Ensemble Classification",
                "description": "Extremely randomized trees with random thresholds",
                "pros": "Faster than Random Forest, reduces overfitting further",
                "cons": "May have slightly lower accuracy than Random Forest",
                "best_for": "Large datasets, when Random Forest overfits",
                "complexity": "Medium",
                "category": "Classification"
            },
            
            # Regression Models
            "Linear Regression": {
                "type": "Linear Regression",
                "description": "Linear model using least squares to fit relationship between features and target",
                "pros": "Simple, fast, interpretable, no hyperparameters",
                "cons": "Assumes linear relationship, sensitive to outliers",
                "best_for": "Linear relationships, baseline model, interpretability",
                "complexity": "Low",
                "category": "Regression"
            },
            "Ridge Regression": {
                "type": "Regularized Linear Regression",
                "description": "Linear regression with L2 regularization to prevent overfitting",
                "pros": "Handles multicollinearity, reduces overfitting",
                "cons": "Doesn't perform feature selection, assumes linear relationship",
                "best_for": "Multicollinearity present, many features",
                "complexity": "Low",
                "category": "Regression"
            },
            "Lasso Regression": {
                "type": "Regularized Linear Regression",
                "description": "Linear regression with L1 regularization for feature selection",
                "pros": "Automatic feature selection, handles multicollinearity",
                "cons": "Can be unstable with high correlation, assumes linear relationship",
                "best_for": "Feature selection needed, sparse solutions",
                "complexity": "Low",
                "category": "Regression"
            },
            "ElasticNet": {
                "type": "Regularized Linear Regression",
                "description": "Combines L1 and L2 regularization (Ridge + Lasso)",
                "pros": "Balances feature selection and grouping, stable",
                "cons": "Two hyperparameters to tune, assumes linear relationship",
                "best_for": "Correlated features, feature selection with stability",
                "complexity": "Medium",
                "category": "Regression"
            },
            
            # Clustering Models
            "KMeans": {
                "type": "Centroid-based Clustering",
                "description": "Partitions data into k clusters by minimizing within-cluster sum of squares",
                "pros": "Fast, simple, works well with spherical clusters",
                "cons": "Need to specify k, assumes spherical clusters, sensitive to outliers",
                "best_for": "Spherical clusters, known number of clusters",
                "complexity": "Low",
                "category": "Clustering"
            },
            "DBSCAN": {
                "type": "Density-based Clustering",
                "description": "Groups together points in high-density areas and marks outliers",
                "pros": "Finds arbitrary shaped clusters, identifies outliers, no need to specify clusters",
                "cons": "Sensitive to hyperparameters, struggles with varying densities",
                "best_for": "Arbitrary shaped clusters, outlier detection, unknown cluster count",
                "complexity": "Medium",
                "category": "Clustering"
            },
            "Spectral Clustering": {
                "type": "Graph-based Clustering",
                "description": "Uses eigenvalues of similarity matrix for dimensionality reduction before clustering",
                "pros": "Handles non-convex clusters, works with similarity matrix",
                "cons": "Computationally expensive, need to specify number of clusters",
                "best_for": "Non-convex clusters, graph-structured data",
                "complexity": "High",
                "category": "Clustering"
            }
        }
    
    @staticmethod
    def get_model_list(problem_type):
        """Get available models for problem type"""
        # Map Indonesian problem types to English categories
        problem_type_mapping = {
            "Klasifikasi": "Classification",
            "Regresi": "Regression", 
            "Klastering": "Clustering",
            "Classification": "Classification",
            "Regression": "Regression",
            "Clustering": "Clustering"
        }
        
        english_category = problem_type_mapping.get(problem_type, problem_type)
        model_info = ModelManager.get_model_info()
        return [name for name, info in model_info.items() if info["category"] == english_category]
    
    @staticmethod
    def create_model_with_params(model_name, problem_type):
        """Create model with parameter inputs"""
        params = {}
        
        if model_name == "Logistic Regression":
            col1, col2 = st.columns(2)
            with col1:
                C = st.number_input("C (Regularization strength)", 0.01, 10.0, 1.0, 0.01,
                    help="Smaller values specify stronger regularization")
                max_iter = st.number_input("Max Iterations", 100, 5000, 1000)
            with col2:
                solver = st.selectbox("Solver", ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga'],
                    help="Algorithm to use in optimization")
                penalty = st.selectbox("Penalty", ['l2', 'l1', 'elasticnet', 'none'] if solver in ['saga'] else ['l2', 'none'])
            
            params = {"C": [C], "max_iter": [max_iter], "solver": [solver]}
            model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, penalty=penalty, random_state=42)
        
        elif model_name in ["Random Forest", "Random Forest Regressor", "Extra Trees", "Extra Trees Regressor"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                n_estimators = st.number_input("Number of Trees", 10, 1000, 100)
                max_depth = st.number_input("Max Depth (None for unlimited)", 1, 100, 10)
                max_depth = None if st.checkbox("Unlimited depth") else max_depth
            with col2:
                min_samples_split = st.number_input("Min Samples Split", 2, 20, 2)
                min_samples_leaf = st.number_input("Min Samples Leaf", 1, 20, 1)
            with col3:
                max_features = st.selectbox("Max Features", ['sqrt', 'log2', 'none'], index=0)
                bootstrap = st.checkbox("Bootstrap", value=True)
            
            params = {"n_estimators": [n_estimators], "max_depth": [max_depth], 
                     "min_samples_split": [min_samples_split]}
            
            if problem_type == "Klasifikasi":
                if "Extra Trees" in model_name:
                    model = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, 
                                               min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                               max_features=max_features, bootstrap=bootstrap, random_state=42)
                else:
                    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                 max_features=max_features, bootstrap=bootstrap, random_state=42)
            else:
                if "Extra Trees" in model_name:
                    model = ExtraTreesRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                              min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                              max_features=max_features, bootstrap=bootstrap, random_state=42)
                else:
                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                                min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                                max_features=max_features, bootstrap=bootstrap, random_state=42)
        
        elif model_name in ["SVM", "SVM Regressor"]:
            col1, col2, col3 = st.columns(3)
            with col1:
                C = st.number_input("C (Regularization)", 0.01, 10.0, 1.0, 0.01)
                kernel = st.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
            with col2:
                if kernel in ["rbf", "poly", "sigmoid"]:
                    gamma = st.selectbox("Gamma", ['scale', 'auto', 0.001, 0.01, 0.1, 1])
                else:
                    gamma = 'scale'
                if kernel == "poly":
                    degree = st.number_input("Polynomial Degree", 2, 5, 3)
                else:
                    degree = 3
            with col3:
                if model_name == "SVM Regressor":
                    epsilon = st.number_input("Epsilon", 0.01, 1.0, 0.1, 0.01)
                else:
                    epsilon = 0.1
            
            params = {"C": [C], "kernel": [kernel], "gamma": [gamma]}
            if problem_type == "Klasifikasi":
                model = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree, probability=True, random_state=42)
            else:
                model = SVR(C=C, kernel=kernel, gamma=gamma, degree=degree, epsilon=epsilon)
        
        elif model_name in ["KNN", "KNN Regressor"]:
            col1, col2 = st.columns(2)
            with col1:
                n_neighbors = st.number_input("Number of Neighbors", 1, 50, 5)
                weights = st.selectbox("Weights", ['uniform', 'distance'])
            with col2:
                algorithm = st.selectbox("Algorithm", ['auto', 'ball_tree', 'kd_tree', 'brute'])
                metric = st.selectbox("Distance Metric", ['euclidean', 'manhattan', 'minkowski'])
            
            params = {"n_neighbors": [n_neighbors], "weights": [weights]}
            if problem_type == "Klasifikasi":
                model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, 
                                           algorithm=algorithm, metric=metric)
            else:
                model = KNeighborsRegressor(n_neighbors=n_neighbors, weights=weights,
                                          algorithm=algorithm, metric=metric)
        
        elif model_name in ["Decision Tree", "Decision Tree Regressor"]:
            col1, col2 = st.columns(2)
            with col1:
                criterion = st.selectbox("Criterion", ["gini", "entropy"] if problem_type == "Klasifikasi" else ["squared_error", "friedman_mse", "absolute_error"])
                max_depth = st.number_input("Max Depth", 1, 100, 10)
                max_depth = None if st.checkbox("Unlimited depth") else max_depth
            with col2:
                min_samples_split = st.number_input("Min Samples Split", 2, 20, 2)
                min_samples_leaf = st.number_input("Min Samples Leaf", 1, 20, 1)

            params = {"max_depth": [max_depth], "min_samples_split": [min_samples_split], "min_samples_leaf": [min_samples_leaf]}
            if problem_type == "Klasifikasi":
                model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
            else:
                model = DecisionTreeRegressor(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)

        elif model_name in ["Gradient Boosting", "Gradient Boosting Regressor"]:
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.number_input("Number of Estimators", 50, 1000, 100)
                learning_rate = st.number_input("Learning Rate", 0.01, 1.0, 0.1, 0.01)
            with col2:
                max_depth = st.number_input("Max Depth", 1, 10, 3)
                subsample = st.number_input("Subsample", 0.5, 1.0, 1.0, 0.1)
            
            params = {"n_estimators": [n_estimators], "learning_rate": [learning_rate], "max_depth": [max_depth]}
            if problem_type == "Klasifikasi":
                model = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, random_state=42)
            else:
                model = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, subsample=subsample, random_state=42)

        elif model_name in ["AdaBoost", "AdaBoost Regressor"]:
            col1, col2 = st.columns(2)
            with col1:
                n_estimators = st.number_input("Number of Estimators", 10, 500, 50)
            with col2:
                learning_rate = st.number_input("Learning Rate", 0.01, 2.0, 1.0, 0.01)
            
            params = {"n_estimators": [n_estimators], "learning_rate": [learning_rate]}
            if problem_type == "Klasifikasi":
                model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
            else:
                model = AdaBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)

        elif model_name in ["MLP Classifier", "MLP Regressor"]:
            hidden_layer_sizes = st.text_input("Hidden Layer Sizes (comma-separated)", "100,")
            hidden_layer_sizes = tuple(int(x.strip()) for x in hidden_layer_sizes.split(',') if x.strip())
            
            col1, col2 = st.columns(2)
            with col1:
                activation = st.selectbox("Activation Function", ["relu", "identity", "logistic", "tanh"])
                solver = st.selectbox("Solver", ["adam", "sgd", "lbfgs"])
            with col2:
                alpha = st.number_input("Alpha (L2 regularization)", 0.00001, 0.1, 0.0001, format="%.5f")
                learning_rate = st.selectbox("Learning Rate", ["constant", "invscaling", "adaptive"])

            params = {"hidden_layer_sizes": [hidden_layer_sizes], "activation": [activation], "solver": [solver], "alpha": [alpha]}
            if problem_type == "Klasifikasi":
                model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, learning_rate=learning_rate, max_iter=1000, random_state=42)
            else:
                model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, learning_rate=learning_rate, max_iter=1000, random_state=42)

        # Add more model configurations...
        elif model_name == "Naive Bayes":
            nb_type = st.selectbox("Naive Bayes Type", ["Gaussian", "Multinomial", "Bernoulli"])
            if nb_type == "Gaussian":
                var_smoothing = st.number_input("Variance Smoothing", 1e-10, 1e-5, 1e-9, format="%.2e")
                model = GaussianNB(var_smoothing=var_smoothing)
            elif nb_type == "Multinomial":
                alpha = st.number_input("Alpha (Smoothing)", 0.0, 2.0, 1.0, 0.1)
                model = MultinomialNB(alpha=alpha)
            else:  # Bernoulli
                alpha = st.number_input("Alpha (Smoothing)", 0.0, 2.0, 1.0, 0.1)
                binarize = st.number_input("Binarize threshold", 0.0, 1.0, 0.0)
                model = BernoulliNB(alpha=alpha, binarize=binarize)
            params = {}
        
        elif model_name in ["Ridge Regression", "Lasso Regression", "ElasticNet"]:
            col1, col2 = st.columns(2)
            with col1:
                alpha = st.number_input("Alpha (Regularization)", 0.01, 10.0, 1.0, 0.01)
            with col2:
                max_iter = st.number_input("Max Iterations", 100, 5000, 1000)
                if model_name == "ElasticNet":
                    l1_ratio = st.number_input("L1 Ratio", 0.0, 1.0, 0.5, 0.1)
            
            params = {"alpha": [alpha]}
            if model_name == "Ridge Regression":
                model = Ridge(alpha=alpha, max_iter=max_iter, random_state=42)
            elif model_name == "Lasso Regression":
                model = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
            else:  # ElasticNet
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=42)
        
        elif model_name == "Spectral Clustering":
            col1, col2 = st.columns(2)
            with col1:
                n_clusters = st.number_input("Number of Clusters", 2, 20, 3)
                affinity = st.selectbox("Affinity", ['rbf', 'nearest_neighbors', 'precomputed'])
            with col2:
                if affinity == 'rbf':
                    gamma = st.number_input("Gamma", 0.1, 10.0, 1.0, 0.1)
                elif affinity == 'nearest_neighbors':
                    n_neighbors = st.number_input("Number of Neighbors", 2, 50, 10)
            
            params = {"n_clusters": [n_clusters]}
            if affinity == 'rbf':
                model = SpectralClustering(n_clusters=n_clusters, affinity=affinity, gamma=gamma, random_state=42)
            elif affinity == 'nearest_neighbors':
                model = SpectralClustering(n_clusters=n_clusters, affinity=affinity, n_neighbors=n_neighbors, random_state=42)
            else: # precomputed
                model = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=42)
        
        # Default fallback for other models
        else:
            # Add basic parameters for other models
            if model_name == "Linear Regression":
                model = LinearRegression()
                params = {}
            elif model_name == "KMeans":
                n_clusters = st.number_input("Number of Clusters", 2, 20, 3)
                init = st.selectbox("Initialization", ['k-means++', 'random'])
                model = KMeans(n_clusters=n_clusters, init=init, random_state=42, n_init=10)
                params = {"n_clusters": [n_clusters]}
            elif model_name == "DBSCAN":
                eps = st.number_input("Epsilon", 0.1, 5.0, 0.5, 0.1)
                min_samples = st.number_input("Min Samples", 1, 20, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
                params = {"eps": [eps], "min_samples": [min_samples]}
            elif model_name == "AgglomerativeClustering":
                n_clusters = st.number_input("Number of Clusters", 2, 20, 3)
                linkage = st.selectbox("Linkage", ['ward', 'complete', 'average', 'single'])
                model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
                params = {"n_clusters": [n_clusters]}
            elif model_name == "Birch":
                n_clusters = st.number_input("Number of Clusters", 2, 20, 3)
                threshold = st.number_input("Threshold", 0.1, 1.0, 0.5, 0.1)
                model = Birch(n_clusters=n_clusters, threshold=threshold)
                params = {"n_clusters": [n_clusters]}
            else:
                # Default fallback for any unhandled model
                st.error(f"Model '{model_name}' is not yet implemented in the create_model_with_params function.")
                model = None
                params = {}
        
        return model, params
    
    @staticmethod
    def get_base_model(model_name):
        """Get base model for GridSearchCV"""
        base_models = {
            "Logistic Regression": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Random Forest Regressor": RandomForestRegressor(random_state=42),
            "SVM": SVC(probability=True, random_state=42),
            "SVM Regressor": SVR(),
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(random_state=42),
            "Lasso Regression": Lasso(random_state=42),
            "ElasticNet": ElasticNet(random_state=42),
            "KNN": KNeighborsClassifier(),
            "KNN Regressor": KNeighborsRegressor(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
            "AdaBoost": AdaBoostClassifier(random_state=42),
            "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            "MLP Classifier": MLPClassifier(random_state=42, max_iter=1000),
            "MLP Regressor": MLPRegressor(random_state=42, max_iter=1000),
            "Naive Bayes": GaussianNB(),
            "Extra Trees": ExtraTreesClassifier(random_state=42),
            "Extra Trees Regressor": ExtraTreesRegressor(random_state=42),
            "KMeans": KMeans(random_state=42),
            "DBSCAN": DBSCAN(),
            "Spectral Clustering": SpectralClustering(random_state=42),
            "AgglomerativeClustering": AgglomerativeClustering(),
            "Birch": Birch()
        }
        return base_models.get(model_name)
    
    @staticmethod
    def evaluate_classification(y_test, y_pred, y_proba=None):
        """Comprehensive classification evaluation"""
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0),
            "classification_report": classification_report(y_test, y_pred, zero_division=0, output_dict=True),
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }
        
        # Add AUC for binary classification
        if len(np.unique(y_test)) == 2 and y_proba is not None:
            try:
                results["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
                results["roc_curve"] = roc_curve(y_test, y_proba[:, 1])
                results["pr_curve"] = precision_recall_curve(y_test, y_proba[:, 1])
            except:
                pass
        
        return results
    
    @staticmethod
    def evaluate_regression(y_test, y_pred):
        """Comprehensive regression evaluation"""
        results = {
            "r2_score": r2_score(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "explained_variance": explained_variance_score(y_test, y_pred),
            "max_error": max_error(y_test, y_pred),
            "mape": np.mean(np.abs((y_test - y_pred) / y_test)) * 100 if np.all(y_test != 0) else np.nan
        }
        return results
    
    @staticmethod
    def evaluate_clustering(X, labels):
        """Comprehensive clustering evaluation"""
        results = {}
        
        # Only calculate metrics if we have more than one cluster
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and -1 not in unique_labels:  # -1 indicates noise in DBSCAN
            results["silhouette_score"] = silhouette_score(X, labels)
            results["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)
            results["davies_bouldin_score"] = davies_bouldin_score(X, labels)
        
        results["n_clusters"] = len(unique_labels)
        results["n_noise"] = np.sum(labels == -1) if -1 in unique_labels else 0
        
        return results
    
    @staticmethod
    def plot_classification_results(y_test, y_pred, y_proba=None, class_names=None):
        """Enhanced classification visualization"""
        # Generate default class names if not provided
        if class_names is None:
            unique_classes = sorted(set(list(y_test) + list(y_pred)))
            class_names = [str(cls) for cls in unique_classes]
        
        n_plots = 2 if y_proba is None else 4
        fig, axes = plt.subplots(2, 2 if n_plots == 4 else 1, figsize=(15, 10))
        axes = axes.flatten() if n_plots == 4 else [axes[0], axes[1]]
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                   xticklabels=class_names, yticklabels=class_names)
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Actual")
        axes[0].set_title("Confusion Matrix")
        
        # Classification Report Heatmap
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        # Convert to DataFrame for better visualization
        report_df = pd.DataFrame(report).iloc[:-1, :-3].T  # Remove support column and summary rows
        sns.heatmap(report_df.astype(float), annot=True, fmt=".3f", cmap="Greens", ax=axes[1])
        axes[1].set_title("Classification Report")
        axes[1].set_xlabel("Metrics")
        axes[1].set_ylabel("Classes")
        
        # ROC Curve (for binary classification)
        if len(np.unique(y_test)) == 2 and y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            auc_score = roc_auc_score(y_test, y_proba[:, 1])
            axes[2].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
            axes[2].plot([0, 1], [0, 1], 'k--', label='Random')
            axes[2].set_xlabel("False Positive Rate")
            axes[2].set_ylabel("True Positive Rate")
            axes[2].set_title("ROC Curve")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
            
            # Precision-Recall Curve
            precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
            axes[3].plot(recall, precision, label='PR Curve')
            axes[3].set_xlabel("Recall")
            axes[3].set_ylabel("Precision")
            axes[3].set_title("Precision-Recall Curve")
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_regression_results(y_test, y_pred):
        """Enhanced regression visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
        axes[0, 0].set_xlabel("Actual Values")
        axes[0, 0].set_ylabel("Predicted Values")
        axes[0, 0].set_title("Actual vs. Predicted")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add R² score to the plot
        r2 = r2_score(y_test, y_pred)
        axes[0, 0].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[0, 0].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        # Residuals Plot
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 1].axhline(y=0, color='red', linestyle='--')
        axes[0, 1].set_xlabel("Predicted Values")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residuals Plot")
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals Distribution
        axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_xlabel("Residuals")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].set_title("Residuals Distribution")
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q Plot for residuals normality
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title("Q-Q Plot (Residuals Normality)")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_clustering_results(X, labels, method_name="Clustering"):
        """Enhanced clustering visualization"""
        n_clusters = len(np.unique(labels))
        
        if X.shape[1] >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 2D scatter plot
            scatter = axes[0].scatter(X.iloc[:, 0] if hasattr(X, 'iloc') else X[:, 0], 
                                    X.iloc[:, 1] if hasattr(X, 'iloc') else X[:, 1], 
                                    c=labels, cmap='viridis', alpha=0.7)
            axes[0].set_xlabel(f"Feature 1")
            axes[0].set_ylabel(f"Feature 2")
            axes[0].set_title(f"{method_name} Results (2D)")
            plt.colorbar(scatter, ax=axes[0])
            
            # Cluster size distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            bars = axes[1].bar(range(len(unique_labels)), counts, color='skyblue', alpha=0.7)
            axes[1].set_xlabel("Cluster")
            axes[1].set_ylabel("Number of Points")
            axes[1].set_title("Cluster Size Distribution")
            axes[1].set_xticks(range(len(unique_labels)))
            axes[1].set_xticklabels([f"Cluster {i}" if i != -1 else "Noise" for i in unique_labels])
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                           str(count), ha='center', va='bottom')
        else:
            # 1D visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            scatter = ax.scatter(range(len(labels)), X.iloc[:, 0] if hasattr(X, 'iloc') else X[:, 0], 
                               c=labels, cmap='viridis', alpha=0.7)
            ax.set_xlabel("Sample Index")
            ax.set_ylabel("Feature Value")
            ax.set_title(f"{method_name} Results (1D)")
            plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def display_evaluation_metrics(results, problem_type):
        """Display evaluation metrics in a formatted table with proper data types"""
        if problem_type == "Klasifikasi":
            # Convert to proper data types to avoid Arrow serialization issues
            metrics_data = {
                "Metrik": ["Accuracy", "Precision", "Recall", "F1-Score"],
                "Skor": [
                    float(results["accuracy"]),
                    float(results["precision"]), 
                    float(results["recall"]), 
                    float(results["f1_score"])
                ]
            }
            
            # Add AUC if available
            if "roc_auc" in results:
                metrics_data["Metrik"].append("ROC AUC")
                metrics_data["Skor"].append(float(results["roc_auc"]))
                
        elif problem_type == "Regresi":
            metrics_data = {
                "Metrik": ["R² Score", "MAE", "MSE", "RMSE", "Explained Variance", "Max Error"],
                "Skor": [
                    float(results["r2_score"]), 
                    float(results["mae"]), 
                    float(results["mse"]),
                    float(results["rmse"]), 
                    float(results["explained_variance"]), 
                    float(results["max_error"])
                ]
            }
            
            # Add MAPE if available
            if "mape" in results and not np.isnan(results["mape"]):
                metrics_data["Metrik"].append("MAPE (%)")
                metrics_data["Skor"].append(float(results["mape"]))
                
        else:  # Clustering
            metrics_data = {
                "Metrik": ["Number of Clusters", "Number of Noise Points"],
                "Skor": [int(results["n_clusters"]), int(results["n_noise"])]
            }
            
            # Add clustering metrics if available
            if "silhouette_score" in results:
                metrics_data["Metrik"].extend(["Silhouette Score", "Calinski-Harabasz Score", "Davies-Bouldin Score"])
                metrics_data["Skor"].extend([
                    float(results["silhouette_score"]),
                    float(results["calinski_harabasz_score"]),
                    float(results["davies_bouldin_score"])
                ])
        
        # Create DataFrame and format
        eval_df = pd.DataFrame(metrics_data)
        
        # Format numerical values for better readability
        if problem_type in ["Klasifikasi", "Regresi"]:
            eval_df["Skor"] = eval_df["Skor"].round(4)
        
        return eval_df.set_index("Metrik")
