import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold
from scipy import stats
from imblearn.over_sampling import SMOTE
from logger import PreprocessingLogger

class DataValidator:
    """Separate class for data validation concerns"""
    @staticmethod
    def validate_input(X, y=None):
        """Validate input data"""
        if X is None:
            raise ValueError("X cannot be None")
        
        if X.isnull().any().any():
            raise ValueError("Input features contain null values")
            
        if y is not None:
            if y.isnull().any():
                raise ValueError("Target variable contains null values")
            if len(X) != len(y):
                raise ValueError("X and y must have same length")

class DataLoader:
    """Separate class for data loading concerns"""
    @staticmethod
    def load_data(filepath, target_col, drop_cols):
        """Load and prepare the dataset"""
        df = pd.read_csv(filepath)
        df = df.drop(columns=drop_cols)
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return X, y

class OutlierDetector:
    """Separate class for outlier detection"""
    @staticmethod
    def detect_outliers(X, z_threshold=3):
        """Detect outliers using z-score method"""
        outliers = np.zeros(X.shape[0], dtype=bool)
        
        for column in X.columns:
            z_scores = np.abs(stats.zscore(X[column]))
            outliers = outliers | (z_scores > z_threshold)
            
        return outliers

class CrossValidationSplitter:
    """Separate class for cross-validation splitting"""
    def __init__(self, random_seed):
        self.random_seed = random_seed

    def get_stratified_folds(self, X, y, n_splits):
        """Create folds for cross-validation based on task type"""
        if isinstance(y.iloc[0], (int, bool)) and len(np.unique(y)) < 10:  # Classification
            cv = StratifiedKFold(
                n_splits=n_splits, 
                shuffle=True, 
                random_state=self.random_seed
            )
        else:  # Regression
            cv = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=self.random_seed
            )
        return cv.split(X, y)

class DataScaler:
    """Separate class for data scaling"""
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, X):
        """Scale features using StandardScaler"""
        return pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    
    def transform(self, X):
        """Transform data using pre-fitted scaler"""
        return pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )

class DataBalancer:
    """Separate class for handling class imbalance"""
    def __init__(self, random_seed):
        self.random_seed = random_seed
        self.smote = SMOTE(random_state=random_seed)

    def balance_classes(self, X, y):
        """Apply SMOTE oversampling to handle class imbalance"""
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled)
        
        return X_resampled, y_resampled

class DataPreprocessor:
    """Main class orchestrating all preprocessing steps"""
    def __init__(self, config):
        self.config = config
        self.validator = DataValidator()
        self.loader = DataLoader()
        self.outlier_detector = OutlierDetector()
        self.cv_splitter = CrossValidationSplitter(config.RANDOM_SEED)
        self.scaler = DataScaler()
        self.balancer = DataBalancer(config.RANDOM_SEED) if config.data.TASK_TYPE == 'classification' else None
        self.logger = PreprocessingLogger()
        self.log_transform = config.data.TASK_TYPE == 'regression' and config.regression.LOG_TRANSFORM_TARGET

    def load_data(self, filepath, target_col, drop_cols):
        """Load and validate data"""
        X, y = self.loader.load_data(filepath, target_col, drop_cols)
        self.validator.validate_input(X, y)
        
        # Apply log transformation to target if specified
        if self.log_transform:
            if (y <= 0).any():
                raise ValueError("Cannot apply log transform to non-positive values")
            y = np.log(y)
            self.logger.log_info("Applied log transformation to target variable")
            
        self.logger.log_data_shape(X, y)
        return X, y

    def detect_outliers(self, X, z_threshold=3):
        """Detect outliers in data"""
        outliers = self.outlier_detector.detect_outliers(X, z_threshold)
        self.logger.log_outliers(np.sum(outliers))
        return outliers

    def get_stratified_folds(self, X, y, n_splits):
        """Get cross-validation folds"""
        return self.cv_splitter.get_stratified_folds(X, y, n_splits)

    def fit_scale_features(self, X):
        """Scale features"""
        self.validator.validate_input(X)
        return self.scaler.fit_transform(X)

    def transform_features(self, X):
        """Transform features using fitted scaler"""
        return self.scaler.transform(X)

    def apply_oversampling(self, X, y):
        """Apply oversampling only for classification tasks"""
        if self.config.data.TASK_TYPE == 'classification':
            return self.balancer.balance_classes(X, y)
        return X, y  # For regression, return original data

    def preprocess_pipeline(self, X, y, detect_outliers=True, apply_scaling=True, 
                          apply_balancing=True, z_threshold=3):
        """Complete preprocessing pipeline"""
        try:
            # Validate input
            self.validator.validate_input(X, y)
            
            # Detect outliers if requested
            if detect_outliers:
                outliers = self.detect_outliers(X, z_threshold)
                X = X[~outliers]
                y = y[~outliers]
            
            # Scale features if requested
            if apply_scaling:
                X = self.fit_scale_features(X)
            
            # Apply oversampling only for classification tasks
            if apply_balancing and self.config.data.TASK_TYPE == 'classification':
                X, y = self.apply_oversampling(X, y)
            
            return X, y
            
        except Exception as e:
            self.logger.log_error(str(e))
            raise 