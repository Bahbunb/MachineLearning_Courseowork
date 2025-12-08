"""
Data preprocessing pipeline for ML framework.

Handles data loading, validation, cleaning, scaling, and cross-validation splitting.
"""

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
    def validate_input(X, y=None, allow_nulls=False):
        """
        Validate input data.
        
        Args:
            X: Feature DataFrame/array
            y: Target variable (optional)
            allow_nulls: If True, skip null check (for pre-cleaning validation)
        """
        if X is None:
            raise ValueError("X cannot be None")
        
        if not allow_nulls and X.isnull().any().any():
            null_cols = X.columns[X.isnull().any()].tolist()
            raise ValueError(
                f"Input features contain null values in columns: {null_cols[:5]}..."
            )
            
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
        
        # Only drop columns that exist
        existing_drop_cols = [col for col in drop_cols if col in df.columns]
        if existing_drop_cols:
            df = df.drop(columns=existing_drop_cols)
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        return X, y


class MissingValueHandler:
    """Handle missing values in the dataset"""
    
    def __init__(self, numeric_strategy='median', categorical_strategy='mode', 
                 drop_threshold=0.5):
        """
        Initialize missing value handler.
        
        Args:
            numeric_strategy: How to fill numeric nulls ('median', 'mean', 'zero')
            categorical_strategy: How to fill categorical nulls ('mode', 'missing')
            drop_threshold: Drop columns with more than this fraction of nulls
        """
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        self.drop_threshold = drop_threshold
        self.fill_values_ = {}
        self.columns_to_drop_ = []
    
    def fit(self, X):
        """
        Learn fill values from training data.
        
        Args:
            X: Training DataFrame
        """
        self.fill_values_ = {}
        self.columns_to_drop_ = []
        
        for col in X.columns:
            null_frac = X[col].isnull().sum() / len(X)
            
            # Drop columns with too many nulls
            if null_frac > self.drop_threshold:
                self.columns_to_drop_.append(col)
                continue
            
            # Skip if no nulls
            if null_frac == 0:
                continue
            
            # Determine fill value based on dtype
            if X[col].dtype in ['int64', 'float64']:
                if self.numeric_strategy == 'median':
                    self.fill_values_[col] = X[col].median()
                elif self.numeric_strategy == 'mean':
                    self.fill_values_[col] = X[col].mean()
                else:
                    self.fill_values_[col] = 0
            else:
                if self.categorical_strategy == 'mode':
                    mode_val = X[col].mode()
                    self.fill_values_[col] = mode_val[0] if len(mode_val) > 0 else 'Missing'
                else:
                    self.fill_values_[col] = 'Missing'
        
        return self
    
    def transform(self, X):
        """
        Apply missing value handling to data.
        
        Args:
            X: DataFrame to transform
            
        Returns:
            Cleaned DataFrame
        """
        X = X.copy()
        
        # Drop high-null columns
        X = X.drop(columns=[c for c in self.columns_to_drop_ if c in X.columns])
        
        # Fill remaining nulls
        for col, fill_val in self.fill_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_val)
        
        return X
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class CategoricalEncoder:
    """Handle categorical variables"""
    
    def __init__(self):
        self.encodings_ = {}
        self.categorical_columns_ = []
    
    def fit(self, X):
        """
        Learn encodings from training data.
        
        Args:
            X: Training DataFrame
        """
        self.encodings_ = {}
        self.categorical_columns_ = []
        
        for col in X.columns:
            if X[col].dtype == 'object':
                self.categorical_columns_.append(col)
                # Create label encoding
                unique_vals = X[col].unique()
                self.encodings_[col] = {val: i for i, val in enumerate(unique_vals)}
        
        return self
    
    def transform(self, X):
        """
        Apply encoding to data.
        
        Args:
            X: DataFrame to transform
            
        Returns:
            Encoded DataFrame
        """
        X = X.copy()
        
        for col in self.categorical_columns_:
            if col in X.columns:
                # Map known values, assign -1 to unknown
                X[col] = X[col].map(self.encodings_[col]).fillna(-1).astype(int)
        
        return X
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)


class OutlierDetector:
    """Separate class for outlier detection"""
    
    @staticmethod
    def detect_outliers(X, z_threshold=3):
        """Detect outliers using z-score method"""
        outliers = np.zeros(X.shape[0], dtype=bool)
        
        # Only check numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        for column in numeric_cols:
            z_scores = np.abs(stats.zscore(X[column], nan_policy='omit'))
            outliers = outliers | (z_scores > z_threshold)
            
        return outliers


class CrossValidationSplitter:
    """Separate class for cross-validation splitting"""
    
    def __init__(self, random_seed):
        self.random_seed = random_seed

    def get_stratified_folds(self, X, y, n_splits):
        """Create folds for cross-validation based on task type"""
        if isinstance(y.iloc[0], (int, bool)) and len(np.unique(y)) < 10:
            # Classification
            cv = StratifiedKFold(
                n_splits=n_splits, 
                shuffle=True, 
                random_state=self.random_seed
            )
        else:
            # Regression
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
        self.feature_names_ = None

    def fit_transform(self, X):
        """Scale features using StandardScaler"""
        self.feature_names_ = X.columns if hasattr(X, 'columns') else None
        
        scaled_data = self.scaler.fit_transform(X)
        
        if self.feature_names_ is not None:
            return pd.DataFrame(
                scaled_data,
                columns=self.feature_names_,
                index=X.index
            )
        return scaled_data
    
    def transform(self, X):
        """Transform data using pre-fitted scaler"""
        scaled_data = self.scaler.transform(X)
        
        if self.feature_names_ is not None:
            return pd.DataFrame(
                scaled_data,
                columns=self.feature_names_,
                index=X.index
            )
        return scaled_data


class DataBalancer:
    """Separate class for handling class imbalance"""
    
    def __init__(self, random_seed):
        self.random_seed = random_seed
        self.smote = SMOTE(random_state=random_seed)

    def balance_classes(self, X, y):
        """Apply SMOTE oversampling to handle class imbalance"""
        X_resampled, y_resampled = self.smote.fit_resample(X, y)
        
        if hasattr(X, 'columns'):
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled)
        
        return X_resampled, y_resampled


class DataPreprocessor:
    """Main class orchestrating all preprocessing steps"""
    
    def __init__(self, config):
        self.config = config
        self.validator = DataValidator()
        self.loader = DataLoader()
        self.missing_handler = MissingValueHandler(
            numeric_strategy='median',
            categorical_strategy='mode',
            drop_threshold=0.5  # Drop columns with >50% missing
        )
        self.encoder = CategoricalEncoder()
        self.outlier_detector = OutlierDetector()
        self.cv_splitter = CrossValidationSplitter(config.RANDOM_SEED)
        self.scaler = DataScaler()
        self.balancer = DataBalancer(config.RANDOM_SEED) if config.data.TASK_TYPE == 'classification' else None
        self.logger = PreprocessingLogger()
        self.log_transform = config.data.TASK_TYPE == 'regression' and config.regression.LOG_TRANSFORM_TARGET

    def load_data(self, filepath, target_col, drop_cols):
        """Load, clean, and validate data"""
        # Load raw data
        X, y = self.loader.load_data(filepath, target_col, drop_cols)
        
        self.logger.log_info(f"Loaded data: {X.shape[0]} rows, {X.shape[1]} features")
        
        # Check for nulls before cleaning
        null_count = X.isnull().sum().sum()
        if null_count > 0:
            self.logger.log_info(f"Found {null_count} null values, cleaning...")
        
        # Handle missing values
        X = self.missing_handler.fit_transform(X)
        
        if self.missing_handler.columns_to_drop_:
            self.logger.log_info(
                f"Dropped {len(self.missing_handler.columns_to_drop_)} high-null columns: "
                f"{self.missing_handler.columns_to_drop_[:5]}..."
            )
        
        # Encode categorical variables
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            self.logger.log_info(f"Encoding {len(cat_cols)} categorical columns")
            X = self.encoder.fit_transform(X)
        
        # Validate cleaned data
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
        return self.scaler.fit_transform(X)

    def transform_features(self, X):
        """Transform features using fitted scaler"""
        return self.scaler.transform(X)

    def apply_oversampling(self, X, y):
        """Apply oversampling only for classification tasks"""
        if self.config.data.TASK_TYPE == 'classification':
            return self.balancer.balance_classes(X, y)
        return X, y

    def preprocess_pipeline(self, X, y, detect_outliers=True, apply_scaling=True, 
                          apply_balancing=True, z_threshold=3):
        """Complete preprocessing pipeline"""
        try:
            # Validate input (allow nulls initially since we'll clean them)
            self.validator.validate_input(X, y, allow_nulls=True)
            
            # Handle missing values
            X = self.missing_handler.fit_transform(X)
            
            # Encode categoricals
            X = self.encoder.fit_transform(X)
            
            # Final validation
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