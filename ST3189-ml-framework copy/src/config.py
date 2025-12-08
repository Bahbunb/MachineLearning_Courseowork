from dataclasses import dataclass
from typing import List

@dataclass
class DataConfig:
    """Configuration for data paths and basic parameters"""
    DATA_PATH_C: str = '/Users/luqman/Desktop/Year 3/ST3189/COURSEWORK/classification/data/data_i.csv'
    DATA_PATH_R: str = '/Users/luqman/Desktop/Year 3/ST3189/COURSEWORK/regression/data/ames_combined_dataset.csv'
    TASK_TYPE: str = 'regression'  # 'classification' or 'regression'
    RANDOM_SEED: int = 24
    N_SPLITS: int = 5

@dataclass
class ClassificationConfig:
    """Configuration for classification task"""
    TARGET_COLUMN: str = 'Bankrupt?'
    DROP_COLUMNS: List[str] = None
    
    def __post_init__(self):
        if self.DROP_COLUMNS is None:
            self.DROP_COLUMNS = [' Net Income Flag']

@dataclass
class FeatureSelectionConfig:
    """Configuration for feature selection methods"""
    # Common parameters
    RANDOM_SEED: int = 24
    
    # Available feature selection methods
    AVAILABLE_METHODS: List[str] = None
    
    # Task-specific methods
    CLASSIFICATION_ONLY_METHODS: List[str] = None
    REGRESSION_ONLY_METHODS: List[str] = None
    
    # PCA parameters
    PCA_VARIANCE_RATIO: float = 0.8
    
    # T-test parameters
    TTEST_SIGNIFICANCE: float = 0.05
    
    # Stepwise selection parameters
    STEPWISE_MIN_FEATURES: int = 10
    STEPWISE_CV: int = 5
    STEPWISE_SCORING: str = 'precision'

    # KMeans parameters
    KMEANS_MAX_CLUSTERS: int = 10
    KMEANS_MIN_CLUSTERS: int = 2
    
    # Correlation parameters
    CORRELATION_THRESHOLD: float = 0.1

    # Lasso parameters
    LASSO_ALPHA: float = 1000
    
    def __post_init__(self):
        if self.AVAILABLE_METHODS is None:
            self.AVAILABLE_METHODS = ['none', 'pca', 'kmeans', 'correlation']
            
        if self.CLASSIFICATION_ONLY_METHODS is None:
            self.CLASSIFICATION_ONLY_METHODS = ['ttest', 'stepwise']
            
        if self.REGRESSION_ONLY_METHODS is None:
            self.REGRESSION_ONLY_METHODS = ['lasso']
    
    @property
    def METHODS_FOR_TASK(self):
        """Get available feature selection methods based on task type"""
        return {
            'classification': self.AVAILABLE_METHODS + self.CLASSIFICATION_ONLY_METHODS,
            'regression': self.AVAILABLE_METHODS + self.REGRESSION_ONLY_METHODS
        }

@dataclass
class ModelConfig:
    """Configuration for different models"""
    # SVM parameters
    SVM_KERNEL: str = 'rbf'
    SVM_C: float = 1.0
    
    # Neural Network parameters
    HIDDEN_LAYERS: int = 1
    HIDDEN_NODES: int = 64
    EPOCHS: int = 50
    
    # Random Forest parameters
    RF_N_ESTIMATORS: int = 100
    RF_MAX_DEPTH: int = 5
    
    # XGBoost parameters
    XGB_N_ESTIMATORS: int = 250
    XGB_MAX_DEPTH: int = 7
    XGB_LEARNING_RATE: float = 0.01

@dataclass
class RegressionConfig:
    """Configuration for regression task"""
    TARGET_COLUMN: str = 'SalePrice'
    DROP_COLUMNS: List[str] = None
    LOG_TRANSFORM_TARGET: bool = True
    
    def __post_init__(self):
        if self.DROP_COLUMNS is None:
            self.DROP_COLUMNS = ['Utilities']

class Config:
    """Main configuration class that combines all config components"""
    def __init__(self, **kwargs):
        """Initialize configuration with optional overrides"""
        self.data = DataConfig()
        self.classification = ClassificationConfig()
        self.feature_selection = FeatureSelectionConfig()
        self.model = ModelConfig()
        self.regression = RegressionConfig()
        
        # Override default values with provided kwargs
        self._override_config(kwargs)
    
    def _override_config(self, kwargs):
        """Override default configuration values"""
        for key, value in kwargs.items():
            # Split the key into component and parameter
            if '.' in key:
                component, param = key.split('.')
                if hasattr(self, component) and hasattr(getattr(self, component), param):
                    setattr(getattr(self, component), param, value)
                else:
                    raise ValueError(f"Invalid configuration parameter: {key}")
            else:
                raise ValueError(f"Configuration parameter must be in format 'component.parameter': {key}")
    
    @property
    def DATA_PATH(self):
        """Get the appropriate data path based on task type"""
        return (self.data.DATA_PATH_C 
                if self.data.TASK_TYPE == 'classification' 
                else self.data.DATA_PATH_R)
    
    @property
    def TARGET_COLUMN(self):
        """Get target column name based on task type"""
        return (self.classification.TARGET_COLUMN 
                if self.data.TASK_TYPE == 'classification' 
                else self.regression.TARGET_COLUMN)
    
    @property
    def DROP_COLUMNS(self):
        """Get columns to drop based on task type"""
        return (self.classification.DROP_COLUMNS 
                if self.data.TASK_TYPE == 'classification' 
                else self.regression.DROP_COLUMNS)
    
    @property
    def RANDOM_SEED(self):
        """Get random seed"""
        return self.data.RANDOM_SEED
    
    @property
    def N_SPLITS(self):
        """Get number of CV splits"""
        return self.data.N_SPLITS

    def __str__(self):
        """String representation of the configuration"""
        return (
            f"Configuration:\n"
            f"Task Type: {self.data.TASK_TYPE}\n"
            f"Random Seed: {self.data.RANDOM_SEED}\n"
            f"CV Splits: {self.data.N_SPLITS}\n"
            f"Feature Selection:\n"
            f"  PCA Variance Ratio: {self.feature_selection.PCA_VARIANCE_RATIO}\n"
            f"  T-test Significance: {self.feature_selection.TTEST_SIGNIFICANCE}\n"
            f"Model Parameters:\n"
            f"  SVM Kernel: {self.model.SVM_KERNEL}\n"
            f"  Hidden Layers: {self.model.HIDDEN_LAYERS}\n"
            f"  Hidden Nodes: {self.model.HIDDEN_NODES}\n"
        )

config = Config()

# Get available feature selection methods based on task
if config.data.TASK_TYPE == 'classification':
    methods = config.feature_selection.CLASSIFICATION_ONLY_METHODS
else:
    methods = config.feature_selection.REGRESSION_ONLY_METHODS

# Or using the property
methods = config.feature_selection.METHODS_FOR_TASK[config.data.TASK_TYPE]