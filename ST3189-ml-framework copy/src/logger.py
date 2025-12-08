import logging
from datetime import datetime
import os

class BaseLogger:
    """Base logger class with common functionality"""
    def __init__(self, name, log_dir="logs"):
        self.name = name
        self.log_dir = log_dir
        self._setup_logger()
        
    def _setup_logger(self):
        """Setup logger with file and console handlers"""
        # Create logs directory if it doesn't exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        
        log_file = os.path.join(
            self.log_dir, 
            f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        
        # Create and configure console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

class FeatureSelectionLogger(BaseLogger):
    """Logger specifically for feature selection operations"""
    def __init__(self):
        super().__init__(name="feature_selection")
        
    def log_info(self, message):
        """Log informational messages"""
        self.logger.info(message)
        
    def log_results(self, method, initial_features, selected_features):
        """Log feature selection results"""
        reduction_rate = ((initial_features - selected_features) / initial_features) * 100
        
        self.logger.info(f"\nFeature Selection Results - Method: {method}")
        self.logger.info("-" * 50)
        self.logger.info(f"Initial number of features: {initial_features}")
        self.logger.info(f"Selected features: {selected_features}")
        self.logger.info(f"Reduction rate: {reduction_rate:.2f}%")
        
    def log_error(self, method, error_msg):
        """Log feature selection errors"""
        self.logger.error(f"Error in {method}: {error_msg}")
        
    def log_parameters(self, params):
        """Log feature selection parameters"""
        self.logger.info("\nFeature Selection Parameters:")
        self.logger.info("-" * 50)
        for key, value in params.items():
            self.logger.info(f"{key}: {value}")

class ModelLogger(BaseLogger):
    """Logger specifically for model operations"""
    def __init__(self):
        super().__init__(name="model")
        
    def log_info(self, message):
        """Log informational messages"""
        self.logger.info(message)
        
    def log_training_start(self, model_name):
        self.logger.info(f"\nStarting training for {model_name}")
        
    def log_metrics(self, metrics):
        self.logger.info("\nModel Performance Metrics:")
        self.logger.info("-" * 50)
        for metric_name, value in metrics.items():
            self.logger.info(f"{metric_name}: {value:.4f}")
            
    def log_error(self, error_msg):
        self.logger.error(f"Model Error: {error_msg}")

class PreprocessingLogger(BaseLogger):
    """Logger specifically for preprocessing operations"""
    def __init__(self):
        super().__init__(name="preprocessing")
        
    def log_info(self, message):
        """Log informational messages"""
        self.logger.info(message)
        
    def log_data_shape(self, X, y):
        self.logger.info(f"\nData Shape:")
        self.logger.info(f"X shape: {X.shape}")
        self.logger.info(f"y shape: {y.shape}")
        
    def log_outliers(self, n_outliers):
        self.logger.info(f"Detected {n_outliers} outliers")
        
    def log_error(self, error_msg):
        self.logger.error(f"Preprocessing Error: {error_msg}") 