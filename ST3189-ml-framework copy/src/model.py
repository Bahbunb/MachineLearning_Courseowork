"""
Model implementations for classification and regression tasks.

This module provides a unified interface for various ML models with support for:
- Classification: Neural Network, LDA, SVM
- Regression: Linear Regression, Random Forest, XGBoost

Design Patterns:
- Abstract Factory: ModelFactory creates appropriate model instances
- Template Method: BaseModel defines the training/prediction interface
- Strategy: Different models implement the same interface
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Union, Any
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    confusion_matrix, 
    mean_squared_error, 
    r2_score, 
    mean_absolute_error
)
import xgboost as xgb

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from logger import ModelLogger
from visualization import DataVisualizer


class BaseModel(ABC):
    """
    Abstract base class for all models.
    
    Provides common functionality for model training, prediction, and validation.
    All concrete model implementations must inherit from this class.
    
    Attributes:
        config: Configuration object containing model hyperparameters
        model: The underlying sklearn/xgboost model instance
        logger: Logger for tracking model operations
        is_fitted: Boolean indicating if the model has been trained
    """
    
    def __init__(self, config):
        """
        Initialize the base model.
        
        Args:
            config: Configuration object with model parameters
        """
        self.config = config
        self.model = None
        self.logger = ModelLogger()
        self.is_fitted = False
    
    @abstractmethod
    def _initialize_model(self) -> None:
        """Initialize the underlying model. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'BaseModel':
        """
        Train the model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            self: The fitted model instance
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def get_detailed_metrics(self, y_true: np.ndarray, 
                            y_pred: np.ndarray) -> Tuple:
        """
        Calculate detailed performance metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Tuple of metric values
        """
        pass
    
    def _validate_fitted(self) -> None:
        """Validate that the model has been fitted before prediction."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before making predictions. "
                "Call fit() first."
            )
    
    def _validate_model_initialized(self) -> None:
        """Validate that the model has been initialized."""
        if self.model is None:
            raise RuntimeError(
                f"{self.__class__.__name__} model not initialized. "
                "This is likely an implementation error."
            )
    
    def _convert_to_array(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Convert input to numpy array if needed."""
        if isinstance(X, pd.DataFrame):
            return X.values
        return np.asarray(X)
    
    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame], 
                       y: Optional[Union[np.ndarray, pd.Series]] = None) -> None:
        """
        Validate input data.
        
        Args:
            X: Feature matrix
            y: Optional target variable
            
        Raises:
            ValueError: If input validation fails
        """
        if X is None:
            raise ValueError("X cannot be None")
        
        X_arr = self._convert_to_array(X)
        
        if X_arr.size == 0:
            raise ValueError("X cannot be empty")
        
        if np.isnan(X_arr).any():
            raise ValueError("X contains NaN values")
        
        if np.isinf(X_arr).any():
            raise ValueError("X contains infinite values")
        
        if y is not None:
            y_arr = np.asarray(y)
            if len(X_arr) != len(y_arr):
                raise ValueError(
                    f"X and y must have the same number of samples. "
                    f"Got X: {len(X_arr)}, y: {len(y_arr)}"
                )


class ClassificationModel(BaseModel):
    """
    Base class for classification models.
    
    Extends BaseModel with classification-specific functionality including
    probability predictions and classification metrics (accuracy, Type I/II errors).
    """
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'ClassificationModel':
        """
        Train the classification model.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            self: The fitted model instance
        """
        self._validate_model_initialized()
        self._validate_input(X, y)
        
        self.logger.log_training_start(self.__class__.__name__)
        
        X_arr = self._convert_to_array(X)
        y_arr = np.asarray(y)
        
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
        """
        self._validate_fitted()
        self._validate_input(X)
        
        X_arr = self._convert_to_array(X)
        return self.model.predict(X_arr)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Class probability matrix
        """
        self._validate_fitted()
        self._validate_input(X)
        
        X_arr = self._convert_to_array(X)
        
        if not hasattr(self.model, 'predict_proba'):
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support probability predictions"
            )
        
        return self.model.predict_proba(X_arr)
    
    def get_detailed_metrics(self, y_true: np.ndarray, 
                            y_pred: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate classification metrics.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Tuple of (accuracy, type_i_error, type_ii_error)
            - accuracy: Overall classification accuracy
            - type_i_error: False positive rate (FP / (FP + TN))
            - type_ii_error: False negative rate (FN / (FN + TP))
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Handle binary classification
        cm = confusion_matrix(y_true, y_pred)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            # Type I Error (False Positive Rate)
            type_i_error = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            # Type II Error (False Negative Rate)  
            type_ii_error = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        else:
            # Multi-class: use macro-averaged metrics
            accuracy = np.trace(cm) / np.sum(cm)
            type_i_error = 0.0  # Not directly applicable
            type_ii_error = 0.0  # Not directly applicable
            self.logger.log_info(
                "Multi-class classification detected. "
                "Type I/II errors not applicable."
            )
        
        return accuracy, type_i_error, type_ii_error


class RegressionModel(BaseModel):
    """
    Base class for regression models.
    
    Extends BaseModel with regression-specific functionality including
    support for log-transformed targets and regression metrics (RMSE, R², MAE).
    
    Attributes:
        log_transform: Whether the target was log-transformed during preprocessing
        visualizer: DataVisualizer instance for generating plots
    """
    
    def __init__(self, config):
        """
        Initialize the regression model.
        
        Args:
            config: Configuration object with model parameters
        """
        super().__init__(config)
        self.log_transform = (
            config.data.TASK_TYPE == 'regression' and 
            config.regression.LOG_TRANSFORM_TARGET
        )
        self.visualizer = DataVisualizer(output_dir="plots")
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame], 
            y: Union[np.ndarray, pd.Series]) -> 'RegressionModel':
        """
        Train the regression model.
        
        Args:
            X: Feature matrix
            y: Target values (in log-space if log_transform=True)
            
        Returns:
            self: The fitted model instance
        """
        self._validate_model_initialized()
        self._validate_input(X, y)
        
        self.logger.log_training_start(self.__class__.__name__)
        
        X_arr = self._convert_to_array(X)
        y_arr = np.asarray(y)
        
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True
        
        return self
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions in the same space as training targets.
        
        If log_transform was applied during preprocessing, predictions
        are returned in log-space. Use predict_original_scale() to get
        predictions in the original scale.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions (in log-space if log_transform=True)
        """
        self._validate_fitted()
        self._validate_input(X)
        
        X_arr = self._convert_to_array(X)
        return self.model.predict(X_arr)
    
    def predict_original_scale(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions in original scale.
        
        If log_transform was applied, exponentiates the predictions.
        Otherwise, returns predictions directly.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions in original scale
        """
        predictions = self.predict(X)
        
        if self.log_transform:
            return np.exp(predictions)
        return predictions
    
    def get_detailed_metrics(self, y_true: np.ndarray, 
                            y_pred: np.ndarray) -> Tuple:
        """
        Calculate regression metrics.
        
        Both y_true and y_pred should be in the same space (both log or both original).
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            If log_transform=True:
                Tuple of (log_rmse, r2, log_mae, approx_original_rmse)
            Else:
                Tuple of (rmse, r2, mae)
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        if len(y_true) == 0:
            raise ValueError("Cannot calculate metrics on empty arrays")
        
        if self.log_transform:
            return self._calculate_log_space_metrics(y_true, y_pred)
        else:
            return self._calculate_original_space_metrics(y_true, y_pred)
    
    def _calculate_log_space_metrics(self, y_true: np.ndarray, 
                                     y_pred: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Calculate metrics when working in log-space.
        
        Args:
            y_true: True values in log-space
            y_pred: Predicted values in log-space
            
        Returns:
            Tuple of (log_rmse, r2, log_mae, approx_original_rmse)
        """
        # Metrics in log-space
        log_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        log_mae = mean_absolute_error(y_true, y_pred)
        
        # Approximate RMSE in original scale
        # Using the property that for log-normal errors:
        # RMSE_orig ≈ mean(y_true_orig) * sqrt(exp(log_rmse²) - 1)
        y_true_orig = np.exp(y_true)
        mean_y_orig = np.mean(y_true_orig)
        approx_rmse_orig = mean_y_orig * np.sqrt(np.exp(log_rmse**2) - 1)
        
        self.logger.log_info(f"\nMetrics in log-space:")
        self.logger.log_info(f"  Log-space RMSE: {log_rmse:.4f}")
        self.logger.log_info(f"  R²: {r2:.4f}")
        self.logger.log_info(f"  Log-space MAE: {log_mae:.4f}")
        self.logger.log_info(f"  Approx. Original-scale RMSE: ${approx_rmse_orig:,.2f}")
        
        return log_rmse, r2, log_mae, approx_rmse_orig
    
    def _calculate_original_space_metrics(self, y_true: np.ndarray, 
                                          y_pred: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate metrics in original space.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Tuple of (rmse, r2, mae)
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        self.logger.log_info(f"\nRegression Metrics:")
        self.logger.log_info(f"  RMSE: {rmse:.4f}")
        self.logger.log_info(f"  R²: {r2:.4f}")
        self.logger.log_info(f"  MAE: {mae:.4f}")
        
        return rmse, r2, mae
    
    def get_feature_importance(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get feature importance scores.
        
        Uses model coefficients for linear models, built-in feature_importances_
        for tree-based models, or SHAP values as fallback.
        
        Args:
            X: Feature matrix (used for SHAP calculation if needed)
            
        Returns:
            Array of feature importance scores
        """
        self._validate_fitted()
        
        # Linear models: use coefficients
        if hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_)
        
        # Tree-based models: use built-in importance
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        
        # Fallback: use SHAP values
        if SHAP_AVAILABLE:
            X_arr = self._convert_to_array(X)
            explainer = shap.Explainer(self.model, X_arr)
            shap_values = explainer(X_arr)
            return np.abs(shap_values.values).mean(axis=0)
        
        raise NotImplementedError(
            f"Feature importance not available for {self.__class__.__name__}"
        )
    
    def plot_predictions(self, X: Union[np.ndarray, pd.DataFrame], 
                        y: Union[np.ndarray, pd.Series],
                        use_original_scale: bool = True) -> None:
        """
        Plot predicted vs actual values.
        
        Args:
            X: Feature matrix
            y: True target values
            use_original_scale: If True and log_transform=True, convert to original scale
        """
        self._validate_fitted()
        
        y_true = np.asarray(y)
        
        if use_original_scale and self.log_transform:
            y_pred = self.predict_original_scale(X)
            y_true = np.exp(y_true)
            title = f"Prediction vs Actual - {self.__class__.__name__} (Original Scale)"
            log_scale = True
        else:
            y_pred = self.predict(X)
            title = f"Prediction vs Actual - {self.__class__.__name__}"
            log_scale = False
        
        self.visualizer.plot_prediction_vs_actual(
            y_true, y_pred,
            title=title,
            log_scale=log_scale
        )
    
    def plot_feature_importance(self, X: Union[np.ndarray, pd.DataFrame],
                               feature_names: Optional[List[str]] = None) -> None:
        """
        Plot feature importance.
        
        For tree-based models (RF, XGBoost), generates SHAP beeswarm plots.
        For linear models, generates coefficient bar plots.
        
        Args:
            X: Feature matrix
            feature_names: Optional list of feature names
        """
        self._validate_fitted()
        
        # Get feature names
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = X.columns.tolist()
            else:
                feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        X_arr = self._convert_to_array(X)
        
        # Linear models: plot coefficients
        if isinstance(self.model, LinearRegression):
            self.visualizer.plot_feature_importance(
                self.model,
                feature_names,
                title=f"Feature Importance - {self.__class__.__name__}"
            )
            return
        
        # Tree-based models: use SHAP
        if SHAP_AVAILABLE and isinstance(self.model, (RandomForestRegressor, xgb.XGBRegressor)):
            try:
                explainer = shap.Explainer(self.model, X_arr)
                shap_values = explainer(X_arr)
                shap_values.feature_names = feature_names
                
                self.visualizer.plot_shap_values(
                    shap_values,
                    feature_names,
                    title=f"SHAP Values - {self.__class__.__name__}"
                )
                
                self.visualizer.plot_shap_force(
                    explainer,
                    shap_values,
                    X_arr,
                    n_samples=2,
                    title=f"SHAP Force Plot - {self.__class__.__name__}"
                )
            except Exception as e:
                self.logger.log_error(f"Error generating SHAP plots: {str(e)}")
        else:
            self.logger.log_info(
                "SHAP not available or model type not supported for SHAP plots"
            )


# =============================================================================
# Classification Model Implementations
# =============================================================================

class NeuralNetwork(ClassificationModel):
    """
    Multi-layer Perceptron (MLP) classifier.
    
    A neural network with configurable hidden layers and nodes.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the MLP classifier."""
        hidden_layer_sizes = (
            (self.config.model.HIDDEN_NODES,) * self.config.model.HIDDEN_LAYERS
        )
        
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=self.config.model.EPOCHS,
            random_state=self.config.data.RANDOM_SEED,
            early_stopping=True,
            validation_fraction=0.1
        )


class LDA(ClassificationModel):
    """
    Linear Discriminant Analysis classifier.
    
    A linear classifier that models class-conditional densities
    as multivariate Gaussians with shared covariance.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the LDA classifier."""
        self.model = LinearDiscriminantAnalysis(solver='svd')


class SVM(ClassificationModel):
    """
    Support Vector Machine classifier.
    
    A kernel-based classifier with configurable kernel and regularization.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the SVM classifier."""
        self.model = SVC(
            kernel=self.config.model.SVM_KERNEL,
            gamma='scale',
            C=self.config.model.SVM_C,
            random_state=self.config.data.RANDOM_SEED,
            class_weight='balanced',
            probability=True
        )


# =============================================================================
# Regression Model Implementations
# =============================================================================

class LinearReg(RegressionModel):
    """
    Ordinary Least Squares Linear Regression.
    
    Fits a linear model to minimize the residual sum of squares.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the Linear Regression model."""
        self.model = LinearRegression()


class RandomForestReg(RegressionModel):
    """
    Random Forest Regressor.
    
    An ensemble of decision trees with configurable depth and number of estimators.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the Random Forest model."""
        self.model = RandomForestRegressor(
            n_estimators=self.config.model.RF_N_ESTIMATORS,
            max_depth=self.config.model.RF_MAX_DEPTH,
            random_state=self.config.data.RANDOM_SEED,
            n_jobs=-1  # Use all available cores
        )


class XGBoostReg(RegressionModel):
    """
    XGBoost Regressor.
    
    Gradient boosted decision trees with configurable learning rate and depth.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the XGBoost model."""
        self.model = xgb.XGBRegressor(
            n_estimators=self.config.model.XGB_N_ESTIMATORS,
            max_depth=self.config.model.XGB_MAX_DEPTH,
            learning_rate=self.config.model.XGB_LEARNING_RATE,
            random_state=self.config.data.RANDOM_SEED,
            n_jobs=-1,  # Use all available cores
            verbosity=0  # Suppress warnings
        )


# =============================================================================
# Model Factory
# =============================================================================

class ModelFactory:
    """
    Factory class for creating model instances.
    
    Provides a centralized way to instantiate models based on their type string.
    """
    
    # Registry of available models
    _models = {
        # Classification models
        'nn': NeuralNetwork,
        'lda': LDA,
        'svm': SVM,
        # Regression models
        'linear': LinearReg,
        'rf': RandomForestReg,
        'xgb': XGBoostReg
    }
    
    # Model type categories
    CLASSIFICATION_MODELS = {'nn', 'lda', 'svm'}
    REGRESSION_MODELS = {'linear', 'rf', 'xgb'}
    
    @classmethod
    def create_model(cls, model_type: str, config) -> BaseModel:
        """
        Create a model instance based on the specified type.
        
        Args:
            model_type: String identifier for the model type
                       ('nn', 'lda', 'svm', 'linear', 'rf', 'xgb')
            config: Configuration object with model parameters
            
        Returns:
            Instantiated model object
            
        Raises:
            ValueError: If model_type is unknown or incompatible with task type
        """
        model_type = model_type.lower()
        
        if model_type not in cls._models:
            available = ', '.join(sorted(cls._models.keys()))
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Available models: {available}"
            )
        
        # Validate model-task compatibility
        task_type = config.data.TASK_TYPE
        
        if task_type == 'classification' and model_type in cls.REGRESSION_MODELS:
            raise ValueError(
                f"Model '{model_type}' is a regression model but task type is 'classification'. "
                f"Use one of: {', '.join(sorted(cls.CLASSIFICATION_MODELS))}"
            )
        
        if task_type == 'regression' and model_type in cls.CLASSIFICATION_MODELS:
            raise ValueError(
                f"Model '{model_type}' is a classification model but task type is 'regression'. "
                f"Use one of: {', '.join(sorted(cls.REGRESSION_MODELS))}"
            )
        
        return cls._models[model_type](config)
    
    @classmethod
    def get_available_models(cls, task_type: Optional[str] = None) -> List[str]:
        """
        Get list of available model types.
        
        Args:
            task_type: Optional filter by task type ('classification' or 'regression')
            
        Returns:
            List of model type strings
        """
        if task_type is None:
            return sorted(cls._models.keys())
        elif task_type == 'classification':
            return sorted(cls.CLASSIFICATION_MODELS)
        elif task_type == 'regression':
            return sorted(cls.REGRESSION_MODELS)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    @classmethod
    def register_model(cls, model_type: str, model_class: type, 
                      is_regression: bool = False) -> None:
        """
        Register a new model type.
        
        Args:
            model_type: String identifier for the model
            model_class: Model class (must inherit from BaseModel)
            is_regression: Whether this is a regression model
        """
        if not issubclass(model_class, BaseModel):
            raise TypeError(
                f"Model class must inherit from BaseModel, got {model_class}"
            )
        
        cls._models[model_type.lower()] = model_class
        
        if is_regression:
            cls.REGRESSION_MODELS.add(model_type.lower())
        else:
            cls.CLASSIFICATION_MODELS.add(model_type.lower())


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Base classes
    'BaseModel',
    'ClassificationModel', 
    'RegressionModel',
    # Classification models
    'NeuralNetwork',
    'LDA',
    'SVM',
    # Regression models
    'LinearReg',
    'RandomForestReg',
    'XGBoostReg',
    # Factory
    'ModelFactory'
]
