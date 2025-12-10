from abc import ABC, abstractmethod
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from logger import ModelLogger
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import confusion_matrix, mean_squared_error, r2_score
import shap
import numpy as np
from visualization import DataVisualizer

class BaseModel(ABC):
    """Abstract base class for all models"""
    def __init__(self, config):
        self.config = config
        self.model = None
        self.logger = ModelLogger()

    @abstractmethod
    def fit(self, X, y):
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass

    def validate_model(self):
        """Validate model initialization"""
        if self.model is None:
            raise ValueError("Model not initialized")

class ClassificationModel(BaseModel):
    """Base class for classification models"""
    @abstractmethod
    def get_probabilities(self, X):
        """Get probability predictions"""
        pass

    def get_detailed_metrics(self, y_true, y_pred):
        """Calculate detailed classification metrics"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        type1_error = fp / (fp + tn) if (fp + tn) > 0 else 0
        type2_error = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return accuracy, type1_error, type2_error

class RegressionModel(BaseModel):
    """Base class for regression models"""
    def __init__(self, config):
        super().__init__(config)
        self.log_transform = config.regression.LOG_TRANSFORM_TARGET

    def get_detailed_metrics(self, y_true, y_pred):
        """Calculate regression metrics"""
        if self.log_transform:
            # First calculate metrics in log space
            log_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            log_r2 = r2_score(y_true, y_pred)
            log_error = np.mean(np.abs(y_true - y_pred))
            
            # Convert log-RMSE to original scale RMSE
            # If log(y_pred) - log(y_true) = x, then y_pred/y_true = e^x
            # Therefore, RMSE ≈ mean(y_true) * (e^log_rmse - e^-log_rmse)/2
            mean_y_true_orig = np.mean(np.exp(y_true))
            rmse_orig = mean_y_true_orig * (np.exp(log_rmse) - np.exp(-log_rmse))/2
            
            self.logger.log_info(f"\nMetrics in log space:")
            self.logger.log_info(f"Log-space RMSE: {log_rmse:.4f}")
            self.logger.log_info(f"Log-space R²: {log_r2:.4f}")
            self.logger.log_info(f"Log-space MAE: {log_error:.4f}")
            self.logger.log_info(f"Approximate RMSE: {rmse_orig:.4f}")
            
            return log_rmse, log_r2, log_error, rmse_orig
        else:
            # For non-log-transformed data, calculate metrics directly
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            return rmse, r2

    def get_feature_importance(self, X):
        """Get feature importance using appropriate method"""
        self.validate_model()
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif isinstance(self.model, LinearRegression):
            return self.model.coef_
        else:
            explainer = shap.Explainer(self.model)
            return explainer(X)

    def predict(self, X):
        """Make predictions"""
        predictions = super().predict(X)
        
        # If log transform was applied, convert predictions back to original scale
        if self.log_transform:
            predictions = np.exp(predictions)
            
        return predictions

    def plot_predictions(self, X, y):
        """Plot predicted vs actual values"""
        self.validate_model()
        y_pred = self.predict(X)
        
        visualizer = DataVisualizer()
        visualizer.plot_prediction_vs_actual(
            y, y_pred,
            title=f"Prediction vs Actual ({self.__class__.__name__})",
            log_scale=self.log_transform
        )

    def plot_feature_importance(self, X):
        """Plot feature importance"""
        self.validate_model()
        visualizer = DataVisualizer()
        
        if isinstance(self.model, LinearRegression):
            # For Linear Regression, use coefficients
            visualizer.plot_feature_importance(
                self.model,
                X.columns,
                title=f"Top 10 Important Features ({self.__class__.__name__})"
            )
        elif isinstance(self.model, (RandomForestRegressor, xgb.XGBRegressor)):
            # For RF and XGBoost, use SHAP values
            explainer = shap.Explainer(self.model)
            shap_values = explainer(X)
            
            # Plot beeswarm plot
            visualizer.plot_shap_values(
                shap_values,
                X.columns,
                title=f"SHAP Values ({self.__class__.__name__})"
            )
            
            # Add force plots for random observations
            visualizer.plot_shap_force(
                explainer,
                shap_values,
                X,
                n_samples=2,
                title=f"SHAP Force Plot ({self.__class__.__name__})"
            )

# Classification Models
class NeuralNetwork(ClassificationModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = MLPClassifier(
            hidden_layer_sizes=(config.model.HIDDEN_NODES,) * config.model.HIDDEN_LAYERS,
            max_iter=config.model.EPOCHS,
            random_state=config.data.RANDOM_SEED
        )
    
    def fit(self, X, y):
        self.validate_model()
        return self.model.fit(X, y)
    
    def predict(self, X):
        self.validate_model()
        return self.model.predict(X)
    
    def get_probabilities(self, X):
        self.validate_model()
        return self.model.predict_proba(X)

class LDA(ClassificationModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LinearDiscriminantAnalysis(solver='svd')
    
    def fit(self, X, y):
        self.validate_model()
        return self.model.fit(X, y)
    
    def predict(self, X):
        self.validate_model()
        return self.model.predict(X)
    
    def get_probabilities(self, X):
        self.validate_model()
        return self.model.predict_proba(X)

class SVM(ClassificationModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = SVC(
            kernel=config.model.SVM_KERNEL,
            gamma='scale',
            C=config.model.SVM_C,
            random_state=config.data.RANDOM_SEED,
            class_weight='balanced',
            probability=True
        )
    
    def fit(self, X, y):
        self.validate_model()
        return self.model.fit(X, y)
    
    def predict(self, X):
        self.validate_model()
        return self.model.predict(X)
    
    def get_probabilities(self, X):
        self.validate_model()
        return self.model.predict_proba(X)

# Regression Models
class LinearReg(RegressionModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LinearRegression()
    
    def fit(self, X, y):
        self.validate_model()
        return self.model.fit(X, y)
    
    def predict(self, X):
        self.validate_model()
        return self.model.predict(X)

class RandomForestReg(RegressionModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = RandomForestRegressor(
            n_estimators=config.model.RF_N_ESTIMATORS,
            max_depth=config.model.RF_MAX_DEPTH,
            random_state=config.data.RANDOM_SEED
        )
    
    def fit(self, X, y):
        self.validate_model()
        return self.model.fit(X, y)
    
    def predict(self, X):
        self.validate_model()
        return self.model.predict(X)

class XGBoostReg(RegressionModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = xgb.XGBRegressor(
            n_estimators=config.model.XGB_N_ESTIMATORS,
            max_depth=config.model.XGB_MAX_DEPTH,
            learning_rate=config.model.XGB_LEARNING_RATE,
            random_state=config.data.RANDOM_SEED
        )
    
    def fit(self, X, y):
        self.validate_model()
        return self.model.fit(X, y)
    
    def predict(self, X):
        self.validate_model()
        return self.model.predict(X)

# Factory for creating models
class ModelFactory:
    """Factory class for creating model instances"""
    @staticmethod
    def create_model(model_type: str, config) -> BaseModel:
        """Create a model instance based on the specified type"""
        models = {
            'nn': NeuralNetwork,
            'lda': LDA,
            'svm': SVM,
            'linear': LinearReg,
            'rf': RandomForestReg,
            'xgb': XGBoostReg
        }
        
        if model_type not in models:
            raise ValueError(f"Unknown model type: {model_type}")
            
        return models[model_type](config)