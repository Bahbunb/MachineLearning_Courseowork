from abc import ABC, abstractmethod
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_selection import f_classif, RFECV
from sklearn.linear_model import LogisticRegression, Lasso
import numpy as np
from logger import FeatureSelectionLogger


class FeatureSelectionStrategy(ABC):
    """Abstract base class for feature selection strategies"""
    @abstractmethod
    def select_features(self, X, y):
        """Select features from input data"""
        pass

    @abstractmethod
    def transform(self, X):
        """Transform new data using selected features"""
        pass

class PCASelector(FeatureSelectionStrategy):
    def __init__(self, config):
        self.selector = PCA(
            n_components=config.feature_selection.PCA_VARIANCE_RATIO,
            random_state=config.data.RANDOM_SEED
        )

    def select_features(self, X, y):
        return self.selector.fit_transform(X)

    def transform(self, X):
        return self.selector.transform(X)

class KMeansSelector(FeatureSelectionStrategy):
    def __init__(self, config):
        self.n_clusters = config.feature_selection.KMEANS_MIN_CLUSTERS
        self.selector = KMeans(
            n_clusters=self.n_clusters,
            random_state=config.data.RANDOM_SEED,
            n_init=10  # Reduce number of initializations
        )
        self.random_seed = config.data.RANDOM_SEED
        self.cluster_centers_ = None

    def set_n_clusters(self, n_clusters):
        self.n_clusters = n_clusters
        self.selector = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_seed,
            n_init=10
        )

    def select_features(self, X, y):
        # Convert to numpy array for faster computation
        X_array = X.values if hasattr(X, 'values') else X
        
        self.selector.fit(X_array)
        self.cluster_centers_ = self.selector.cluster_centers_
        return self._calculate_distances(X_array)

    def transform(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Selector not fitted. Call select_features first.")
        X_array = X.values if hasattr(X, 'values') else X
        return self._calculate_distances(X_array)

    def _calculate_distances(self, X):
        # Vectorized distance calculation
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, self.n_clusters))
        
        # Use broadcasting for faster computation
        for i in range(self.n_clusters):
            distances[:, i] = np.sqrt(np.sum((X - self.cluster_centers_[i])**2, axis=1))
        
        return distances

class TTestSelector(FeatureSelectionStrategy):
    def __init__(self, config):
        self.significance = config.TTEST_SIGNIFICANCE
        self.significant_features = None

    def select_features(self, X, y):
        _, p_values = f_classif(X, y)
        self.significant_features = p_values < self.significance
        return X[:, self.significant_features]

    def transform(self, X):
        if self.significant_features is None:
            raise ValueError("Selector not fitted. Call select_features first.")
        return X[:, self.significant_features]

class StepwiseSelector(FeatureSelectionStrategy):
    def __init__(self, config):
        self.selector = RFECV(
            estimator=LogisticRegression(random_state=config.data.RANDOM_SEED),
            step=1,
            cv=config.feature_selection.STEPWISE_CV,
            scoring=config.feature_selection.STEPWISE_SCORING,
            min_features_to_select=config.feature_selection.STEPWISE_MIN_FEATURES
        )

    def select_features(self, X, y):
        return self.selector.fit_transform(X, y)

    def transform(self, X):
        return self.selector.transform(X)

class CorrelationSelector(FeatureSelectionStrategy):
    def __init__(self, config):
        self.threshold = config.feature_selection.CORRELATION_THRESHOLD
        self.selected_features = None

    def select_features(self, X, y):
        # Calculate correlation with target
        correlations = np.abs(np.corrcoef(X.T, y)[:-1, -1])
        # Select features above threshold
        self.selected_features = correlations >= self.threshold
        return X[:, self.selected_features]

    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("Selector not fitted. Call select_features first.")
        return X[:, self.selected_features]

class LassoSelector(FeatureSelectionStrategy):
    def __init__(self, config):
        self.selector = Lasso(
            alpha=config.feature_selection.LASSO_ALPHA,
            random_state=config.data.RANDOM_SEED
        )
        self.selected_features = None

    def select_features(self, X, y):
        self.selector.fit(X, y)
        self.selected_features = self.selector.coef_ != 0
        return X[:, self.selected_features]

    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("Selector not fitted. Call select_features first.")
        return X[:, self.selected_features]

class FeatureSelector:
    """Main feature selection class using Strategy pattern"""
    def __init__(self, config):
        if not hasattr(config, 'feature_selection'):
            raise ValueError("Config must have feature_selection attribute")
            
        self.config = config
        self.logger = FeatureSelectionLogger()
        self.strategy = None
        self._current_method = None
        
        # Validate and log initial parameters
        try:
            self.logger.log_parameters({
                "random_seed": self.config.data.RANDOM_SEED,
                "pca_variance_ratio": self.config.feature_selection.PCA_VARIANCE_RATIO,
                "ttest_significance": self.config.feature_selection.TTEST_SIGNIFICANCE,
                "stepwise_min_features": self.config.feature_selection.STEPWISE_MIN_FEATURES
            })
        except AttributeError as e:
            self.logger.log_error(f"Missing configuration parameter: {str(e)}")
            raise

    def set_strategy(self, method: str):
        """Set the feature selection strategy"""
        strategies = {
            'pca': PCASelector,
            'kmeans': KMeansSelector,
            'ttest': TTestSelector,
            'stepwise': StepwiseSelector,
            'correlation': CorrelationSelector,
            'lasso': LassoSelector
        }
        
        if method not in strategies:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        try:
            self.strategy = strategies[method](self.config)
            self._current_method = method
        except Exception as e:
            self.logger.log_error(f"Error initializing {method} strategy: {str(e)}")
            raise

    def select_features(self, X, y, method):
        """Select features using specified method"""
        try:
            # Convert to numpy array if it's a DataFrame
            X_array = X.values if hasattr(X, 'values') else np.array(X)
            
            # Set strategy if not already set or different method requested
            if self.strategy is None or method != self._current_method:
                self.set_strategy(method)
            
            # Perform feature selection
            X_selected = self.strategy.select_features(X_array, y)
            
            # Log results
            self.logger.log_results(method, X_array.shape[1], X_selected.shape[1])
            
            return X_selected
            
        except Exception as e:
            self.logger.log_error(f"Error in feature selection: {str(e)}")
            raise

    def transform(self, X):
        """Transform new data using the fitted selector"""
        if self.strategy is None:
            raise ValueError("No feature selection strategy set. Call select_features first.")
            
        X_array = X.values if hasattr(X, 'values') else np.array(X)
        return self.strategy.transform(X_array)

    def set_best_n_clusters(self, n_clusters):
        """Set the optimal number of clusters for KMeans strategy"""
        if not isinstance(self.strategy, KMeansSelector):
            raise ValueError("Current strategy is not KMeans")
        self.strategy.set_n_clusters(n_clusters)