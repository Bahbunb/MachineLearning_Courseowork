from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score
import numpy as np
from data_preprocessing import DataPreprocessor
from feature_selection import FeatureSelector
from visualization import DataVisualizer
from model import ModelFactory
from logger import ModelLogger
from sklearn.cluster import KMeans
from joblib import Parallel, delayed

class ExperimentStrategy(ABC):
    """Abstract base class for experiment strategies"""
    @abstractmethod
    def run_experiment(self, X, y):
        """Run a single experiment"""
        pass

class SingleExperimentStrategy(ExperimentStrategy):
    """Strategy for running a single experiment"""
    def __init__(self, config, model_type, feature_method):
        self.config = config
        self.model_type = model_type
        self.feature_method = feature_method
        self.preprocessor = DataPreprocessor(config)
        self.feature_selector = FeatureSelector(config)
        self.logger = ModelLogger()

    def find_optimal_clusters(self, X, y):
        """Find optimal number of clusters for KMeans feature selection"""
        best_score = -1
        best_n_clusters = 2
        
        # Initialize model once outside the loop
        model = ModelFactory.create_model(self.model_type, self.config)
        
        # Pre-compute X shape
        n_samples = X.shape[0]
        
        # Use numpy array for faster computation
        X_array = X.values if hasattr(X, 'values') else X
        
        # Try different numbers of clusters in parallel
        def evaluate_n_clusters(n_clusters):
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=self.config.RANDOM_SEED,
                n_init=10  # Reduce number of initializations
            )
            kmeans.fit(X_array)
            
            # Vectorized distance calculation
            distances = np.zeros((n_samples, n_clusters))
            for i in range(n_clusters):
                distances[:, i] = np.linalg.norm(X_array - kmeans.cluster_centers_[i], axis=1)
            
            scores = cross_val_score(
                model.model, 
                distances,
                y, 
                cv=self.config.N_SPLITS,
                scoring='precision',
                n_jobs=-1  # Use all available cores
            )
            
            return n_clusters, np.mean(scores)
        
        # Parallel execution of cluster evaluation
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_n_clusters)(n_clusters) 
            for n_clusters in range(2, min(11, n_samples))
        )
        
        # Find best number of clusters
        for n_clusters, score in results:
            if score > best_score:
                best_score = score
                best_n_clusters = n_clusters
        
        return best_n_clusters

    def run_experiment(self, X, y):
        """Run a single experiment with specified parameters"""
        self.logger.log_info(f"\nStarting experiment: {self.model_type} with {self.feature_method}")
        self.logger.log_info("=" * 50)
        
        # Get cross-validation splits
        cv_splits = self.preprocessor.get_stratified_folds(X, y, self.config.N_SPLITS)
        
        metrics = []
        n_features_list = []
        
        # Store the last trained model and data for visualization
        last_model = None
        last_X_train = None
        last_y_train = None
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, 1):
            self.logger.log_info(f"\nFold {fold_idx}/{self.config.N_SPLITS}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Scale features
            X_train_scaled = self.preprocessor.fit_scale_features(X_train)
            X_test_scaled = self.preprocessor.transform_features(X_test)
            
            # Apply SMOTE if classification
            if self.config.data.TASK_TYPE == 'classification':
                X_train_balanced, y_train_balanced = self.preprocessor.apply_oversampling(
                    X_train_scaled, y_train
                )
            else:
                X_train_balanced, y_train_balanced = X_train_scaled, y_train
            
            # Select features
            X_train_selected, X_test_selected = self._select_features(
                X_train_balanced, y_train_balanced, X_test_scaled
            )
            
            # Track number of features
            n_features_list.append(X_train_selected.shape[1])
            
            # Train and evaluate model
            model = ModelFactory.create_model(self.model_type, self.config)
            model.fit(X_train_selected, y_train_balanced)
            y_pred = model.predict(X_test_selected)
            fold_metrics = model.get_detailed_metrics(y_test, y_pred)
            metrics.append(fold_metrics)
            
            # Store last fold's model and data for visualization
            if fold_idx == self.config.N_SPLITS:
                last_model = model
                last_X_train = X_train_selected
                last_y_train = y_train_balanced
            
            # Log fold metrics
            self._log_fold_metrics(fold_idx, fold_metrics)
        
        # Generate visualizations using the last fold's model and data
        if last_model is not None and self.config.data.TASK_TYPE == 'regression':
            self.logger.log_info("\nGenerating visualizations...")
            
            # Plot predictions vs actual
            last_model.plot_predictions(last_X_train, last_y_train)
            
            # Plot feature importance
            if self.model_type in ['linear', 'rf', 'xgb']:
                last_model.plot_feature_importance(last_X_train)
            
            self.logger.log_info("Visualizations have been saved to the 'plots' directory")
        
        return self._calculate_final_metrics(metrics, n_features_list, X.shape[1])

    def _select_features(self, X_train, y_train, X_test):
        """Select features using specified method"""
        if self.feature_method == 'none':
            return X_train, X_test
        elif self.feature_method == 'kmeans':
            # First set the strategy to KMeans
            self.feature_selector.set_strategy('kmeans')
            # Find optimal number of clusters
            n_clusters = self.find_optimal_clusters(X_train, y_train)
            # Set the number of clusters
            self.feature_selector.strategy.set_n_clusters(n_clusters)
            # Now perform feature selection
            X_train_selected = self.feature_selector.select_features(
                X_train, y_train, method='kmeans'
            )
            X_test_selected = self.feature_selector.transform(X_test)
        else:
            X_train_selected = self.feature_selector.select_features(
                X_train, y_train, method=self.feature_method
            )
            X_test_selected = self.feature_selector.transform(X_test)
        
        return X_train_selected, X_test_selected

    def _train_and_evaluate(self, X_train, y_train, X_test, y_test):
        """Train and evaluate model"""
        model = ModelFactory.create_model(self.model_type, self.config)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return model.get_detailed_metrics(y_test, y_pred)

    def _log_fold_metrics(self, fold_idx, metrics):
        """Log metrics for current fold"""
        self.logger.log_info(f"Performance Metrics (Fold {fold_idx})")
        if self.config.data.TASK_TYPE == 'classification':
            self.logger.log_metrics({
                'accuracy': metrics[0],
                'type1_error': metrics[1],
                'type2_error': metrics[2]
            })
        else:  # regression
            self.logger.log_metrics({
                'rmse': metrics[0],
                'r2': metrics[1]
            })

    def _calculate_final_metrics(self, metrics, n_features_list, original_n_features):
        """Calculate and log final metrics"""
        metrics = np.array(metrics)
        avg_metrics = np.mean(metrics, axis=0)
        std_metrics = np.std(metrics, axis=0)
        
        avg_n_features = np.mean(n_features_list)
        std_n_features = np.std(n_features_list)
        avg_reduction_rate = (original_n_features - avg_n_features) / original_n_features * 100
        
        self._log_final_metrics(
            avg_metrics, std_metrics,
            avg_n_features, std_n_features,
            original_n_features, avg_reduction_rate
        )
        
        return avg_metrics

    def _log_final_metrics(self, avg_metrics, std_metrics, 
                          avg_n_features, std_n_features,
                          original_n_features, avg_reduction_rate):
        """Log final experiment metrics"""
        self.logger.log_info("\nFinal Results:")
        self.logger.log_info("-" * 50)
        self.logger.log_info(f"Number of folds: {self.config.N_SPLITS}")
        self.logger.log_info(f"Original number of features: {original_n_features}")
        self.logger.log_info(f"Average number of selected features: {avg_n_features:.1f} (±{std_n_features:.1f})")
        self.logger.log_info(f"Average feature reduction rate: {avg_reduction_rate:.1f}%")
        
        if self.config.data.TASK_TYPE == 'classification':
            self.logger.log_info(f"Average Accuracy: {avg_metrics[0]:.4f} (±{std_metrics[0]:.4f})")
            self.logger.log_info(f"Average Type I Error: {avg_metrics[1]:.4f} (±{std_metrics[1]:.4f})")
            self.logger.log_info(f"Average Type II Error: {avg_metrics[2]:.4f} (±{std_metrics[2]:.4f})")
        else:  # regression
            if len(avg_metrics) > 3:  # log transform was applied
                self.logger.log_info(f"Average Log-space RMSE: {avg_metrics[0]:.4f} (±{std_metrics[0]:.4f})")
                self.logger.log_info(f"Average R²: {avg_metrics[1]:.4f} (±{std_metrics[1]:.4f})")
                self.logger.log_info(f"Average Log Error: {avg_metrics[2]:.4f} (±{std_metrics[2]:.4f})")
                self.logger.log_info(f"Average Approximate RMSE: {avg_metrics[3]:.4f} (±{std_metrics[3]:.4f})")
            else:
                self.logger.log_info(f"Average RMSE: {avg_metrics[0]:.4f} (±{std_metrics[0]:.4f})")
                self.logger.log_info(f"Average R²: {avg_metrics[1]:.4f} (±{std_metrics[1]:.4f})")

class ExperimentRunner:
    """Main class for running experiments"""
    def __init__(self, config):
        self.config = config
        self.preprocessor = DataPreprocessor(config)
        self.visualizer = DataVisualizer(output_dir="plots")
        self.results = {}
        self.logger = ModelLogger()


    def run_single_experiment(self, model_type, feature_method):
        """Run a single experiment"""
        # Load data
        X, y = self.preprocessor.load_data(
            self.config.DATA_PATH,
            self.config.TARGET_COLUMN,
            self.config.DROP_COLUMNS
        )
        
        # Plot target variable skewness (for regression tasks)
        if self.config.data.TASK_TYPE == 'regression':
            self.logger.log_info("Generating target variable distribution plot...")
            self.visualizer.plot_target_skewness(y, title="Sale Price Distribution")
        
        # Create and run experiment strategy
        strategy = SingleExperimentStrategy(self.config, model_type, feature_method)
        return strategy.run_experiment(X, y)

    def run_all_experiments(self):
        """Run all experiment combinations"""
        model_types = ['nn', 'lda', 'svm']
        feature_methods = self.config.feature_selection.METHODS_FOR_TASK[self.config.data.TASK_TYPE]
        
        for model_type in model_types:
            self.results[model_type] = {}
            for feature_method in feature_methods:
                print(f"\nRunning experiment: {model_type} with {feature_method}")
                metrics = self.run_single_experiment(model_type, feature_method)
                self.results[model_type][feature_method] = metrics
        
        self._visualize_results()

    def _visualize_results(self):
        """Visualize experiment results"""
        for model_type, results in self.results.items():
            methods = list(results.keys())
            accuracies = [results[m][0] for m in methods]
            type1_errors = [results[m][1] for m in methods]
            type2_errors = [results[m][2] for m in methods]
            
            self.visualizer.plot_performance_metrics(
                methods,
                accuracies,
                type1_errors,
                type2_errors
            )
