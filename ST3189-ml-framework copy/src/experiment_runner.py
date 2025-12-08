"""
Experiment runner for ML framework.

Orchestrates the execution of machine learning experiments with support for:
- Single experiment runs with specified model and feature selection method
- Full experiment suites across all model/feature method combinations
- Cross-validation with performance metrics calculation
- Automatic visualization generation
"""

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
        best_score = -np.inf
        best_n_clusters = 2
        
        # Initialize model once outside the loop
        model = ModelFactory.create_model(self.model_type, self.config)
        
        # Pre-compute X shape
        n_samples = X.shape[0]
        
        # Use numpy array for faster computation
        X_array = X.values if hasattr(X, 'values') else X
        
        # Determine scoring based on task type
        if self.config.data.TASK_TYPE == 'classification':
            scoring = 'precision'
        else:
            scoring = 'neg_mean_squared_error'
        
        # Try different numbers of clusters
        def evaluate_n_clusters(n_clusters):
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=self.config.RANDOM_SEED,
                n_init=10
            )
            kmeans.fit(X_array)
            
            # Vectorized distance calculation
            distances = np.zeros((n_samples, n_clusters))
            for i in range(n_clusters):
                distances[:, i] = np.linalg.norm(
                    X_array - kmeans.cluster_centers_[i], axis=1
                )
            
            scores = cross_val_score(
                model.model, 
                distances,
                y, 
                cv=min(self.config.N_SPLITS, n_samples),
                scoring=scoring,
                n_jobs=-1
            )
            
            return n_clusters, np.mean(scores)
        
        # Parallel execution of cluster evaluation
        max_clusters = min(11, n_samples)
        results = Parallel(n_jobs=-1)(
            delayed(evaluate_n_clusters)(n_clusters) 
            for n_clusters in range(2, max_clusters)
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
        cv_splits = list(self.preprocessor.get_stratified_folds(X, y, self.config.N_SPLITS))
        
        metrics = []
        n_features_list = []
        
        # Store the last trained model and data for visualization
        last_model = None
        last_X_train = None
        last_y_train = None
        last_X_test = None
        last_y_test = None
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splits, 1):
            self.logger.log_info(f"\nFold {fold_idx}/{self.config.N_SPLITS}")
            
            # Split data
            X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
            y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
            
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
            n_features_list.append(
                X_train_selected.shape[1] if hasattr(X_train_selected, 'shape') 
                else X_train_selected.shape[1]
            )
            
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
                last_X_test = X_test_selected
                last_y_test = y_test
            
            # Log fold metrics
            self._log_fold_metrics(fold_idx, fold_metrics)
        
        # Generate visualizations using the last fold's model and data
        if last_model is not None:
            self._generate_visualizations(
                last_model, last_X_train, last_y_train, 
                last_X_test, last_y_test
            )
        
        return self._calculate_final_metrics(metrics, n_features_list, X.shape[1])

    def _generate_visualizations(self, model, X_train, y_train, X_test, y_test):
        """Generate all visualizations for the experiment"""
        self.logger.log_info("\nGenerating visualizations...")
        
        if self.config.data.TASK_TYPE == 'regression':
            # Plot predictions vs actual
            try:
                model.plot_predictions(X_test, y_test)
                self.logger.log_info("  ✓ Prediction vs Actual plot saved")
            except Exception as e:
                self.logger.log_info(f"  ✗ Prediction plot failed: {str(e)}")
            
            # Plot feature importance
            if self.model_type in ['linear', 'rf', 'xgb']:
                try:
                    # Get feature names
                    if hasattr(X_train, 'columns'):
                        feature_names = X_train.columns.tolist()
                    else:
                        feature_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
                    
                    model.plot_feature_importance(X_train, feature_names)
                    self.logger.log_info("  ✓ Feature importance plot saved")
                except Exception as e:
                    self.logger.log_info(f"  ✗ Feature importance plot failed: {str(e)}")
        
        self.logger.log_info("Visualizations saved to 'plots/' directory")

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
            if len(metrics) >= 4:  # log transform was applied
                self.logger.log_metrics({
                    'log_rmse': metrics[0],
                    'r2': metrics[1],
                    'log_mae': metrics[2],
                    'approx_rmse': metrics[3]
                })
            else:
                self.logger.log_metrics({
                    'rmse': metrics[0],
                    'r2': metrics[1],
                    'mae': metrics[2] if len(metrics) > 2 else 0
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
        self.logger.log_info("\n" + "=" * 50)
        self.logger.log_info("FINAL RESULTS")
        self.logger.log_info("=" * 50)
        self.logger.log_info(f"Number of folds: {self.config.N_SPLITS}")
        self.logger.log_info(f"Original number of features: {original_n_features}")
        self.logger.log_info(f"Average selected features: {avg_n_features:.1f} (±{std_n_features:.1f})")
        self.logger.log_info(f"Feature reduction rate: {avg_reduction_rate:.1f}%")
        self.logger.log_info("-" * 50)
        
        if self.config.data.TASK_TYPE == 'classification':
            self.logger.log_info(f"Average Accuracy: {avg_metrics[0]*100:.2f}% (±{std_metrics[0]*100:.2f}%)")
            self.logger.log_info(f"Average Type I Error: {avg_metrics[1]*100:.2f}% (±{std_metrics[1]*100:.2f}%)")
            self.logger.log_info(f"Average Type II Error: {avg_metrics[2]*100:.2f}% (±{std_metrics[2]*100:.2f}%)")
        else:  # regression
            if len(avg_metrics) >= 4:  # log transform was applied
                self.logger.log_info(f"Average Log-RMSE: {avg_metrics[0]:.4f} (±{std_metrics[0]:.4f})")
                self.logger.log_info(f"Average R²: {avg_metrics[1]*100:.2f}% (±{std_metrics[1]*100:.2f}%)")
                self.logger.log_info(f"Average Log-MAE: {avg_metrics[2]:.4f} (±{std_metrics[2]:.4f})")
                self.logger.log_info(f"Average Approx RMSE: ${avg_metrics[3]:,.2f} (±${std_metrics[3]:,.2f})")
            else:
                self.logger.log_info(f"Average RMSE: ${avg_metrics[0]:,.2f} (±${std_metrics[0]:,.2f})")
                self.logger.log_info(f"Average R²: {avg_metrics[1]*100:.2f}% (±{std_metrics[1]*100:.2f}%)")


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
        
        # Plot target variable distribution
        if self.config.data.TASK_TYPE == 'regression':
            self.logger.log_info("Generating target variable distribution plot...")
            self.visualizer.plot_target_skewness(y, title="Sale Price Distribution")
        else:
            self.logger.log_info("Generating class distribution plot...")
            self.visualizer.plot_class_distribution(y, title="Class Distribution")
        
        # Create and run experiment strategy
        strategy = SingleExperimentStrategy(self.config, model_type, feature_method)
        return strategy.run_experiment(X, y)

    def run_all_experiments(self):
        """Run all experiment combinations based on task type"""
        # Select models based on task type
        if self.config.data.TASK_TYPE == 'classification':
            model_types = ['lda', 'svm', 'nn']
        else:  # regression
            model_types = ['linear', 'rf', 'xgb']
        
        feature_methods = self.config.feature_selection.METHODS_FOR_TASK[self.config.data.TASK_TYPE]
        
        self.logger.log_info(f"\nTask type: {self.config.data.TASK_TYPE}")
        self.logger.log_info(f"Models to run: {model_types}")
        self.logger.log_info(f"Feature methods: {feature_methods}")
        self.logger.log_info("=" * 50)
        
        for model_type in model_types:
            self.results[model_type] = {}
            for feature_method in feature_methods:
                self.logger.log_info(f"\n>>> Running: {model_type.upper()} with {feature_method.upper()}")
                try:
                    metrics = self.run_single_experiment(model_type, feature_method)
                    self.results[model_type][feature_method] = metrics
                except Exception as e:
                    self.logger.log_error(f"Experiment failed: {str(e)}")
                    self.results[model_type][feature_method] = None
        
        # Print summary table
        self._print_summary_table()
        
        return self.results

    def _print_summary_table(self):
        """Print a summary table of all results"""
        self.logger.log_info("\n" + "=" * 70)
        self.logger.log_info("EXPERIMENT SUMMARY")
        self.logger.log_info("=" * 70)
        
        if self.config.data.TASK_TYPE == 'classification':
            self.logger.log_info(f"{'Model':<10} {'Method':<12} {'Accuracy':<12} {'Type I':<12} {'Type II':<12}")
            self.logger.log_info("-" * 70)
            for model_type, methods in self.results.items():
                for method, metrics in methods.items():
                    if metrics is not None:
                        self.logger.log_info(
                            f"{model_type:<10} {method:<12} "
                            f"{metrics[0]*100:>10.2f}% {metrics[1]*100:>10.2f}% {metrics[2]*100:>10.2f}%"
                        )
        else:  # regression
            self.logger.log_info(f"{'Model':<10} {'Method':<12} {'RMSE':<15} {'R²':<12}")
            self.logger.log_info("-" * 70)
            for model_type, methods in self.results.items():
                for method, metrics in methods.items():
                    if metrics is not None:
                        rmse = metrics[3] if len(metrics) >= 4 else metrics[0]
                        r2 = metrics[1]
                        self.logger.log_info(
                            f"{model_type:<10} {method:<12} ${rmse:>12,.2f} {r2*100:>10.2f}%"
                        )
        
        self.logger.log_info("=" * 70)

    def _visualize_results(self):
        """Visualize experiment results (for classification)"""
        if self.config.data.TASK_TYPE != 'classification':
            return
            
        for model_type, results in self.results.items():
            if not results:
                continue
                
            methods = list(results.keys())
            valid_results = {m: r for m, r in results.items() if r is not None}
            
            if not valid_results:
                continue
                
            methods = list(valid_results.keys())
            accuracies = [valid_results[m][0] for m in methods]
            type1_errors = [valid_results[m][1] for m in methods]
            type2_errors = [valid_results[m][2] for m in methods]
            
            self.visualizer.plot_performance_metrics(
                methods,
                accuracies,
                type1_errors,
                type2_errors
            )