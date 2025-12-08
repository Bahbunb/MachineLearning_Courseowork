from config import Config
import warnings
from sklearn.exceptions import ConvergenceWarning
from experiment_runner import ExperimentRunner
import argparse
import sys
from logger import ModelLogger

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run machine learning experiments')
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['nn', 'svm', 'lda', 'linear', 'rf', 'xgb', 'all'],
        default='all',
        help='Model type to run (default: all)'
    )
    parser.add_argument(
        '--feature-method', 
        type=str,
        choices=['none', 'pca', 'kmeans', 'ttest', 'stepwise', 'correlation', 'lasso', 'all'],
        default='all',
        help='Feature selection method (default: all)'
    )
    parser.add_argument(
        '--log-transform',
        action='store_true',
        help='Apply log transformation to target variable'
    )
    return parser.parse_args()

def setup_environment():
    """Setup environment configurations"""
    # Suppress convergence warnings
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    
    # Initialize logger
    logger = ModelLogger()
    return logger

def run_experiments(config, args, logger):
    """Run experiments based on command line arguments"""
    runner = ExperimentRunner(config)
    
    try:
        # Determine available models and methods based on task type
        if config.data.TASK_TYPE == 'classification':
            available_models = ['nn', 'svm', 'lda']
            available_methods = config.feature_selection.METHODS_FOR_TASK['classification']
        else:  # regression
            available_models = ['linear', 'rf', 'xgb']
            available_methods = config.feature_selection.METHODS_FOR_TASK['regression']
        
        # Case 1: Single model with all feature methods
        if args.model != 'all' and args.feature_method == 'all':
            logger.log_info(f"Running {args.model} with all feature selection methods")
            for feature_method in available_methods:
                logger.log_info(f"\nRunning combination: {args.model} with {feature_method}")
                runner.run_single_experiment(args.model, feature_method)
        
        # Case 2: All models with single feature method
        elif args.model == 'all' and args.feature_method != 'all':
            logger.log_info(f"Running all models with {args.feature_method}")
            for model_type in available_models:
                logger.log_info(f"\nRunning combination: {model_type} with {args.feature_method}")
                runner.run_single_experiment(model_type, args.feature_method)
        
        # Case 3: All models with all feature methods
        elif args.model == 'all' and args.feature_method == 'all':
            logger.log_info("Running all experiment combinations")
            runner.run_all_experiments()
        
        # Case 4: Single model with single feature method
        else:
            logger.log_info(f"Running single experiment: {args.model} with {args.feature_method}")
            runner.run_single_experiment(args.model, args.feature_method)
            
    except Exception as e:
        logger.log_error(f"Error running experiments: {str(e)}")
        sys.exit(1)

def main():
    """Main entry point of the application"""
    args = parse_arguments()
    
    # Setup environment
    logger = setup_environment()
    logger.log_info("Starting experiment runner")
    
    try:
        # Initialize config with log transform setting
        config = Config()
        config.regression.LOG_TRANSFORM_TARGET = args.log_transform
        logger.log_info("Configuration loaded successfully")
        logger.log_info(f"Log transform: {'enabled' if args.log_transform else 'disabled'}")
        
        # Run experiments
        run_experiments(config, args, logger)
        
        logger.log_info("Experiments completed successfully")
        
    except Exception as e:
        logger.log_error(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()