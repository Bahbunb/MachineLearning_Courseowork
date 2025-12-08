# Machine Learning Framework for Classification and Regression

This repository contains a comprehensive machine learning framework designed for both classification and regression tasks. The framework implements various feature selection methods, preprocessing techniques, and model architectures with a focus on reproducibility and extensibility.

## Overview

The framework is built using a modular architecture with the following key components:

- **Configuration Management**: Centralized configuration system for all parameters
- **Data Preprocessing**: Robust data loading, validation, scaling, and outlier detection
- **Feature Selection**: Multiple feature selection strategies including PCA, KMeans, and statistical methods
- **Model Training**: Support for various classification and regression models
- **Experiment Running**: Structured experiment execution with cross-validation
- **Visualization**: Comprehensive visualization tools for model performance and feature importance
- **Logging**: Detailed logging system for tracking experiments

## Project Structure

```
├── src/
│ ├── config.py # Configuration management
│ ├── data_preprocessing.py # Data preprocessing pipeline
│ ├── feature_selection.py # Feature selection strategies
│ ├── model.py # Model implementations
│ ├── experiment_runner.py # Experiment execution
│ ├── visualization.py # Visualization utilities
│ ├── logger.py # Logging system
│ └── main.py # Main entry point
├── logs/ # Log files directory
└── plots/ # Generated plots directory 
```  
            
## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

This will install all the necessary packages.

## Configuration

The framework uses a hierarchical configuration system defined in `config.py`. The main configuration components include:

- **DataConfig**: Data paths and basic parameters
- **ClassificationConfig**: Classification-specific settings
- **RegressionConfig**: Regression-specific settings
- **FeatureSelectionConfig**: Feature selection parameters
- **ModelConfig**: Model hyperparameters

## Data Preprocessing

The preprocessing pipeline (`data_preprocessing.py`) handles:

- Data loading and validation
- Outlier detection using z-scores
- Feature scaling with StandardScaler
- Cross-validation splitting (stratified for classification)
- Class imbalance handling with SMOTE (for classification)
- Log transformation of target variables (for regression)

## Feature Selection Methods

The framework implements multiple feature selection strategies (`feature_selection.py`) using the Strategy pattern:

| Method | Description | Applicable Tasks |
|--------|-------------|------------------|
| PCA | Principal Component Analysis | Both |
| KMeans | K-Means clustering distances as features | Both |
| Correlation | Feature selection based on correlation with target | Both |
| T-Test | Statistical significance testing | Classification only |
| Stepwise | Recursive feature elimination with cross-validation | Classification only |
| Lasso | L1 regularization for feature selection | Regression only |

## Models

The framework supports various models (`model.py`) for both classification and regression tasks:

### Classification Models
- Neural Network (MLP)
- Linear Discriminant Analysis (LDA)
- Support Vector Machine (SVM)

### Regression Models
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

All models implement a common interface with `fit()`, `predict()`, and `get_detailed_metrics()` methods.

## Experiment Runner

The `experiment_runner.py` module orchestrates the execution of experiments:

- Single experiment execution with specified model and feature selection method
- Cross-validation with performance metrics calculation
- Automatic optimization of KMeans clusters
- Visualization of results

## Visualization

The `visualization.py` module provides tools for:

- Target variable distribution analysis
- Prediction vs. actual value plots
- Feature importance visualization
- SHAP value plots for model interpretability
- Performance metrics comparison across methods

## Logging

The framework includes a comprehensive logging system (`logger.py`) with specialized loggers for:

- Feature selection operations
- Model training and evaluation
- Data preprocessing steps

Logs are stored in the `logs/` directory with timestamps.

## Usage

## Reproducing Results

To run a single experiment:

1. Install the required dependencies
2. Configure the data paths in `config.py`
3. Run the desired experiment using the command line interface

Example for classification task:
```bash
python src/main.py --model svm --feature-method pca
```

Example for regression task:
```bash
python src/main.py --model linear --feature-method lasso --log-transform
```

## Performance Metrics

### Classification Metrics
- Accuracy
- Type I Error (False Positive Rate)
- Type II Error (False Negative Rate)

### Regression Metrics
- Root Mean Squared Error (RMSE)
- R² Score
- Mean Absolute Error (MAE)
- Log-space metrics (when log transformation is applied)

## Feature Selection Effectiveness

The framework automatically calculates and reports:
- Original number of features
- Number of selected features after feature selection
- Feature reduction rate (percentage)

## Extensibility

The framework is designed to be easily extended:

- Add new feature selection methods by implementing the `FeatureSelectionStrategy` interface
- Add new models by extending the `BaseModel` class
- Add new preprocessing steps by extending the `DataPreprocessor` class

## Limitations and Future Work

- Currently supports tabular data only
- No support for time series data
- Limited hyperparameter tuning capabilities
- Future work could include:
  - Automated hyperparameter optimization
  - Support for deep learning models
  - Integration with MLflow for experiment tracking
  - Support for categorical feature encoding

## Conclusion

This framework provides a comprehensive solution for machine learning experiments on classification and regression tasks, with a focus on feature selection and model comparison. The modular design allows for easy extension and customization for specific use cases.   
            