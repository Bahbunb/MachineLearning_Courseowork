import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
import os

class DataVisualizer:
    def __init__(self, output_dir="plots"):
        """Initialize with output directory for plots"""
        self.output_dir = output_dir
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    @staticmethod
    def plot_class_distribution(y, title="Class Distribution"):
        plt.figure(figsize=(8, 6))
        sns.countplot(x=y)
        plt.title(title)
        plt.xlabel("Class")
        plt.ylabel("Count")
        plt.show()
    
    @staticmethod
    def plot_outliers(X, outliers):
        n_features = len(X.columns)
        n_rows = (n_features + 1) // 2  # Ceiling division to handle odd number of features
        
        plt.figure(figsize=(15, 5 * n_rows))
        for i, column in enumerate(X.columns):
            plt.subplot(n_rows, 2, i+1)
            
            # Plot non-outliers in blue and outliers in red
            normal_points = ~outliers
            plt.scatter(range(len(X[normal_points])), X[column][normal_points], 
                    c='blue', label='Normal', alpha=0.5)
            plt.scatter(range(len(X[outliers])), X[column][outliers], 
                    c='red', label='Outlier', alpha=0.7)
            
            plt.title(f'Outliers in {column}')
            plt.ylabel(column)
            plt.xlabel('Sample Index')
            plt.legend()
            
        plt.tight_layout(pad=3.0)  # Add more padding between subplots
        plt.show()

    @staticmethod
    def plot_performance_metrics(methods, accuracies, type1_errors, type2_errors):
        """
        Plot separate graphs for accuracy, type I and type II errors for each feature selection method.
        
        Parameters:
        - methods: List of method names
        - accuracies: List of accuracy scores
        - type1_errors: List of type I error rates
        - type2_errors: List of type II error rates
        """
        # Create figure with three subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Format x-axis labels
        x_labels = ['No Selection' if m[0] is None else m[0].upper() for m in methods]
        x_pos = np.arange(len(methods))
        
        # Plot Accuracy
        ax1.bar(x_pos, accuracies, color='blue', alpha=0.7)
        ax1.set_ylabel('Score')
        ax1.set_title('Accuracy')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(x_labels, rotation=45)
        
        # Plot Type I Error
        ax2.bar(x_pos, type1_errors, color='red', alpha=0.7)
        ax2.set_ylabel('Score')
        ax2.set_title('Type I Error')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(x_labels, rotation=45)
        
        # Plot Type II Error
        ax3.bar(x_pos, type2_errors, color='green', alpha=0.7)
        ax3.set_ylabel('Score')
        ax3.set_title('Type II Error')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(x_labels, rotation=45)
        
        # Add overall title
        plt.suptitle('Performance Metrics by Feature Selection Method', y=1.05)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()

    def plot_prediction_vs_actual(self, y_true, y_pred, title="Prediction vs Actual", log_scale=False):
        """Plot scatter plot of predicted vs actual values with a diagonal line"""
        plt.figure(figsize=(10, 8))
        
        # Create scatter plot
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add diagonal line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual Sale Price')
        plt.ylabel('Predicted Sale Price')
        plt.title(title)
        
        if log_scale:
            plt.xscale('log')
            plt.yscale('log')
        
        plt.tight_layout()
        # Save plot instead of showing
        plt.savefig(os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}.png"))
        plt.close()

    def plot_feature_importance(self, model, feature_names, title="Top 10 Important Features"):
        """Plot top 10 feature importances"""
        # Get feature importances
        importances = np.abs(model.coef_)
        
        # Create dataframe of features and their importances
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort and get top 10
        top_10 = feature_importance.sort_values('importance', ascending=True).tail(10)
        
        # Create horizontal bar plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(top_10)), top_10['importance'])
        plt.yticks(range(len(top_10)), top_10['feature'])
        plt.xlabel('Absolute Coefficient Value')
        plt.title(title)
        plt.tight_layout()
        # Save plot instead of showing
        plt.savefig(os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}.png"))
        plt.close()

    def plot_shap_values(self, shap_values, feature_names, title="SHAP Values"):
        """Create beeswarm plot of SHAP values"""
        plt.figure(figsize=(14, 10))
        
        # Create the beeswarm plot
        shap.plots.beeswarm(
            shap_values,
            max_display=10,  # Show top 10 features
            show=False,      # Don't display yet
            plot_size=(12, 8)
        )
        
        # Get current axis
        ax = plt.gca()
        
        # Format x-axis ticks to be more readable (in thousands/millions)
        def format_tick(x, pos):
            if abs(x) >= 1e6:
                return f'{x/1e6:.1f}M'
            elif abs(x) >= 1e3:
                return f'{x/1e3:.1f}K'
            else:
                return f'{x:.1f}'
        
        ax.xaxis.set_major_formatter(plt.FuncFormatter(format_tick))
        
        # Add gridlines for better readability
        ax.grid(True, axis='x', linestyle='--', alpha=0.6)
        
        # Adjust title and labels
        plt.title(title, pad=20, size=14, fontweight='bold')
        plt.xlabel("SHAP value (impact on price prediction)", size=12)
        
        # Adjust layout to prevent text cutoff
        plt.tight_layout()
        
        # Save plot with high DPI for better quality
        plt.savefig(
            os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()

    def plot_shap_force(self, explainer, shap_values, X, n_samples=2, title="SHAP Force Plot"):
        """
        Create force plots for random observations
        """
        # Convert to numpy array if needed
        if hasattr(X, 'values'):
            X = X.values
        
        # Randomly select observations
        np.random.seed(42)  # for reproducibility
        selected_indices = np.random.choice(X.shape[0], size=n_samples, replace=False)
        
        # Create force plots for each selected observation
        for i, idx in enumerate(selected_indices):
            # Create figure with extra height for title
            plt.figure(figsize=(20, 4))
            
            # Round all the values to 3 decimal places
            rounded_values = np.round(shap_values.values[idx:idx+1], 3)
            rounded_base = np.round(explainer.expected_value, 3)
            rounded_data = np.round(X[idx:idx+1], 3)
            
            # Create force plot using the Explanation object with rounded data
            shap_explanation = shap.Explanation(
                values=rounded_values,
                base_values=np.array([rounded_base]),
                data=rounded_data,
                feature_names=shap_values.feature_names
            )
            
            shap.plots.force(
                shap_explanation,
                show=False,
                matplotlib=True,
                plot_cmap=['#FF4B4B', '#4B4BFF']  # Red for positive, Blue for negative
            )

            # Save plot with high DPI for better quality
            plt.savefig(
                os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}_sample_{i+1}.png"),
                dpi=300,
                bbox_inches='tight'
            )
            plt.close()

    def plot_target_skewness(self, y, title="Target Variable Distribution"):
        """
        Plot a simple histogram with KDE of the target variable
        """
        # Calculate skewness
        skewness = y.skew()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot histogram with KDE and get the axes
        ax = sns.histplot(y, kde=True, color='blue', alpha=0.6)
        
        # Change the KDE line color to red
        ax.lines[0].set_color('crimson')
        ax.lines[0].set_linewidth(2)  # Make the line thicker
        
        # Add title and labels
        plt.title(f'{title} (Skewness: {skewness:.3f})', fontsize=14)
        plt.xlabel('Value', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        
        # Add grid for better readability
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig(
            os.path.join(self.output_dir, f"{title.lower().replace(' ', '_')}.png"),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()