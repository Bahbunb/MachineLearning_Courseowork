"""
Post-ML Analysis for Taiwan Bankruptcy Prediction
=================================================

This script provides rigorous post-modeling diagnostics that are MISSING
from your report:

1. ROC and Precision-Recall curves (critical for imbalanced data)
2. Confusion matrix visualization
3. Cross-validation score distributions
4. Error analysis by feature characteristics
5. Model calibration assessment
6. Statistical significance testing of model differences

Your report only shows aggregate metrics. This is insufficient for
academic work - you need to show:
- Model stability across folds
- Whether performance differences are statistically significant
- Where models fail (error analysis)
- Whether models are well-calibrated
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score, roc_auc_score,
    make_scorer, cohen_kappa_score, matthews_corrcoef, 
    brier_score_loss, log_loss
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


class PostMLAnalysis:
    """
    Comprehensive post-modeling analysis for bankruptcy prediction.
    
    This class addresses critical gaps in your results section:
    - Your report shows only mean accuracy, Type I, Type II errors
    - NO assessment of model stability across folds
    - NO ROC/PR curves (essential for imbalanced data)
    - NO statistical testing of model differences
    - NO error analysis to understand WHERE models fail
    """
    
    def __init__(self, X, y, random_state=24, output_dir='plots'):
        """Initialize with preprocessed data."""
        self.X = X
        self.y = y
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
        # Setup output directory
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"\n✓ Created output directory: {self.output_dir}")
        else:
            print(f"\n✓ Using output directory: {self.output_dir}")
        
        # Initialize models with YOUR parameters
        self.models = {
            'LDA': LinearDiscriminantAnalysis(solver='svd'),
            'SVM': SVC(
                kernel='rbf',
                gamma='scale',
                C=75,
                random_state=random_state,
                class_weight='balanced',
                probability=True
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(64,),
                max_iter=50,
                random_state=random_state,
                early_stopping=True
            )
        }
        
        print("="*80)
        print("POST-ML ANALYSIS INITIALIZED")
        print("="*80)
        print(f"Data shape: {X.shape}")
        print(f"Class distribution: {np.bincount(y)}")
        print(f"Models: {list(self.models.keys())}")
        print(f"Cross-validation: {5}-fold stratified")
    
    def cross_validation_detailed(self):
        """
        CRITICAL: Detailed cross-validation with multiple metrics.
        
        Your report shows only MEAN performance. This hides:
        1. Performance variability across folds
        2. Whether models are stable
        3. Statistical significance of differences
        """
        print("\n" + "="*80)
        print("DETAILED CROSS-VALIDATION ANALYSIS")
        print("="*80)
        
        # Define custom scoring functions
        def type1_error(y_true, y_pred):
            """Type I error: False Positive Rate"""
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return fp / (fp + tn) if (fp + tn) > 0 else 0
        
        def type2_error(y_true, y_pred):
            """Type II error: False Negative Rate"""
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            return fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Scoring dictionary
        scoring = {
            'accuracy': 'accuracy',
            'precision': 'precision',
            'recall': 'recall',
            'f1': 'f1',
            'roc_auc': 'roc_auc',
            'type1_error': make_scorer(type1_error, greater_is_better=False),
            'type2_error': make_scorer(type2_error, greater_is_better=False)
        }
        
        cv_results = {}
        
        for name, model in self.models.items():
            print(f"\n{'-'*80}")
            print(f"Model: {name}")
            print(f"{'-'*80}")
            
            # Perform cross-validation
            scores = cross_validate(
                model, self.X, self.y,
                cv=self.cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            cv_results[name] = scores
            
            # Print results
            print("\nTest Set Performance:")
            for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 
                          'type1_error', 'type2_error']:
                test_scores = scores[f'test_{metric}']
                
                # Handle negative scoring
                if metric in ['type1_error', 'type2_error']:
                    test_scores = -test_scores
                
                mean_score = test_scores.mean()
                std_score = test_scores.std()
                
                print(f"  {metric:15s}: {mean_score:.4f} (±{std_score:.4f})")
            
            # Train-test gap (overfitting check)
            print("\nOverfitting Assessment (Train - Test gap):")
            train_acc = scores['train_accuracy'].mean()
            test_acc = scores['test_accuracy'].mean()
            gap = train_acc - test_acc
            
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy:  {test_acc:.4f}")
            print(f"  Gap:            {gap:.4f}")
            
            if gap > 0.05:
                print(f"  ⚠️  WARNING: Overfitting detected (gap > 5%)")
            else:
                print(f"  ✓ No significant overfitting")
        
        # Statistical comparison of models
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("="*80)
        print("\nCritical Question: Are performance differences statistically significant?")
        print("Your report doesn't test this - you can't claim one model is 'better'")
        print("without statistical evidence.")
        
        # Extract accuracy scores for each model
        accuracy_scores = {
            name: cv_results[name]['test_accuracy']
            for name in self.models.keys()
        }
        
        # Pairwise Wilcoxon signed-rank tests
        print("\nPairwise Wilcoxon Signed-Rank Tests (Accuracy):")
        print("-" * 80)
        
        model_names = list(self.models.keys())
        for i in range(len(model_names)):
            for j in range(i+1, len(model_names)):
                model1 = model_names[i]
                model2 = model_names[j]
                
                scores1 = accuracy_scores[model1]
                scores2 = accuracy_scores[model2]
                
                statistic, p_value = wilcoxon(scores1, scores2)
                
                mean_diff = scores1.mean() - scores2.mean()
                
                print(f"\n{model1} vs {model2}:")
                print(f"  Mean difference: {mean_diff:.4f}")
                print(f"  P-value: {p_value:.4f}")
                
                if p_value < 0.05:
                    better = model1 if mean_diff > 0 else model2
                    print(f"  ✓ Statistically significant (p < 0.05)")
                    print(f"  → {better} is significantly better")
                else:
                    print(f"  ✗ NOT statistically significant")
                    print(f"  → Cannot claim one is better than the other")
        
        # Friedman test (non-parametric alternative to repeated measures ANOVA)
        print("\n" + "-"*80)
        print("Friedman Test (Overall model comparison):")
        print("-" * 80)
        
        scores_array = np.array([accuracy_scores[name] for name in model_names])
        statistic, p_value = friedmanchisquare(*scores_array)
        
        print(f"Friedman statistic: {statistic:.4f}")
        print(f"P-value: {p_value:.4f}")
        
        if p_value < 0.05:
            print("✓ Significant difference detected between models")
        else:
            print("✗ No significant difference between models")
        
        # Visualize cross-validation results
        self._plot_cv_results(cv_results)
        
        return cv_results
    
    def _plot_cv_results(self, cv_results):
        """Plot cross-validation results."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'type1_error']
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Prepare data for box plot
            data = []
            labels = []
            
            for name in self.models.keys():
                scores = cv_results[name][f'test_{metric}']
                
                # Handle negative scoring
                if metric in ['type1_error', 'type2_error']:
                    scores = -scores
                
                data.append(scores)
                labels.append(name)
            
            # Box plot
            bp = ax.boxplot(data, labels=labels, patch_artist=True,
                           showmeans=True, meanline=True)
            
            # Color boxes
            colors = ['lightblue', 'lightgreen', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            # Add individual points
            for i, scores in enumerate(data):
                y = scores
                x = np.random.normal(i+1, 0.04, size=len(y))
                ax.scatter(x, y, alpha=0.6, s=50, color='darkblue')
            
            # Formatting
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()} Across CV Folds', 
                        fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add mean line
            means = [np.mean(d) for d in data]
            ax.plot(range(1, len(means)+1), means, 'r--', linewidth=2, 
                   label='Mean', alpha=0.7)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/cv_detailed_results.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved: {self.output_dir}/cv_detailed_results.png")
    
    def roc_pr_curves(self):
        """
        ROC and Precision-Recall curves.
        
        CRITICAL for imbalanced data. Your report shows NONE of this.
        
        Why this matters:
        - ROC curves show trade-off between TPR and FPR
        - PR curves are more informative for imbalanced data
        - AUC summarizes overall discriminative ability
        """
        print("\n" + "="*80)
        print("ROC AND PRECISION-RECALL CURVE ANALYSIS")
        print("="*80)
        print("\nCritical Note: These plots are ESSENTIAL for imbalanced classification")
        print("but are MISSING from your report.")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Store results
        roc_results = {}
        pr_results = {}
        
        for name, model in self.models.items():
            print(f"\n{name}:")
            
            # Store ROC and PR data for each fold
            tprs = []
            fprs_interp = np.linspace(0, 1, 100)
            precisions = []
            recalls_interp = np.linspace(0, 1, 100)
            aucs_roc = []
            aucs_pr = []
            
            fold_idx = 0
            for train_idx, test_idx in self.cv.split(self.X, self.y):
                # Split data
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                # Train model
                model.fit(X_train, y_train)
                
                # Get probability predictions
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    y_proba = model.decision_function(X_test)
                
                # ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                aucs_roc.append(roc_auc)
                
                # Interpolate TPR
                tpr_interp = np.interp(fprs_interp, fpr, tpr)
                tpr_interp[0] = 0.0
                tprs.append(tpr_interp)
                
                # Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_proba)
                pr_auc = average_precision_score(y_test, y_proba)
                aucs_pr.append(pr_auc)
                
                # Interpolate precision
                precision_interp = np.interp(recalls_interp[::-1], recall[::-1], precision[::-1])
                precisions.append(precision_interp[::-1])
                
                fold_idx += 1
            
            # Calculate mean and std
            mean_tpr = np.mean(tprs, axis=0)
            std_tpr = np.std(tprs, axis=0)
            mean_auc_roc = np.mean(aucs_roc)
            std_auc_roc = np.std(aucs_roc)
            
            mean_precision = np.mean(precisions, axis=0)
            std_precision = np.std(precisions, axis=0)
            mean_auc_pr = np.mean(aucs_pr)
            std_auc_pr = np.std(aucs_pr)
            
            print(f"  ROC AUC: {mean_auc_roc:.4f} (±{std_auc_roc:.4f})")
            print(f"  PR AUC:  {mean_auc_pr:.4f} (±{std_auc_pr:.4f})")
            
            # Plot ROC curve
            ax1 = axes[0]
            ax1.plot(fprs_interp, mean_tpr, 
                    label=f'{name} (AUC = {mean_auc_roc:.3f} ± {std_auc_roc:.3f})',
                    linewidth=2)
            
            # Plot PR curve
            ax2 = axes[1]
            ax2.plot(recalls_interp, mean_precision,
                    label=f'{name} (AP = {mean_auc_pr:.3f} ± {std_auc_pr:.3f})',
                    linewidth=2)
            
            roc_results[name] = {
                'mean_auc': mean_auc_roc,
                'std_auc': std_auc_roc,
                'tpr': mean_tpr,
                'fpr': fprs_interp
            }
            
            pr_results[name] = {
                'mean_ap': mean_auc_pr,
                'std_ap': std_auc_pr,
                'precision': mean_precision,
                'recall': recalls_interp
            }
        
        # Format ROC plot
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        ax1.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax1.set_title('ROC Curves (Mean ± Std over CV folds)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(alpha=0.3)
        
        # Format PR plot
        baseline = self.y.sum() / len(self.y)  # Proportion of positive class
        ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1,
                   label=f'Random Classifier (baseline={baseline:.3f})')
        ax2.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax2.set_title('Precision-Recall Curves\n(More informative for imbalanced data)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/roc_pr_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved: {self.output_dir}/roc_pr_curves.png")
        
        return roc_results, pr_results
    
    def confusion_matrix_analysis(self):
        """
        Detailed confusion matrix analysis.
        
        Your report shows Type I and Type II errors but no confusion matrices.
        Visualization helps understand the error patterns.
        """
        print("\n" + "="*80)
        print("CONFUSION MATRIX ANALYSIS")
        print("="*80)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (name, model) in enumerate(self.models.items()):
            print(f"\n{name}:")
            
            # Aggregate confusion matrix across all folds
            cm_total = np.zeros((2, 2))
            
            for train_idx, test_idx in self.cv.split(self.X, self.y):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                cm = confusion_matrix(y_test, y_pred)
                cm_total += cm
            
            # Calculate metrics
            tn, fp, fn, tp = cm_total.ravel()
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            type1_error = fp / (fp + tn) if (fp + tn) > 0 else 0
            type2_error = fn / (fn + tp) if (fn + tp) > 0 else 0
            
            print(f"  True Negatives:  {int(tn):,}")
            print(f"  False Positives: {int(fp):,} (Type I Error: {type1_error*100:.2f}%)")
            print(f"  False Negatives: {int(fn):,} (Type II Error: {type2_error*100:.2f}%)")
            print(f"  True Positives:  {int(tp):,}")
            print(f"  Accuracy:  {accuracy*100:.2f}%")
            print(f"  Precision: {precision*100:.2f}%")
            print(f"  Recall:    {recall*100:.2f}%")
            print(f"  F1-Score:  {f1*100:.2f}%")
            
            # Plot confusion matrix
            ax = axes[idx]
            
            # Normalize for visualization
            cm_norm = cm_total / cm_total.sum(axis=1, keepdims=True)
            
            # Create heatmap
            sns.heatmap(cm_norm, annot=False, fmt='.2%', cmap='Blues',
                       ax=ax, cbar=True, square=True,
                       vmin=0, vmax=1)
            
            # Add count annotations
            for i in range(2):
                for j in range(2):
                    count = int(cm_total[i, j])
                    pct = cm_norm[i, j] * 100
                    ax.text(j + 0.5, i + 0.5,
                           f'{count:,}\n({pct:.1f}%)',
                           ha='center', va='center',
                           fontsize=12, fontweight='bold',
                           color='white' if cm_norm[i, j] > 0.5 else 'black')
            
            # Labels
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
            ax.set_title(f'{name}\nAccuracy: {accuracy*100:.2f}%', 
                        fontsize=12, fontweight='bold')
            ax.set_xticklabels(['Non-Bankrupt (0)', 'Bankrupt (1)'])
            ax.set_yticklabels(['Non-Bankrupt (0)', 'Bankrupt (1)'])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/confusion_matrices.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved: {self.output_dir}/confusion_matrices.png")
    
    def error_analysis(self):
        """
        Analyze WHERE models make errors.
        
        CRITICAL MISSING ANALYSIS: Your report doesn't explain WHY models fail.
        This section identifies patterns in misclassifications.
        """
        print("\n" + "="*80)
        print("ERROR ANALYSIS")
        print("="*80)
        print("\nCritical Question: WHERE do models fail?")
        print("Your report doesn't address this - it's essential for understanding")
        print("model limitations and improving future iterations.")
        
        # Train on full data for error analysis
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)
        
        error_data = {}
        
        for name, model in self.models.items():
            print(f"\n{'-'*80}")
            print(f"Model: {name}")
            print(f"{'-'*80}")
            
            # Use one fold for error analysis
            train_idx, test_idx = next(self.cv.split(X_scaled, self.y))
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = self.y[train_idx], self.y[test_idx]
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Get probability predictions
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.decision_function(X_test)
                # Normalize to [0, 1]
                y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
            
            # Identify errors
            errors = y_pred != y_test
            
            # Separate error types
            false_positives = (y_pred == 1) & (y_test == 0)
            false_negatives = (y_pred == 0) & (y_test == 1)
            
            print(f"\nError Breakdown:")
            print(f"  Total errors: {errors.sum()} ({errors.sum()/len(y_test)*100:.2f}%)")
            print(f"  False Positives (Type I): {false_positives.sum()}")
            print(f"  False Negatives (Type II): {false_negatives.sum()}")
            
            # Analyze confidence of errors
            print(f"\nPrediction Confidence Analysis:")
            print(f"  Correct predictions:")
            print(f"    Mean confidence: {y_proba[~errors].mean():.3f}")
            print(f"    Std confidence:  {y_proba[~errors].std():.3f}")
            print(f"  Incorrect predictions:")
            print(f"    Mean confidence: {y_proba[errors].mean():.3f}")
            print(f"    Std confidence:  {y_proba[errors].std():.3f}")
            
            # Check if model is overconfident in errors
            high_conf_errors = ((y_proba > 0.7) | (y_proba < 0.3)) & errors
            print(f"\n  High-confidence errors: {high_conf_errors.sum()} " +
                  f"({high_conf_errors.sum()/errors.sum()*100:.1f}% of errors)")
            
            if high_conf_errors.sum() / errors.sum() > 0.3:
                print(f"  ⚠️  Model is overconfident - calibration may be poor")
            
            error_data[name] = {
                'errors': errors,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'y_proba': y_proba,
                'y_test': y_test,
                'y_pred': y_pred
            }
        
        # Visualize error patterns
        self._plot_error_analysis(error_data)
        
        return error_data
    
    def _plot_error_analysis(self, error_data):
        """Plot error analysis visualizations."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (name, data) in enumerate(error_data.items()):
            # Plot 1: Confidence distribution by correctness
            ax1 = axes[idx]
            
            correct_proba = data['y_proba'][~data['errors']]
            incorrect_proba = data['y_proba'][data['errors']]
            
            ax1.hist(correct_proba, bins=30, alpha=0.5, label='Correct', 
                    color='green', edgecolor='black')
            ax1.hist(incorrect_proba, bins=30, alpha=0.5, label='Incorrect',
                    color='red', edgecolor='black')
            
            ax1.set_xlabel('Prediction Confidence (Prob of Bankrupt)', 
                          fontsize=10, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax1.set_title(f'{name}\nPrediction Confidence Distribution', 
                         fontsize=11, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Plot 2: Error types by confidence
            ax2 = axes[idx + 3]
            
            fp_proba = data['y_proba'][data['false_positives']]
            fn_proba = data['y_proba'][data['false_negatives']]
            
            ax2.hist(fp_proba, bins=20, alpha=0.6, label='False Positives (Type I)',
                    color='orange', edgecolor='black')
            ax2.hist(fn_proba, bins=20, alpha=0.6, label='False Negatives (Type II)',
                    color='purple', edgecolor='black')
            
            ax2.set_xlabel('Prediction Confidence', fontsize=10, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=10, fontweight='bold')
            ax2.set_title(f'{name}\nError Types by Confidence', 
                         fontsize=11, fontweight='bold')
            ax2.legend()
            ax2.grid(alpha=0.3)
            
            # Add vertical lines for decision boundary
            ax2.axvline(0.5, color='red', linestyle='--', linewidth=2,
                       label='Decision Boundary', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/error_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved: {self.output_dir}/error_analysis.png")
    
    def calibration_analysis(self):
        """
        Model calibration analysis.
        
        ADVANCED: Check if predicted probabilities match actual frequencies.
        Well-calibrated models are more trustworthy.
        """
        print("\n" + "="*80)
        print("MODEL CALIBRATION ANALYSIS")
        print("="*80)
        print("\nCalibration measures whether predicted probabilities are reliable.")
        print("A well-calibrated model's predicted 70% probability means the event")
        print("actually occurs 70% of the time.")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (name, model) in enumerate(self.models.items()):
            print(f"\n{name}:")
            
            # Collect calibration data across folds
            y_true_all = []
            y_proba_all = []
            
            for train_idx, test_idx in self.cv.split(self.X, self.y):
                X_train, X_test = self.X[train_idx], self.X[test_idx]
                y_train, y_test = self.y[train_idx], self.y[test_idx]
                
                model.fit(X_train, y_train)
                
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                else:
                    continue  # Skip if no probability
                
                y_true_all.extend(y_test)
                y_proba_all.extend(y_proba)
            
            y_true_all = np.array(y_true_all)
            y_proba_all = np.array(y_proba_all)
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true_all, y_proba_all, n_bins=10, strategy='uniform'
            )
            
            # Calculate calibration metrics
            brier = brier_score_loss(y_true_all, y_proba_all)
            logloss = log_loss(y_true_all, y_proba_all)
            
            print(f"  Brier Score: {brier:.4f} (lower is better, 0 = perfect)")
            print(f"  Log Loss:    {logloss:.4f} (lower is better)")
            
            # Plot calibration curve
            ax = axes[idx]
            
            ax.plot([0, 1], [0, 1], 'k--', linewidth=2, 
                   label='Perfect Calibration')
            ax.plot(mean_predicted_value, fraction_of_positives, 
                   'o-', linewidth=2, markersize=8,
                   label=f'{name}\n(Brier={brier:.3f})')
            
            ax.set_xlabel('Mean Predicted Probability', fontsize=11, fontweight='bold')
            ax.set_ylabel('Fraction of Positives', fontsize=11, fontweight='bold')
            ax.set_title(f'{name}\nCalibration Curve', fontsize=12, fontweight='bold')
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/calibration_curves.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\n✓ Saved: {self.output_dir}/calibration_curves.png")


def run_post_ml_analysis(filepath='data.csv', output_dir='plots'):
    """
    Run complete post-ML analysis pipeline.
    
    This provides the rigorous model evaluation that is MISSING
    from your report.
    """
    
    print("\n" + "="*80)
    print("POST-ML ANALYSIS FOR TAIWAN BANKRUPTCY PREDICTION")
    print("="*80)
    print("\nThis analysis addresses CRITICAL GAPS in your results section:")
    print("1. Cross-validation stability (not shown in report)")
    print("2. Statistical significance testing (not done in report)")
    print("3. ROC/PR curves (essential but missing)")
    print("4. Error analysis (WHY do models fail?)")
    print("5. Model calibration (are probabilities trustworthy?)")
    
    # Load data
    print("\nLoading data...")
    df = pd.read_csv(filepath)
    
    target_col = 'Bankrupt?'
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Remove outliers (as per your methodology)
    from scipy.stats import zscore
    z_scores = np.abs(zscore(X_scaled, axis=0))
    outliers = (z_scores > 3).any(axis=1)
    
    X_clean = X_scaled[~outliers]
    y_clean = y[~outliers]
    
    print(f"Data shape after outlier removal: {X_clean.shape}")
    print(f"Removed {outliers.sum()} outliers ({outliers.sum()/len(X)*100:.2f}%)")
    
    # Initialize analysis
    analyzer = PostMLAnalysis(X_clean, y_clean, random_state=24, output_dir=output_dir)
    
    # Run analyses
    print("\n" + "="*80)
    print("RUNNING ANALYSES...")
    print("="*80)
    
    # 1. Detailed cross-validation
    cv_results = analyzer.cross_validation_detailed()
    
    # 2. ROC and PR curves
    roc_results, pr_results = analyzer.roc_pr_curves()
    
    # 3. Confusion matrices
    analyzer.confusion_matrix_analysis()
    
    # 4. Error analysis
    error_results = analyzer.error_analysis()
    
    # 5. Calibration
    analyzer.calibration_analysis()
    
    print("\n" + "="*80)
    print("POST-ML ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nAll visualizations saved to '{output_dir}/' directory")
    print("\nKEY TAKEAWAYS:")
    print("1. Check if model differences are statistically significant")
    print("2. Examine ROC/PR curves - are they substantially different?")
    print("3. Review error analysis - where do models fail?")
    print("4. Assess calibration - are predicted probabilities reliable?")
    print("\nThese analyses should inform your discussion and conclusions.")


if __name__ == "__main__":
    run_post_ml_analysis(filepath='data.csv', output_dir='plots')