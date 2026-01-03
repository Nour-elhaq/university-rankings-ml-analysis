"""
Enhanced Machine Learning Analysis with Advanced Visualizations
================================================================
This script adds comprehensive ML visualizations including:
- Regression models (with RMSE, MAE, R²)
- Actual vs Predicted plots
- Residual plots
- ROC curves
- Precision-Recall curves
- Learning curves
- Cross-validation scores
- Feature distributions by tier
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class EnhancedMLAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_cleaned = None
        
    def load_and_prepare_data(self):
        """Load and prepare data"""
        print("=" * 80)
        print("LOADING DATA FOR ADVANCED ML ANALYSIS")
        print("=" * 80)
        
        self.df = pd.read_excel(self.file_path)
        self.df_cleaned = self.df.copy()
        
        # Handle missing values
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                self.df_cleaned[col].fillna(self.df_cleaned[col].median(), inplace=True)
        
        print(f"✓ Data loaded: {self.df_cleaned.shape[0]} rows × {self.df_cleaned.shape[1]} columns\n")
        return self
    
    def regression_analysis(self):
        """Perform regression to predict actual ranking"""
        print("=" * 80)
        print("REGRESSION ANALYSIS - PREDICTING ACTUAL RANKING")
        print("=" * 80)
        
        # Prepare features (exclude ranking columns)
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if 'rank' not in col.lower()]
        
        X = self.df_cleaned[feature_cols].fillna(0)
        
        # Target: actual ranking
        rank_col = [col for col in self.df_cleaned.columns if 'rank' in col.lower()][0]
        y = self.df_cleaned[rank_col]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"\nFeatures: {len(feature_cols)}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}\n")
        
        # Train multiple regression models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5),
            'Ridge Regression': Ridge(alpha=1.0),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        
        print("-" * 80)
        print("MODEL PERFORMANCE")
        print("-" * 80)
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'rmse': rmse,
                'mae': mae,
                'r2': r2
            }
            
            print(f"\n{name}:")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  R²:   {r2:.4f}")
        
        # Best model
        best_model_name = min(results.keys(), key=lambda k: results[k]['rmse'])
        best_model = results[best_model_name]
        
        print(f"\n{'='*80}")
        print(f"BEST MODEL: {best_model_name}")
        print(f"RMSE: {best_model['rmse']:.4f} | MAE: {best_model['mae']:.4f} | R²: {best_model['r2']:.4f}")
        print(f"{'='*80}\n")
        
        # === VISUALIZATION 1: Actual vs Predicted ===
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for idx, (name, result) in enumerate(results.items()):
            ax = axes[idx]
            ax.scatter(y_test, result['predictions'], alpha=0.6, edgecolors='k', s=80)
            
            # Perfect prediction line
            min_val = min(y_test.min(), result['predictions'].min())
            max_val = max(y_test.max(), result['predictions'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual Rank', fontsize=12, fontweight='bold')
            ax.set_ylabel('Predicted Rank', fontsize=12, fontweight='bold')
            ax.set_title(f'{name}\nRMSE: {result["rmse"]:.2f} | R²: {result["r2"]:.3f}', 
                        fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('10_actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 10_actual_vs_predicted.png")
        
        # === VISUALIZATION 2: Residual Plots ===
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        axes = axes.flatten()
        
        for idx, (name, result) in enumerate(results.items()):
            ax = axes[idx]
            residuals = y_test - result['predictions']
            
            ax.scatter(result['predictions'], residuals, alpha=0.6, edgecolors='k', s=80)
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            
            ax.set_xlabel('Predicted Rank', fontsize=12, fontweight='bold')
            ax.set_ylabel('Residuals', fontsize=12, fontweight='bold')
            ax.set_title(f'{name} - Residual Plot', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('11_residual_plots.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 11_residual_plots.png")
        
        # === VISUALIZATION 3: Model Metrics Comparison ===
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        model_names = list(results.keys())
        rmse_values = [results[m]['rmse'] for m in model_names]
        mae_values = [results[m]['mae'] for m in model_names]
        r2_values = [results[m]['r2'] for m in model_names]
        
        # RMSE
        axes[0].bar(model_names, rmse_values, color='coral', edgecolor='black', alpha=0.8)
        axes[0].set_ylabel('RMSE', fontsize=12, fontweight='bold')
        axes[0].set_title('Root Mean Squared Error', fontsize=14, fontweight='bold')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # MAE
        axes[1].bar(model_names, mae_values, color='skyblue', edgecolor='black', alpha=0.8)
        axes[1].set_ylabel('MAE', fontsize=12, fontweight='bold')
        axes[1].set_title('Mean Absolute Error', fontsize=14, fontweight='bold')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # R²
        axes[2].bar(model_names, r2_values, color='lightgreen', edgecolor='black', alpha=0.8)
        axes[2].set_ylabel('R² Score', fontsize=12, fontweight='bold')
        axes[2].set_title('R² Score (Coefficient of Determination)', fontsize=14, fontweight='bold')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        axes[2].axhline(y=0, color='red', linestyle='--', linewidth=1)
        
        plt.tight_layout()
        plt.savefig('12_regression_metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 12_regression_metrics_comparison.png")
        
        # === VISUALIZATION 4: Prediction Error Distribution ===
        best_residuals = y_test - best_model['predictions']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histogram
        axes[0].hist(best_residuals, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        axes[0].set_xlabel('Prediction Error (Actual - Predicted)', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
        axes[0].set_title(f'{best_model_name} - Error Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Q-Q Plot
        from scipy import stats
        stats.probplot(best_residuals, dist="norm", plot=axes[1])
        axes[1].set_title(f'{best_model_name} - Q-Q Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('13_error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 13_error_distribution.png")
        
        return self, results, y_test, X_test_scaled, feature_cols
    
    def classification_advanced_plots(self):
        """Generate advanced classification plots"""
        print("\n" + "=" * 80)
        print("ADVANCED CLASSIFICATION ANALYSIS")
        print("=" * 80)
        
        # Prepare data for classification
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numeric_cols if 'rank' not in col.lower()]
        
        # Create ranking tiers
        rank_col = [col for col in self.df_cleaned.columns if 'rank' in col.lower()][0]
        self.df_cleaned['ranking_tier'] = pd.cut(
            self.df_cleaned[rank_col],
            bins=[0, 50, 100, 200, float('inf')],
            labels=['Top 50', 'Top 100', 'Top 200', 'Beyond 200']
        )
        
        X = self.df_cleaned[feature_cols].fillna(0)
        y = self.df_cleaned['ranking_tier'].dropna()
        
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Encode
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        rf_model.fit(X_train_scaled, y_train)
        y_pred = rf_model.predict(X_test_scaled)
        y_pred_proba = rf_model.predict_proba(X_test_scaled)
        
        print(f"\nClassification Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        
        # === VISUALIZATION 5: ROC Curves (One-vs-Rest) ===
        n_classes = len(le.classes_)
        y_test_bin = label_binarize(y_test, classes=range(n_classes))
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        colors = ['blue', 'red', 'green', 'orange']
        for i, (color, class_name) in enumerate(zip(colors[:n_classes], le.classes_)):
            fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2, 
                   label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curves - Multi-class Classification', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('14_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 14_roc_curves.png")
        
        # === VISUALIZATION 6: Precision-Recall Curves ===
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for i, (color, class_name) in enumerate(zip(colors[:n_classes], le.classes_)):
            precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
            avg_precision = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
            ax.plot(recall, precision, color=color, lw=2,
                   label=f'{class_name} (AP = {avg_precision:.3f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
        ax.legend(loc="lower left", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('15_precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 15_precision_recall_curves.png")
        
        # === VISUALIZATION 7: Learning Curves ===
        train_sizes, train_scores, val_scores = learning_curve(
            rf_model, X_train_scaled, y_train,
            cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(train_sizes, train_mean, 'o-', color='blue', linewidth=2, markersize=8, label='Training Score')
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
        
        ax.plot(train_sizes, val_mean, 'o-', color='red', linewidth=2, markersize=8, label='Cross-Validation Score')
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color='red')
        
        ax.set_xlabel('Training Set Size', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
        ax.set_title('Learning Curves - Random Forest Classifier', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('16_learning_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 16_learning_curves.png")
        
        # === VISUALIZATION 8: Cross-Validation Scores ===
        cv_scores = cross_val_score(rf_model, X_train_scaled, y_train, cv=10, scoring='accuracy')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x_pos = np.arange(len(cv_scores))
        ax.bar(x_pos, cv_scores, color='teal', edgecolor='black', alpha=0.8)
        ax.axhline(y=cv_scores.mean(), color='red', linestyle='--', linewidth=2, 
                  label=f'Mean CV Score: {cv_scores.mean():.4f}')
        ax.axhline(y=cv_scores.mean() - cv_scores.std(), color='orange', linestyle=':', linewidth=1.5,
                  label=f'Mean - 1 Std: {cv_scores.mean() - cv_scores.std():.4f}')
        ax.axhline(y=cv_scores.mean() + cv_scores.std(), color='orange', linestyle=':', linewidth=1.5,
                  label=f'Mean + 1 Std: {cv_scores.mean() + cv_scores.std():.4f}')
        
        ax.set_xlabel('Fold Number', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy Score', fontsize=12, fontweight='bold')
        ax.set_title(f'10-Fold Cross-Validation Scores\nMean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'Fold {i+1}' for i in range(len(cv_scores))], rotation=45)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('17_cross_validation_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 17_cross_validation_scores.png")
        
        # === VISUALIZATION 9: Class Prediction Probabilities ===
        fig, axes = plt.subplots(1, n_classes, figsize=(18, 6))
        
        for i, (ax, class_name) in enumerate(zip(axes, le.classes_)):
            probas = y_pred_proba[:, i]
            true_class = (y_test == i).astype(int)
            
            ax.hist(probas[true_class == 1], bins=20, alpha=0.7, label=f'True {class_name}', 
                   color='green', edgecolor='black')
            ax.hist(probas[true_class == 0], bins=20, alpha=0.7, label=f'Other Classes',
                   color='red', edgecolor='black')
            ax.set_xlabel('Predicted Probability', fontsize=10, fontweight='bold')
            ax.set_ylabel('Count', fontsize=10, fontweight='bold')
            ax.set_title(f'{class_name}', fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('18_prediction_probabilities.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: 18_prediction_probabilities.png")
        
        return self

def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("ENHANCED MACHINE LEARNING ANALYSIS")
    print("=" * 80 + "\n")
    
    analysis = EnhancedMLAnalysis('Top_Universities_THE.xlsx')
    
    # Load data
    analysis.load_and_prepare_data()
    
    # Regression analysis with prediction plots
    analysis.regression_analysis()
    
    # Advanced classification plots
    analysis.classification_advanced_plots()
    
    print("\n" + "=" * 80)
    print("ENHANCED ML ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated Additional Visualizations:")
    print("  - 10_actual_vs_predicted.png (4 regression models)")
    print("  - 11_residual_plots.png (residual analysis)")
    print("  - 12_regression_metrics_comparison.png (RMSE, MAE, R²)")
    print("  - 13_error_distribution.png (error histogram & Q-Q plot)")
    print("  - 14_roc_curves.png (ROC curves for each class)")
    print("  - 15_precision_recall_curves.png (PR curves)")
    print("  - 16_learning_curves.png (model learning progression)")
    print("  - 17_cross_validation_scores.png (10-fold CV results)")
    print("  - 18_prediction_probabilities.png (probability distributions)")
    print("\nTotal new plots: 9")
    print("=" * 80)

if __name__ == "__main__":
    main()
