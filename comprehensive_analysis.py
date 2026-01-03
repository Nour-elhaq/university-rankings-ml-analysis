"""
Comprehensive University Rankings Data Analysis
================================================
This script performs a complete analysis of university rankings data including:
- Data inspection and structure analysis
- Data cleaning and preprocessing
- Statistical summary
- Distribution analysis
- Correlation analysis
- Country/Region based analysis
- Machine Learning prediction (ranking tier classification)
- Visualization generation (all as PNG files)
- Summary report generation
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to avoid Qt errors
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class UniversityRankingAnalysis:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.df_cleaned = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load the university rankings data"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)
        self.df = pd.read_excel(self.file_path)
        print(f"✓ Data loaded successfully: {self.df.shape[0]} rows × {self.df.shape[1]} columns\n")
        return self
    
    def inspect_data(self):
        """Inspect data structure and content"""
        print("=" * 80)
        print("DATA STRUCTURE AND CONTENT INSPECTION")
        print("=" * 80)
        
        print(f"\nDataset Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
        print("\n" + "-" * 80)
        print("COLUMNS:")
        print("-" * 80)
        for i, col in enumerate(self.df.columns, 1):
            print(f"{i:2d}. {col}")
        
        print("\n" + "-" * 80)
        print("DATA TYPES:")
        print("-" * 80)
        print(self.df.dtypes)
        
        print("\n" + "-" * 80)
        print("MISSING VALUES:")
        print("-" * 80)
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Percentage': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        print("\n" + "-" * 80)
        print("FIRST 5 ROWS:")
        print("-" * 80)
        print(self.df.head())
        
        # Visualize missing values
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
        plt.title('Missing Values Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('02_missing_values.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 02_missing_values.png")
        
        return self
    
    def clean_data(self):
        """Clean and preprocess the data"""
        print("\n" + "=" * 80)
        print("DATA CLEANING AND PREPROCESSING")
        print("=" * 80)
        
        self.df_cleaned = self.df.copy()
        
        # Get numeric and categorical columns
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df_cleaned.select_dtypes(include=['object']).columns.tolist()
        
        print(f"\nNumeric columns: {len(numeric_cols)}")
        print(f"Categorical columns: {len(categorical_cols)}")
        
        # Handle missing values in numeric columns (median imputation)
        for col in numeric_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                median_val = self.df_cleaned[col].median()
                self.df_cleaned[col].fillna(median_val, inplace=True)
                print(f"✓ Filled {col} with median: {median_val:.2f}")
        
        # Handle missing values in categorical columns (mode imputation)
        for col in categorical_cols:
            if self.df_cleaned[col].isnull().sum() > 0:
                mode_val = self.df_cleaned[col].mode()[0] if len(self.df_cleaned[col].mode()) > 0 else 'Unknown'
                self.df_cleaned[col].fillna(mode_val, inplace=True)
                print(f"✓ Filled {col} with mode: {mode_val}")
        
        # Remove duplicates
        duplicates = self.df_cleaned.duplicated().sum()
        if duplicates > 0:
            self.df_cleaned = self.df_cleaned.drop_duplicates()
            print(f"\n✓ Removed {duplicates} duplicate rows")
        
        # Save cleaned data
        self.df_cleaned.to_csv('cleaned_data.csv', index=False)
        print(f"\n✓ Cleaned dataset saved: {self.df_cleaned.shape[0]} rows × {self.df_cleaned.shape[1]} columns")
        print("✓ Saved: cleaned_data.csv")
        
        return self
    
    def statistical_summary(self):
        """Generate statistical summary"""
        print("\n" + "=" * 80)
        print("STATISTICAL SUMMARY")
        print("=" * 80)
        
        print("\n" + "-" * 80)
        print("DESCRIPTIVE STATISTICS:")
        print("-" * 80)
        print(self.df_cleaned.describe())
        
        return self
    
    def distribution_analysis(self):
        """Analyze distributions of key variables"""
        print("\n" + "=" * 80)
        print("DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create distribution plots for numeric features
        n_cols = min(len(numeric_cols), 6)
        if n_cols > 0:
            fig, axes = plt.subplots(nrows=(n_cols + 1) // 2, ncols=2, figsize=(15, 4 * ((n_cols + 1) // 2)))
            axes = axes.flatten() if n_cols > 1 else [axes]
            
            for idx, col in enumerate(numeric_cols[:6]):
                axes[idx].hist(self.df_cleaned[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
                axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel('Frequency')
                axes[idx].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(n_cols, len(axes)):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig('01_data_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: 01_data_distribution.png")
        
        return self
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 1:
            correlation_matrix = self.df_cleaned[numeric_cols].corr()
            
            print("\n" + "-" * 80)
            print("CORRELATION MATRIX (Top correlations):")
            print("-" * 80)
            
            # Get top correlations
            corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))
            
            corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
            for feat1, feat2, corr in corr_pairs_sorted[:10]:
                print(f"{feat1} <-> {feat2}: {corr:.3f}")
            
            # Visualize correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                       center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
            plt.title('Correlation Heatmap', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('03_correlation_heatmap.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("\n✓ Saved: 03_correlation_heatmap.png")
        
        return self
    
    def country_region_analysis(self):
        """Analyze data by country/region"""
        print("\n" + "=" * 80)
        print("COUNTRY/REGION BASED ANALYSIS")
        print("=" * 80)
        
        # Look for country/region columns
        country_cols = [col for col in self.df_cleaned.columns if 'country' in col.lower() or 'region' in col.lower() or 'location' in col.lower()]
        
        if len(country_cols) > 0:
            country_col = country_cols[0]
            print(f"\nAnalyzing by: {country_col}")
            
            # Count by country
            country_counts = self.df_cleaned[country_col].value_counts().head(20)
            print(f"\n" + "-" * 80)
            print(f"TOP 20 COUNTRIES/REGIONS:")
            print("-" * 80)
            print(country_counts)
            
            # Visualize top countries
            plt.figure(figsize=(12, 8))
            country_counts.plot(kind='barh', color='steelblue', edgecolor='black')
            plt.title(f'Top 20 Universities by {country_col}', fontsize=16, fontweight='bold')
            plt.xlabel('Number of Universities')
            plt.ylabel(country_col)
            plt.tight_layout()
            plt.savefig('04_country_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("\n✓ Saved: 04_country_analysis.png")
            
            # Regional performance analysis if we have ranking data
            ranking_cols = [col for col in self.df_cleaned.columns if 'rank' in col.lower() and self.df_cleaned[col].dtype in [np.int64, np.float64]]
            
            if len(ranking_cols) > 0:
                rank_col = ranking_cols[0]
                regional_perf = self.df_cleaned.groupby(country_col)[rank_col].agg(['mean', 'median', 'count']).sort_values('mean').head(15)
                
                print(f"\n" + "-" * 80)
                print(f"REGIONAL PERFORMANCE (Top 15 by average {rank_col}):")
                print("-" * 80)
                print(regional_perf)
                
                # Visualize regional performance
                fig, ax = plt.subplots(figsize=(12, 8))
                x = range(len(regional_perf))
                ax.barh(x, regional_perf['mean'], color='coral', edgecolor='black', label='Mean Rank')
                ax.set_yticks(x)
                ax.set_yticklabels(regional_perf.index)
                ax.set_xlabel(f'Average {rank_col}')
                ax.set_title(f'Top 15 Countries/Regions by Average {rank_col}', fontsize=16, fontweight='bold')
                ax.invert_xaxis()  # Lower rank is better
                plt.tight_layout()
                plt.savefig('09_regional_performance.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("\n✓ Saved: 09_regional_performance.png")
        
        return self
    
    def feature_engineering(self):
        """Engineer features for machine learning"""
        print("\n" + "=" * 80)
        print("FEATURE ENGINEERING FOR MACHINE LEARNING")
        print("=" * 80)
        
        # Find ranking column to create tiers
        ranking_cols = [col for col in self.df_cleaned.columns if 'rank' in col.lower()]
        
        if len(ranking_cols) > 0:
            rank_col = ranking_cols[0]
            print(f"\nUsing '{rank_col}' to create ranking tiers")
            
            # Create ranking tiers
            if self.df_cleaned[rank_col].dtype in [np.int64, np.float64]:
                # Create tiers based on ranking
                self.df_cleaned['ranking_tier'] = pd.cut(
                    self.df_cleaned[rank_col],
                    bins=[0, 50, 100, 200, float('inf')],
                    labels=['Top 50', 'Top 100', 'Top 200', 'Beyond 200']
                )
            else:
                # If rank is categorical or has ranges, use it directly
                self.df_cleaned['ranking_tier'] = self.df_cleaned[rank_col]
            
            print(f"\nRanking Tier Distribution:")
            print(self.df_cleaned['ranking_tier'].value_counts())
            
            # Visualize ranking distribution
            plt.figure(figsize=(10, 6))
            self.df_cleaned['ranking_tier'].value_counts().plot(kind='bar', color='teal', edgecolor='black')
            plt.title('Distribution of Ranking Tiers', fontsize=16, fontweight='bold')
            plt.xlabel('Ranking Tier')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('05_ranking_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("\n✓ Saved: 05_ranking_distribution.png")
        
        return self
    
    def train_ml_models(self):
        """Train machine learning models to predict ranking tier"""
        print("\n" + "=" * 80)
        print("MACHINE LEARNING MODEL TRAINING")
        print("=" * 80)
        
        if 'ranking_tier' not in self.df_cleaned.columns:
            print("⚠ No ranking tier created. Skipping ML training.")
            return self
        
        # Prepare features and target
        # Select numeric features
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove ranking columns from features
        feature_cols = [col for col in numeric_cols if 'rank' not in col.lower()]
        
        if len(feature_cols) == 0:
            print("⚠ No suitable features found for ML training.")
            return self
        
        print(f"\nUsing {len(feature_cols)} features: {feature_cols}")
        
        # Prepare data
        X = self.df_cleaned[feature_cols].fillna(0)
        y = self.df_cleaned['ranking_tier'].dropna()
        
        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Encode target
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Classes: {list(le.classes_)}")
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"\nTraining set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train multiple models
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        print("\n" + "-" * 80)
        print("TRAINING MODELS")
        print("-" * 80)
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_scaled)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
            
            # Store results
            self.models[name] = {
                'model': model,
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': y_pred,
                'y_test': y_test
            }
        
        # Get best model
        best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['accuracy'])
        best_model_info = self.models[best_model_name]
        
        print(f"\n{'='*80}")
        print(f"BEST MODEL: {best_model_name} (Accuracy: {best_model_info['accuracy']:.4f})")
        print(f"{'='*80}")
        
        # Detailed classification report for best model
        print("\nClassification Report:")
        print(classification_report(best_model_info['y_test'], best_model_info['predictions'], 
                                   target_names=le.classes_))
        
        # Confusion Matrix
        cm = confusion_matrix(best_model_info['y_test'], best_model_info['predictions'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=le.classes_, yticklabels=le.classes_)
        plt.title(f'Confusion Matrix - {best_model_name}', fontsize=16, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('07_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 07_confusion_matrix.png")
        
        # Feature Importance (for tree-based models)
        if hasattr(best_model_info['model'], 'feature_importances_'):
            importances = best_model_info['model'].feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Feature Importances:")
            print(feature_importance_df.head(10))
            
            # Visualize feature importance
            plt.figure(figsize=(12, 8))
            top_features = feature_importance_df.head(15)
            plt.barh(range(len(top_features)), top_features['importance'], color='forestgreen', edgecolor='black')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance')
            plt.title(f'Top 15 Feature Importances - {best_model_name}', fontsize=16, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig('06_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("\n✓ Saved: 06_feature_importance.png")
        
        # Model comparison
        model_names = list(self.models.keys())
        accuracies = [self.models[m]['accuracy'] for m in model_names]
        f1_scores = [self.models[m]['f1_score'] for m in model_names]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(model_names))
        width = 0.35
        
        ax.bar(x - width/2, accuracies, width, label='Accuracy', color='skyblue', edgecolor='black')
        ax.bar(x + width/2, f1_scores, width, label='F1 Score', color='lightcoral', edgecolor='black')
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('08_model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("\n✓ Saved: 08_model_performance.png")
        
        return self
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "=" * 80)
        print("GENERATING SUMMARY REPORT")
        print("=" * 80)
        
        report = []
        report.append("# University Rankings Data Analysis - Summary Report\n")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        report.append("---\n\n")
        
        # Dataset Overview
        report.append("## 1. Dataset Overview\n\n")
        report.append(f"- **Total Records**: {self.df_cleaned.shape[0]}\n")
        report.append(f"- **Total Features**: {self.df_cleaned.shape[1]}\n")
        report.append(f"- **Numeric Features**: {len(self.df_cleaned.select_dtypes(include=[np.number]).columns)}\n")
        report.append(f"- **Categorical Features**: {len(self.df_cleaned.select_dtypes(include=['object']).columns)}\n\n")
        
        report.append("### Data Quality\n\n")
        report.append(f"- Data cleaning successfully handled missing values\n")
        report.append(f"- All visualizations generated as high-resolution PNG files\n\n")
        
        # Key Findings
        report.append("## 2. Key Findings\n\n")
        
        # Statistical insights
        report.append("### Statistical Insights\n\n")
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            for col in numeric_cols[:5]:
                mean_val = self.df_cleaned[col].mean()
                median_val = self.df_cleaned[col].median()
                std_val = self.df_cleaned[col].std()
                report.append(f"- **{col}**: Mean = {mean_val:.2f}, Median = {median_val:.2f}, Std = {std_val:.2f}\n")
        report.append("\n")
        
        # Country analysis
        country_cols = [col for col in self.df_cleaned.columns if 'country' in col.lower() or 'region' in col.lower()]
        if len(country_cols) > 0:
            country_col = country_cols[0]
            top_country = self.df_cleaned[country_col].value_counts().index[0]
            top_count = self.df_cleaned[country_col].value_counts().values[0]
            report.append("### Geographic Distribution\n\n")
            report.append(f"- **Top Country/Region**: {top_country} with {top_count} universities\n")
            report.append(f"- **Total Countries/Regions**: {self.df_cleaned[country_col].nunique()}\n\n")
        
        # ML Results
        if len(self.models) > 0:
            report.append("## 3. Machine Learning Model Performance\n\n")
            report.append("| Model | Accuracy | F1 Score |\n")
            report.append("|-------|----------|----------|\n")
            for name, info in self.models.items():
                report.append(f"| {name} | {info['accuracy']:.4f} | {info['f1_score']:.4f} |\n")
            report.append("\n")
            
            best_model_name = max(self.models.keys(), key=lambda k: self.models[k]['accuracy'])
            report.append(f"**Best Performing Model**: {best_model_name}\n\n")
        
        # Visualizations
        report.append("## 4. Generated Visualizations\n\n")
        viz_files = [
            ("01_data_distribution.png", "Distribution of key numeric features"),
            ("02_missing_values.png", "Missing values heatmap"),
            ("03_correlation_heatmap.png", "Feature correlation matrix"),
            ("04_country_analysis.png", "Universities by country/region"),
            ("05_ranking_distribution.png", "Distribution of ranking tiers"),
            ("06_feature_importance.png", "ML model feature importance"),
            ("07_confusion_matrix.png", "Classification confusion matrix"),
            ("08_model_performance.png", "Model performance comparison"),
            ("09_regional_performance.png", "Regional performance metrics")
        ]
        
        for filename, description in viz_files:
            report.append(f"### {description}\n\n")
            report.append(f"![{description}]({filename})\n\n")
        
        # Recommendations
        report.append("## 5. Recommendations and Conclusions\n\n")
        report.append("Based on the comprehensive analysis:\n\n")
        report.append("1. **Data Quality**: The dataset has been successfully cleaned with appropriate handling of missing values\n")
        report.append("2. **Feature Relationships**: Correlation analysis reveals important relationships between features\n")
        report.append("3. **Geographic Patterns**: Clear patterns emerge in university distribution and performance by region\n")
        if len(self.models) > 0:
            report.append(f"4. **Predictive Capability**: Machine learning models can predict ranking tiers with {max(m['accuracy'] for m in self.models.values()):.1%} accuracy\n")
        report.append("\n---\n\n")
        report.append("*End of Report*\n")
        
        # Save report
        with open('summary_report.md', 'w', encoding='utf-8') as f:
            f.writelines(report)
        
        print("✓ Saved: summary_report.md")
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nGenerated Files:")
        print("  - cleaned_data.csv")
        print("  - summary_report.md")
        print("  - 01_data_distribution.png")
        print("  - 02_missing_values.png")
        print("  - 03_correlation_heatmap.png")
        print("  - 04_country_analysis.png")
        print("  - 05_ranking_distribution.png")
        print("  - 06_feature_importance.png")
        print("  - 07_confusion_matrix.png")
        print("  - 08_model_performance.png")
        print("  - 09_regional_performance.png")
        
        return self

def main():
    """Main execution function"""
    # Initialize analysis
    analysis = UniversityRankingAnalysis('Top_Universities_THE.xlsx')
    
    # Execute full analysis pipeline
    analysis.load_data()
    analysis.inspect_data()
    analysis.clean_data()
    analysis.statistical_summary()
    analysis.distribution_analysis()
    analysis.correlation_analysis()
    analysis.country_region_analysis()
    analysis.feature_engineering()
    analysis.train_ml_models()
    analysis.generate_summary_report()

if __name__ == "__main__":
    main()
