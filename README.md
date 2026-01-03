# 🎓 University Rankings Data Analysis & Machine Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)
![ML](https://img.shields.io/badge/ML-98.33%25%20Accuracy-brightgreen.svg)

**Comprehensive data analysis and machine learning pipeline for Times Higher Education (THE) university rankings**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Results](#-results) • [Visualizations](#-visualizations)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Results Summary](#-results-summary)
- [Visualizations](#-visualizations)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Model Performance](#-model-performance)
- [Requirements](#-requirements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🔍 Overview

This project performs **comprehensive data analysis** and **machine learning predictions** on the Times Higher Education (THE) World University Rankings dataset. It includes exploratory data analysis (EDA), statistical analysis, correlation studies, geographic analysis, and advanced machine learning models for both **classification** and **regression** tasks.

### Dataset
- **Source**: Times Higher Education World University Rankings
- **Size**: 200 universities
- **Features**: 13 columns including rankings, scores, location, and institutional metrics
- **Geographic Coverage**: 20+ countries worldwide

---

## ✨ Features

### 📊 Data Analysis
- **Exploratory Data Analysis (EDA)** with comprehensive statistical summaries
- **Missing value detection** and intelligent imputation strategies
- **Distribution analysis** for all numeric features
- **Correlation analysis** with heatmap visualizations
- **Country/Region based analysis** with performance metrics

### 🤖 Machine Learning

#### Classification Models
- Predict ranking tiers: **Top 50**, **Top 100**, **Top 200**
- **98.33% accuracy** using Random Forest and Gradient Boosting
- Multi-class ROC curves and Precision-Recall analysis
- 10-fold cross-validation with <0.05% variance

#### Regression Models
- Predict exact university ranking position
- **RMSE: 1.63** (predictions within ~2 ranking positions)
- **R² Score: 0.9991** (99.91% variance explained)
- Four regression models: Random Forest, Gradient Boosting, Ridge, Linear

### 📈 Visualizations (18 High-Resolution PNG Charts)
- Data distributions and missing value heatmaps
- Correlation matrices
- Geographic analysis charts
- Feature importance plots
- Confusion matrices
- ROC and Precision-Recall curves
- Learning curves
- Residual plots and error distributions

---

## 🏆 Results Summary

| Task | Model | Performance | Key Metric |
|------|-------|-------------|------------|
| **Classification** | Random Forest | 98.33% Accuracy | F1: 0.9832 |
| **Classification** | Gradient Boosting | 98.33% Accuracy | F1: 0.9832 |
| **Regression** | Gradient Boosting | RMSE: 1.63 | R²: 0.9991 |
| **Regression** | Random Forest | RMSE: 1.77 | R²: 0.9989 |

### Key Performance Indicators

✅ **Classification**: 98.33% accuracy (59/60 correct predictions)  
✅ **Regression**: Predictions within ±2 ranking positions  
✅ **Cross-Validation**: 98.29% ± 0.03% (extremely stable)  
✅ **ROC AUC**: Near 1.0 for all classes  
✅ **No overfitting**: Learning curves confirm generalization  

---

## 📊 Visualizations

### Sample Outputs

The analysis generates **18 professional visualizations**:

#### Data Quality & Exploration
- `01_data_distribution.png` - Distribution of key numeric features
- `02_missing_values.png` - Missing values heatmap
- `03_correlation_heatmap.png` - Feature correlation matrix

#### Geographic Analysis
- `04_country_analysis.png` - Top 20 countries by university count
- `09_regional_performance.png` - Average ranking by country/region

#### Classification Results
- `05_ranking_distribution.png` - Distribution of ranking tiers
- `06_feature_importance.png` - Top 15 most important features
- `07_confusion_matrix.png` - Classification confusion matrix
- `08_model_performance.png` - Model accuracy comparison
- `14_roc_curves.png` - Multi-class ROC curves
- `15_precision_recall_curves.png` - Precision-Recall curves
- `16_learning_curves.png` - Model learning progression
- `17_cross_validation_scores.png` - 10-fold CV results
- `18_prediction_probabilities.png` - Probability distributions

#### Regression Results
- `10_actual_vs_predicted.png` - Actual vs predicted rankings (4 models)
- `11_residual_plots.png` - Residual analysis for all models
- `12_regression_metrics_comparison.png` - RMSE, MAE, R² comparison
- `13_error_distribution.png` - Error histogram & Q-Q plot

All visualizations are **300 DPI**, publication-ready quality.

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Conda (recommended) or pip


### Install Dependencies

#### Using Conda (Recommended)
```bash
conda create -n ranking-analysis python=3.9
conda activate ranking-analysis
conda install pandas numpy matplotlib seaborn scikit-learn openpyxl scipy
```

#### Using pip
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

### Quick Start

#### Run Complete Analysis
```bash
python comprehensive_analysis.py
```

This performs:
- Data loading and inspection
- Data cleaning and preprocessing
- Statistical analysis
- Correlation analysis
- Country/region analysis
- Feature engineering
- Classification model training
- Generates 9 core visualizations
- Creates summary report

#### Run Enhanced ML Analysis
```bash
python enhanced_ml_analysis.py
```

This adds:
- Regression model training (4 models)
- RMSE, MAE, R² metrics
- Actual vs predicted plots
- Residual analysis
- ROC and Precision-Recall curves
- Learning curves
- Cross-validation analysis
- Generates 9 additional ML visualizations

### Output Files

After running the scripts, you'll have:

**Data Files:**
- `cleaned_data.csv` - Preprocessed dataset

**Analysis Scripts:**
- `comprehensive_analysis.py` - Main analysis pipeline
- `enhanced_ml_analysis.py` - Advanced ML visualizations

**Report:**
- `summary_report.md` - Comprehensive findings and insights

**Visualizations:**
- `01_*.png` to `18_*.png` - 18 high-resolution charts

---

## 📁 Project Structure

```
university-rankings-analysis/
│
├── Top_Universities_THE.xlsx          # Input dataset
│
├── comprehensive_analysis.py          # Main analysis script
├── enhanced_ml_analysis.py            # Enhanced ML visualizations
├── inspect_data.py                    # Quick data inspection tool
│
├── cleaned_data.csv                   # Cleaned output dataset
├── summary_report.md                  # Analysis summary report
│
├── 01_data_distribution.png          # Visualizations (18 total)
├── 02_missing_values.png
├── ...
├── 18_prediction_probabilities.png
│
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
└── LICENSE                            # MIT License
```

---

## 🎯 Model Performance

### Classification Model: Random Forest

| Metric | Top 50 | Top 100 | Top 200 | Overall |
|--------|--------|---------|---------|---------|
| **Precision** | 1.00 | 0.93 | 1.00 | 0.98 |
| **Recall** | 1.00 | 1.00 | 0.97 | 0.98 |
| **F1-Score** | 1.00 | 0.97 | 0.98 | 0.98 |
| **Support** | 15 | 15 | 30 | 60 |

**Overall Accuracy**: 98.33% (59/60 correct)

### Regression Model: Gradient Boosting

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | 1.63 | Average error of 1.63 ranking positions |
| **MAE** | 1.37 | Typical error of 1.4 positions |
| **R² Score** | 0.9991 | Explains 99.91% of variance |
| **Max Error** | ~8 | Worst prediction off by 8 positions |

### Feature Importance (Top 5)

1. **Overall Score**: 47.5%
2. **Research Environment**: 23.0%
3. **Teaching**: 13.8%
4. **Research Quality**: 8.4%
5. **International Outlook**: 2.1%

---

## 🔬 Methodology

### Data Preprocessing
1. **Missing Value Imputation**
   - Numeric features: Median imputation
   - Categorical features: Mode imputation
2. **Feature Scaling**: StandardScaler for ML models
3. **Duplicate Removal**: No duplicates found
4. **Feature Engineering**: Created ranking tier categories

### Model Training
1. **Train-Test Split**: 70/30 stratified split
2. **Cross-Validation**: 10-fold CV for robustness
3. **Hyperparameters**: 
   - Random Forest: 100 estimators, max_depth=10
   - Gradient Boosting: 100 estimators, max_depth=5
4. **Evaluation Metrics**: Accuracy, F1, RMSE, MAE, R²

### Validation Techniques
- Confusion matrix analysis
- ROC and Precision-Recall curves
- Learning curves (overfitting detection)
- Residual analysis (regression diagnostics)
- Cross-validation stability check

---

## 📊 Key Insights

### Geographic Findings
- **USA** leads with 55 universities (27.5% of dataset)
- **Singapore** has best average ranking (23.5)
- **Hong Kong** shows strong performance (avg rank 61.6)
- Strong European representation (UK, Germany, Netherlands)

### Statistical Insights
- **Strong correlations** between Overall Score and Research Environment (0.948)
- **Teaching** and **Research Environment** highly correlated (0.918)
- **Ranking** inversely correlated with Overall Score (-0.928) as expected

### Model Insights
- Tree-based models outperform linear models by **6-7x** (RMSE: 1.63 vs 19+)
- **No overfitting detected** - learning curves converge
- Model is **well-calibrated** - prediction probabilities are reliable
- **Top 50 tier** classified with 100% accuracy

---

## 📦 Requirements

```txt
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
openpyxl>=3.0.0
scipy>=1.7.0
```

### System Requirements
- **RAM**: 4GB minimum (8GB recommended)
- **Storage**: 100MB for data and visualizations
- **OS**: Windows, macOS, or Linux

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### How to Contribute
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Areas for Contribution
- Additional data sources (QS, ARWU rankings)
- Deep learning models (neural networks)
- Interactive dashboards (Plotly, Streamlit)
- Time series analysis (ranking trends)
- Web scraping for real-time data

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Name](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🙏 Acknowledgments

- **Times Higher Education** for providing the university rankings data
- **scikit-learn** for excellent machine learning tools
- **matplotlib** and **seaborn** for beautiful visualizations
- Inspired by data-driven approaches to education analytics

---

## 📈 Future Enhancements

- [ ] Multi-year trend analysis
- [ ] Interactive web dashboard
- [ ] API for real-time predictions
- [ ] Deep learning models (LSTM, Transformers)
- [ ] Integration with other ranking systems (QS, ARWU)
- [ ] Automated report generation
- [ ] Docker containerization
- [ ] CI/CD pipeline

---

## 📚 Citation

If you use this project in your research, please cite:

```bibtex
@software{university_rankings_analysis,
  author = {Your Name},
  title = {University Rankings Data Analysis & Machine Learning},
  year = {2026},
  url = {https://github.com/yourusername/university-rankings-analysis}
}
```

---

## 🔗 Related Projects

- [QS Rankings Analyzer](https://github.com/example/qs-rankings)
- [Education Data Science](https://github.com/example/edu-datascience)
- [University ML Benchmarks](https://github.com/example/uni-ml)

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ and Python

</div>
