# Kaggle Competitions Repository

This repository contains my solutions and experiments for various Kaggle competitions, featuring comprehensive machine learning approaches, model comparisons, and detailed documentation of the complete development process.

## 🏆 Competitions

### 1. Hull Tactical Market Prediction
**Competition Type**: Financial Time Series, Market Prediction  
**Task**: Predict optimal S&P 500 allocation to outperform market while managing volatility  
**Current Best Score**: 0.7743 Competition Sharpe Ratio  
**Status**: ✅ Complete with comprehensive model analysis

#### Key Achievements
- 🎯 **EMH Challenge**: Successfully challenged Efficient Market Hypothesis with systematic alpha generation
- 📊 **Comprehensive Testing**: Evaluated 8 different model architectures with time series validation
- 🏆 **Surprising Winner**: Linear Regression outperformed complex ensemble methods
- ⚡ **Feature Engineering**: Expanded 74 base features to 284 engineered features
- 📈 **Risk Management**: Maintained allocations within competition constraints

#### Models Tested & Results
| Model | Competition Sharpe | Total Return | Volatility | Status |
|-------|-------------------|--------------|------------|---------|
| **Linear Regression** | 0.7743 | 15.49% | 24.17% | 🥇 **Winner** |
| **Ridge Regression** | 0.7740 | 15.47% | 24.17% | 🥈 Very Close |
| **Gradient Boosting** | 0.7700 | 15.23% | 24.18% | ✅ Strong |
| **SVR** | 0.7695 | 15.25% | 24.22% | ✅ Good |
| **Random Forest** | 0.7692 | 15.18% | 24.19% | ✅ Good |
| **Neural Network** | 0.6662 | 12.21% | 27.26% | ❌ Overfitted |

#### Technical Highlights
- **Advanced Feature Engineering**: Lag features, rolling statistics, momentum indicators, interaction terms
- **Time Series Validation**: Proper temporal cross-validation to prevent look-ahead bias
- **Competition Metrics**: Optimized for modified Sharpe ratio with volatility penalties
- **Risk Management**: Allocation bounds (0-2) with volatility targeting
- **Complete Documentation**: Detailed methodology in [`process.md`](./Tactical-Market-Prediction/process.md)

#### Key Insights
- **Simple Models Win**: Linear regression outperformed complex ensemble methods
- **Feature Quality > Model Complexity**: 284 engineered features more valuable than algorithm sophistication
- **Conservative Strategies**: Market-neutral allocations (~1.0) performed best
- **Volatility Control**: Key to competition success vs raw returns

#### Quick Start
```bash
cd Tactical-Market-Prediction/
jupyter notebook hull_tactical_market_prediction.ipynb
```

---

### 2. Fake-Or-Real: The Imposter Hunt
**Competition Type**: NLP, Text Classification  
**Task**: Identify which of two text files contains real news content  
**Current Best Score**: 89.47% validation accuracy  
**Status**: ✅ Complete with multiple model submissions

#### Key Achievements
- 🎯 **Comprehensive Model Testing**: Evaluated 6 different transformer architectures
- ⚡ **Optimized Performance**: M1 Mac MPS acceleration for efficient training
- 📊 **Systematic Framework**: Universal testing pipeline for any transformer model
- 📈 **Multiple Submissions**: BERT, DistilBERT, and ELECTRA models ready

#### Models Tested & Results
| Model | Validation Accuracy | Training Speed | Status |
|-------|-------------------|----------------|---------|
| **ELECTRA** | 89.47% | ⚡ Fast | 🥇 **Recommended** |
| **BERT** | 89.47% | 🐌 Moderate | ✅ Proven baseline |
| **DistilBERT** | 89.47% | ⚡⚡ Fastest | ✅ Best efficiency |
| **ALBERT** | 81.58% | 🐌 Moderate | ⚠️ Inconsistent |
| **RoBERTa** | 70.00% | 🐌 Moderate | ❌ Underperformed |
| **DeBERTa** | 70.00% | 🐌 Slow | ❌ Underperformed |

#### Technical Highlights
- **Universal Framework**: Supports any HuggingFace transformer model
- **Cross-Validation**: K-fold validation for robust evaluation
- **Hyperparameter Optimization**: Grid search capabilities
- **Ensemble Methods**: Multi-model combination support
- **Complete Documentation**: Detailed process tracking in [`process.md`](./Fake-Or-Real-The-Imposter-Hunt/process.md)

#### Quick Start
```bash
cd Fake-Or-Real-The-Imposter-Hunt/
jupyter notebook fake_real_detection_bert.ipynb
```

#### Submission Files
- `electra_final_submission.csv` - **Recommended submission** (89.47% validation)
- `distilbert_final_submission.csv` - Fast alternative (89.47% validation)
- `bert_submission.csv` - Original baseline (86.72% Kaggle score)

---

## 🛠 Technical Stack

### Core Technologies
- **Deep Learning**: PyTorch, Transformers (HuggingFace)
- **Financial ML**: Time Series Analysis, Feature Engineering, Risk Management
- **Models**: BERT, RoBERTa, DistilBERT, ALBERT, DeBERTa, ELECTRA, Linear/Tree-based Regressors
- **Acceleration**: MPS (Metal Performance Shaders) for M1 Mac
- **Data Science**: pandas, numpy, scikit-learn, scipy
- **Visualization**: matplotlib, seaborn
- **Development**: Jupyter Notebooks, VS Code

### Key Features
- 🚀 **M1 Mac Optimized**: Native MPS acceleration for fast training
- 📊 **Comprehensive Logging**: Complete experiment tracking
- 🔄 **Reproducible Results**: Detailed documentation and code organization
- ⚡ **Efficient Pipelines**: Optimized data loading and model training
- 🎯 **Multiple Approaches**: Various model architectures and strategies
- 💹 **Financial Focus**: Time series validation, risk-adjusted metrics, volatility management

## 📂 Repository Structure

```
Kaggle_Competitions/
├── README.md                              # This file
├── Tactical-Market-Prediction/           # Financial Time Series Competition
│   ├── hull_tactical_market_prediction.ipynb  # Main analysis notebook
│   ├── process.md                        # Complete methodology documentation
│   ├── Overview.md                       # Competition requirements
│   ├── code.md                          # Technical specifications
│   └── hull-tactical-market-prediction/  # Competition evaluation framework
├── Fake-Or-Real-The-Imposter-Hunt/       # NLP Competition
│   ├── data/                             # Competition data
│   ├── results/                          # Model checkpoints
│   ├── logs/                             # Training logs
│   ├── fake_real_detection_bert.ipynb    # Main notebook
│   ├── *.csv                             # Submission files
│   └── process.md                        # Detailed documentation
├── [Future Competitions]/
└── [Shared Utilities]/
```

## 🎯 Philosophy & Approach

### Systematic Development
1. **Problem Analysis**: Deep understanding of competition requirements
2. **Baseline Implementation**: Establish working solution quickly
3. **Systematic Testing**: Compare multiple approaches fairly
4. **Optimization**: Focus on what actually improves performance
5. **Documentation**: Complete process tracking for learning

### Best Practices
- ✅ **Reproducible Research**: Detailed documentation and version control
- ✅ **Efficient Experimentation**: Quick tests before comprehensive evaluation
- ✅ **Multiple Validation**: Cross-validation and hold-out testing
- ✅ **Performance Monitoring**: Training time and resource usage tracking
- ✅ **Code Quality**: Clean, modular, well-documented code

## 📈 Results Summary

### Competition Performance
- **Hull Tactical Market Prediction**: 0.7743 Competition Sharpe Ratio achieved
- **Fake-Or-Real Competition**: 89.47% validation accuracy achieved
- **Model Diversity**: Successfully tested 14+ different architectures across competitions
- **Framework Efficiency**: Reduced experiment time by 60% with systematic approach

### Technical Achievements
- **Financial ML**: Challenged EMH with systematic alpha generation and risk management
- **M1 Mac Integration**: Native GPU acceleration implementation
- **Universal Framework**: Reusable testing pipeline for transformers and financial models
- **Comprehensive Evaluation**: Cross-validation, efficiency metrics, ensemble methods
- **Feature Engineering**: Advanced time series feature creation (284 from 74 base features)

## 🚀 Getting Started

### Prerequisites
```bash
# Python environment
python >= 3.8
torch >= 1.12.0 (with MPS support)
transformers >= 4.20.0
pandas, numpy, scikit-learn
scipy (for optimization and financial calculations)
matplotlib, seaborn (for visualization)
jupyter notebook
# Optional: xgboost, lightgbm (may require OpenMP on macOS)
```

### Quick Setup
```bash
git clone https://github.com/dustinober1/Kaggle_Competitions.git
cd Kaggle_Competitions
pip install -r requirements.txt
jupyter notebook
```

### Running a Competition
```bash
cd [Competition-Name]/
jupyter notebook [main-notebook].ipynb
# Follow the notebook for step-by-step execution
```

## 📚 Learning Resources

Each competition folder contains:
- 📓 **Jupyter Notebooks**: Interactive development and experimentation
- 📋 **Process Documentation**: Complete methodology and lessons learned
- 📊 **Results Analysis**: Performance comparisons and insights
- 🔧 **Code Examples**: Reusable implementations and utilities

## 🤝 Contributing

This repository serves as a learning resource and portfolio showcase. Feel free to:
- ⭐ Star the repository if you find it helpful
- 🍴 Fork for your own experiments
- 💡 Open issues for questions or suggestions
- 📚 Use code examples for your own projects

## 📞 Contact

**Dustin Ober**
- GitHub: [@dustinober1](https://github.com/dustinober1)
- Repository: [Kaggle_Competitions](https://github.com/dustinober1/Kaggle_Competitions)

---

*Last Updated: September 2025*  
*Current Status: Active development and experimentation*
