# Citation Impact Prediction - Project Roadmap

## Project Overview

**Objective**: Develop and deploy machine learning models to predict citation impact of academic publications using only information available at publication time.

**Collaboration**: American University of Beirut (AUB)

**Timeline**: Capstone Project

---

## Phase 1: Literature Review ✅

**Status**: COMPLETED

### Objectives
- Review existing citation prediction literature
- Identify research gaps and opportunities
- Establish baseline metrics and benchmarks

### Deliverables
- Literature review summary
- Baseline performance metrics
- Research gap analysis

---

## Phase 2: Data Collection ✅

**Status**: COMPLETED

### Objectives
- Coordinate with AUB (Khaled Noubani) for data access
- Obtain Scopus and SciVal publication data
- Ensure data covers 10,000+ publications

### Deliverables
- Scopus dataset (abstracts, metadata)
- SciVal dataset (citations, metrics)
- Data access agreements

**Data Files**:
- `scopus.csv` - Contains abstracts and publication metadata
- `scival.csv` - Contains citation counts and institutional data
- Common identifier: EID (Electronic Identifier)

---

## Phase 3: Data Cleaning & Preprocessing 🔄

**Status**: IN PROGRESS

### Objectives
- Merge Scopus and SciVal datasets using EID
- Clean and validate data quality
- Handle missing values and outliers
- Prepare data for feature engineering

### Tasks

#### 3.1 Data Merging ✅
- [x] Create data merge notebook
- [x] Match records using EID column
- [x] Validate merge accuracy
- [x] Handle encoding issues in Scopus data
- [x] Save merged dataset

#### 3.2 Data Cleaning (Next)
- [ ] Remove garbled text (encoding corruption)
- [ ] Filter non-English papers (if any)
- [ ] Validate author names and metadata
- [ ] Handle missing critical fields
- [ ] Remove duplicates

#### 3.3 Data Validation
- [ ] Check data types and ranges
- [ ] Validate citation counts
- [ ] Verify temporal consistency
- [ ] Document data quality metrics

### Deliverables
- Merged dataset: `data/merged_citation_data.csv`
- Data quality report
- Cleaning documentation

### Tools
- Python (pandas, numpy)
- Jupyter Notebook (`data_merge.ipynb`)
- Regular expressions for text cleaning

---

## Phase 4: Feature Engineering

**Status**: PENDING

### Objectives
- Extract meaningful features from raw data
- Create author, venue, and text features
- Handle missing data appropriately
- Prepare final modeling dataset

### Tasks

#### 4.1 Author Features
- [ ] Extract h-index (max, mean, sum)
- [ ] Calculate author citation counts
- [ ] Compute author collaboration metrics
- [ ] Create author seniority features

#### 4.2 Venue Features
- [ ] Manual curation of top 50+ venue scores
- [ ] Calculate historical citation statistics per venue
- [ ] Create venue impact factor features
- [ ] Compute venue acceptance rate metrics (if available)

#### 4.3 Text Features
- [ ] TF-IDF vectorization of abstracts (100-500 features)
- [ ] Title length and complexity metrics
- [ ] Keyword extraction and analysis
- [ ] Topic modeling (optional)

#### 4.4 Temporal Features
- [ ] Publication year
- [ ] Time since author's first publication
- [ ] Field growth rate at publication time

#### 4.5 Network Features (if data available)
- [ ] Citation network metrics
- [ ] Co-author network features
- [ ] Institutional collaboration metrics

### Deliverables
- Feature engineering notebook
- Feature documentation (descriptions, ranges)
- Final modeling dataset with all features
- Feature importance preliminary analysis

### Tools
- scikit-learn (TF-IDF, preprocessing)
- NLTK (text processing)
- Custom feature extraction scripts

---

## Phase 5: Model Development & Evaluation

**Status**: PENDING

### Objectives
- Train and compare multiple ML algorithms
- Optimize hyperparameters
- Evaluate model performance
- Select best models for deployment

### Tasks

#### 5.1 Classification Models
**Goal**: Identify papers likely to be highly cited (top 25%)

- [ ] **Baseline**: Logistic Regression
- [ ] **Tree-based**: Random Forest
- [ ] **Gradient Boosting**: XGBoost
- [ ] **Gradient Boosting**: LightGBM
- [ ] **Deep Learning** (optional): Neural Network

**Evaluation**:
- 5-fold stratified cross-validation
- Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- Confusion matrix analysis
- Precision-Recall curves

#### 5.2 Regression Models
**Goal**: Predict exact citation counts

- [ ] **Baseline**: Linear Regression (with log-transform)
- [ ] **Tree-based**: Random Forest Regressor
- [ ] **Gradient Boosting**: XGBoost Regressor
- [ ] **Gradient Boosting**: LightGBM Regressor

**Evaluation**:
- 5-fold cross-validation
- Metrics: R², MAE, RMSE, MAPE
- Ranking correlation (Spearman, Kendall)
- Error distribution analysis

#### 5.3 Model Comparison
- [ ] Compare performance across algorithms
- [ ] Analyze computational efficiency
- [ ] Test temporal generalization (train 2015-2017, test 2018-2020)
- [ ] Field-specific performance analysis

#### 5.4 Hyperparameter Tuning
- [ ] Grid search or random search
- [ ] Bayesian optimization (optional)
- [ ] Cross-validation for tuning

#### 5.5 Feature Importance Analysis
- [ ] Tree-based feature importance scores
- [ ] SHAP values for interpretability
- [ ] Feature ablation studies
- [ ] Domain-specific insights

### Deliverables
- Model training notebooks
- Trained model files (`.pkl` or `.joblib`)
- Model performance reports
- Feature importance analysis
- Model comparison matrix

### Tools
- scikit-learn
- XGBoost
- LightGBM
- PyTorch (if using deep learning)
- SHAP (for explainability)

---

## Phase 6: Visualization & Reporting

**Status**: PENDING

### Objectives
- Create comprehensive visualizations
- Document findings and insights
- Prepare final project report

### Tasks

#### 6.1 Performance Visualizations
- [ ] ROC curves for classification
- [ ] Precision-Recall curves
- [ ] Confusion matrices
- [ ] Learning curves
- [ ] Citation distribution (actual vs predicted)

#### 6.2 Feature Analysis Visualizations
- [ ] Feature importance bar charts (top 20)
- [ ] Correlation heatmaps
- [ ] SHAP summary plots
- [ ] Feature interaction visualizations

#### 6.3 Domain Analysis
- [ ] Field-specific performance
- [ ] Temporal trends in predictions
- [ ] Venue impact analysis
- [ ] Author metrics distribution

#### 6.4 Reporting
- [ ] Executive summary
- [ ] Methodology documentation
- [ ] Results and findings
- [ ] Limitations and future work
- [ ] Recommendations for AUB

### Deliverables
- Visualization notebooks
- Figure files (high-resolution)
- Final project report (PDF)
- Presentation slides

### Tools
- matplotlib
- seaborn
- plotly (interactive visualizations)
- LaTeX or Markdown (for report)

---

## Phase 7: Model Deployment (NEW) 🚀

**Status**: PENDING

### Objectives
- Deploy models using Streamlit
- Create interactive prediction interface
- Enable continuous model updates
- Provide real-time citation predictions

### Tasks

#### 7.1 Streamlit Application Development ✅
- [x] Create main Streamlit app structure
- [x] Design user interface (single & batch prediction)
- [x] Implement model loading mechanism
- [x] Create prediction workflows

#### 7.2 Application Features (Next)
- [ ] **Single Paper Prediction**
  - Input form for paper metadata
  - Real-time classification (high/standard impact)
  - Citation count regression prediction
  - Confidence scores and explanations

- [ ] **Batch Predictions**
  - CSV/Excel file upload
  - Bulk prediction processing
  - Downloadable results
  - Summary statistics

- [ ] **Model Insights Dashboard**
  - Feature importance visualization
  - Model performance metrics
  - Citation distribution plots
  - Interactive charts

- [ ] **Model Management**
  - Upload new trained models
  - Version control for models
  - A/B testing capability
  - Model retraining workflow

#### 7.3 Testing & Validation
- [ ] Unit tests for prediction functions
- [ ] Integration tests for Streamlit app
- [ ] User acceptance testing
- [ ] Performance optimization

#### 7.4 Deployment
- [ ] Local deployment instructions
- [ ] Streamlit Cloud deployment (optional)
- [ ] Docker containerization (optional)
- [ ] API endpoint creation (optional)

### Deliverables
- Fully functional Streamlit application
- Deployment documentation
- User guide
- API documentation (if applicable)

### Tools
- Streamlit
- Plotly (interactive visualizations)
- pickle/joblib (model serialization)
- Docker (optional)

**Files Created**:
- `app/streamlit_app.py` - Main Streamlit application
- `app/utils.py` - Utility functions for predictions
- Deployment configuration files

---

## Success Metrics

### Technical Metrics
- **Classification**: AUC-ROC > 0.80, F1-Score > 0.75
- **Regression**: R² > 0.70, MAPE < 30%
- **Temporal Generalization**: < 10% performance drop on future data
- **Deployment**: Response time < 2 seconds per prediction

### Business Metrics
- Enable identification of high-impact research with >75% accuracy
- Provide actionable insights for resource allocation
- Demonstrate clear feature importance patterns
- User-friendly interface for non-technical stakeholders

---

## Risk Management

### Potential Risks
1. **Data Quality**: Missing or inconsistent data
   - Mitigation: Robust data cleaning, multiple data sources

2. **Model Overfitting**: Too specific to training data
   - Mitigation: Cross-validation, temporal holdout, regularization

3. **Feature Availability**: Required features not available at prediction time
   - Mitigation: Use only pre-publication features, validate availability

4. **Deployment Challenges**: Technical issues with Streamlit
   - Mitigation: Thorough testing, fallback options, documentation

### Ethical Considerations
- Avoid bias against certain fields or author demographics
- Ensure transparency in predictions
- Provide confidence intervals and uncertainty quantification
- Document limitations clearly

---

## Timeline Summary

| Phase | Status | Duration |
|-------|--------|----------|
| 1. Literature Review | ✅ Complete | 2 weeks |
| 2. Data Collection | ✅ Complete | 2 weeks |
| 3. Data Cleaning | 🔄 In Progress | 2 weeks |
| 4. Feature Engineering | ⏳ Pending | 3 weeks |
| 5. Model Development | ⏳ Pending | 4 weeks |
| 6. Visualization | ⏳ Pending | 2 weeks |
| 7. Deployment | ⏳ Pending | 2 weeks |

**Total Estimated Duration**: 17 weeks

---

## Next Immediate Steps

1. ✅ Complete data merge (Scopus + SciVal)
2. 🔄 Run data quality validation
3. ⏳ Begin feature engineering for author metrics
4. ⏳ Create exploratory data analysis (EDA) notebook
5. ⏳ Start baseline model development

---

## Contact & Resources

**Collaboration**: AUB - Khaled Noubani

**Documentation**:
- README.md - Project overview
- data_merge.ipynb - Data merging workflow
- app/streamlit_app.py - Deployment application

**Key Files**:
- `/data/` - Data storage
- `/notebooks/` - Analysis notebooks
- `/models/` - Trained models
- `/app/` - Streamlit deployment

---

**Last Updated**: January 2026
**Status**: Phase 3 - Data Cleaning & Preprocessing
