# Citation Impact Prediction - AUB Capstone Project

This repository contains the code and notebooks for predicting citation impact of academic publications using machine learning, developed in collaboration with the American University of Beirut (AUB).

## Project Overview

**Objective**: Develop accurate machine learning models to predict citation counts and identify high-impact papers using only information available at publication time.

**Key Features**:
- Classification models to identify highly-cited papers
- Regression models to predict exact citation counts
- Feature importance analysis to understand citation drivers
- Interpretable prediction framework for decision-making

## Data Setup

### Step 1: Upload Your Data Files

Place your data files in the `data/` directory:

```
citapred-prototype/
├── data/
│   ├── scopus_data.csv      # Scopus file (contains abstracts)
│   ├── scival_data.csv      # SciVal file (contains citations)
│   └── [output files will go here]
├── data_merge.ipynb         # Data merging notebook
└── README.md
```

### Step 2: Run the Data Merge Notebook

Open `data_merge.ipynb` and follow these steps:

1. **Update file paths** in the notebook to match your actual filenames
2. **Update column names** for EID and Abstract fields based on your data
3. **Run all cells** to merge the datasets

The notebook will:
- Load both Scopus and SciVal files
- Match records using the EID column
- Add abstracts from Scopus to SciVal data
- Perform data quality checks
- Save the merged dataset as `data/merged_citation_data.csv`

## Expected Data

The project expects access to:
- **10,000+ publications** from AUB researchers
- **Metadata**: Author names, h-index, publication year, venue, etc.
- **Abstracts**: Full text of paper abstracts
- **Citation counts**: Number of citations received

## Project Phases

### Phase 1: Literature Review ✓
- Review citation prediction literature
- Identify research gaps
- Establish baseline metrics

### Phase 2: Data Collection ✓
- Coordinate with AUB (Khaled Noubani)
- Obtain Scopus and SciVal data

### Phase 3: Data Cleaning & Preprocessing (Current)
- Merge Scopus and SciVal datasets
- Quality filters (encoding, language, missing fields)
- Data validation and exploration

### Phase 4: Feature Engineering (Next)
- Author features (h-index, citation counts)
- Venue features (prestige scores, historical citations)
- Text features (TF-IDF vectorization from abstracts)
- Handle missing data

### Phase 5: Model Development
- Algorithms: Logistic Regression, Random Forest, XGBoost, LightGBM
- Classification: Binary (top 25% vs rest)
- Regression: Log-transformed citation counts
- 5-fold cross-validation

### Phase 6: Visualization & Reporting
- ROC curves and confusion matrices
- Feature importance analysis
- Citation distribution analysis
- Model performance comparison

## Tools & Technologies

- **Python**: Primary programming language
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning models
- **XGBoost/LightGBM**: Gradient boosting models
- **matplotlib/seaborn**: Visualization
- **Jupyter Notebook**: Interactive development

## Getting Started

1. Clone this repository
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn jupyter openpyxl
   ```
3. Place your data files in the `data/` directory
4. Run `jupyter notebook` and open `data_merge.ipynb`
5. Follow the notebook instructions to merge your data

## Contact

This is a capstone project developed in collaboration with the American University of Beirut (AUB).

---

**Note**: This project handles sensitive research data. Ensure all data files are kept confidential and not committed to version control.
