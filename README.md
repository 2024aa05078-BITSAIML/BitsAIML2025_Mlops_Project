## Task 1: Data Acquisition & Exploratory Data Analysis

### Dataset
The Heart Disease dataset was sourced from the UCI Machine Learning Repository.
The processed Cleveland subset was used, containing 14 clinical attributes and
a binary target indicating presence or absence of heart disease.

### Data Acquisition
A Python script (`data/download_scripts/download_ucidata.py`) was used to
programmatically download the dataset to ensure reproducibility.

### Data Cleaning
- Missing values marked as `?` were converted to NaN
- Rows containing missing values were removed
- The target variable was binarized:
  - 0 → No heart disease
  - 1 → Presence of heart disease

### Exploratory Data Analysis
EDA was performed using histograms, class balance plots, and a correlation heatmap.
Key observations include:
- Mild class imbalance
- Strong correlation of `thalach`, `oldpeak`, and `exang` with target

Plots are stored in the `screenshots/` directory.
