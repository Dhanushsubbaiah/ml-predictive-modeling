# Predictive Modeling 

This repository contains two Python scripts that implement regression and classification models using supervised learning techniques in Python.

## Files
- **`regression.py`** — Performs regression analysis using:
  - Least Squares (Polynomial Regression)
  - Custom Gradient Descent
  - LASSO Regression (automatic λ selection)

- **`classification.py`** — Performs text classification using:
  - Logistic Regression
  - Multinomial Naïve Bayes
  - Basic NLP preprocessing and Bag-of-Words features

## How to Run
1. Install dependencies:
   ```bash
   pip install pandas scikit-learn matplotlib numpy
Run the scripts:
   ```bash
  python regression.py
  python classification.py
```

Notes: Datasets are not included. Add your own GasProperties.csv and emails.csv files to the same directory to run the code.
