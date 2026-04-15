# Loan Interest Rate Prediction

End-to-end regression project for predicting offered loan interest rates from borrower, loan, and macroeconomic signals.

## Project Objective

Build a production-style machine learning workflow that:
- starts from intentionally messy lending data
- performs robust cleaning and feature engineering
- compares baseline and non-linear models
- evaluates performance with interpretable business learnings

Target:
- interest_rate_offered (continuous regression target)

Domain:
- Retail lending and credit risk pricing

Dataset:
- 5,000 synthetic loan applications with realistic quality issues

## Repository Structure

- data/raw/loan_applications.csv
- data/processed/loan_applications_engineered.csv
- notebooks/01_eda.ipynb
- notebooks/02_feature_eng.ipynb
- notebooks/03_modeling.ipynb
- src/pipeline.py
- models/best_random_forest_pipeline.joblib
- reports/final_model_metrics.csv
- reports/test_error_analysis.csv
- reports/figures
- reports/num_figures
- reports/cat_figures

## End-to-End Workflow

### 1) Exploratory Data Analysis

Notebook:
- notebooks/01_eda.ipynb

What was done:
- profile of numeric and categorical columns
- missing value checks
- duplicate detection
- univariate plots (histograms and boxplots)
- skewness and kurtosis diagnostics
- bivariate and multivariate analysis:
  - correlation heatmap
  - categorical vs numeric boxplots
  - categorical vs categorical cross-tab heatmap

Key findings:
- strong skew in annual_income, loan_amount, existing_debt, dti_ratio
- multicollinearity signal between origination_year and fed_funds_rate
- category label inconsistencies in employment_type and loan_purpose
- heavy-tail behavior in debt and ratio variables

Generated EDA assets:
- reports/num_figures
- reports/cat_figures

### 2) Feature Engineering and Data Quality Cleaning

Notebook:
- notebooks/02_feature_eng.ipynb

Reusable pipeline implementation:
- src/pipeline.py

Data quality actions:
- duplicate removal (excluding borrower_id from duplication key)
- categorical standardization:
  - capitalization normalization
  - typo and formatting cleanup
- DTI clipping for extreme outliers
- imputation strategy:
  - annual_income by group median (employment_type)
  - credit_score via KNN imputation
- log transforms for skewed monetary features
- categorical encoding for model-ready matrix

Engineered features in pipeline:
- loan_to_income_ratio
- debt_to_loan_ratio
- monthly_payment_est
- payment_to_income
- credit_tier
- log_annual_income
- log_loan_amount
- log_existing_debt
- is_stable_employment
- has_late_payments
- term_bucket
- rate_spread

Important leakage control for modeling:
- rate_spread is removed before training because it is derived from the target.

### 3) Modeling and Evaluation

Notebook:
- notebooks/03_modeling.ipynb

Models compared:
- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Tuned Random Forest (RandomizedSearchCV)

Evaluation:
- RMSE
- MAE
- R2
- residual diagnostics
- feature importance (tree-based and coefficient magnitude views)

Residual and importance plots:
- reports/figures/Linear Regression_residuals.png
- reports/figures/Ridge Regression_residuals.png
- reports/figures/Random Forest_residuals.png
- reports/figures/feature_importance.png

## Results

Baseline comparison (holdout test from notebook run):
- Random Forest: RMSE 0.9665, MAE 0.9341, R2 0.9112
- Linear Regression: RMSE 1.0583, MAE 1.1200, R2 0.8935
- Ridge Regression: RMSE 1.0583, MAE 1.1200, R2 0.8935

Final tuned model metrics:
Source: reports/final_model_metrics.csv

- Model: Tuned Random Forest
- Test RMSE: 0.9598
- Test MAE: 0.5148
- Test R2: 0.9124
- Best CV RMSE: 0.8595

Saved final model:
- models/best_random_forest_pipeline.joblib

Saved error analysis:
- reports/test_error_analysis.csv

## Goals Achieved

- completed full pipeline from raw to model artifact
- handled realistic data quality issues before modeling
- prevented obvious leakage during training
- compared linear and non-linear approaches
- selected a strong final model with reproducible outputs

## Learnings and Takeaways

- data cleaning quality directly improved model stability and interpretability
- non-linear models captured lending behavior better than linear baselines
- skewed financial variables benefited from transformation and clipping
- group-aware and KNN imputation were more robust than naive global fills
- leakage checks are essential in feature engineering for trustworthy metrics
- reporting artifacts (figures, metrics CSV, saved model) make the project portfolio-ready and reproducible

## How To Run

1. Install dependencies.

   pip install -r requirements.txt

2. Run the preprocessing pipeline.

   python src/pipeline.py

3. Run notebooks in order.

- notebooks/01_eda.ipynb
- notebooks/02_feature_eng.ipynb
- notebooks/03_modeling.ipynb

## Notes

- Dataset is synthetic and intended for educational and portfolio use.
- No real borrower personal data is included.