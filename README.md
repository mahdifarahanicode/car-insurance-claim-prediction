# Car Insurance Claim Prediction

## Overview
This project aims to predict whether a car insurance policyholder will file a claim (`OUTCOME = 1`) or not (`OUTCOME = 0`).  
Two machine learning models are compared: **Decision Tree** and **Support Vector Machine (SVM)**.  
The dataset is preprocessed, balanced, and evaluated with accuracy, confusion matrix, and classification report.

## Dataset
The dataset contains information about policyholders, including:
- Demographics (age, gender, education, income, marital status, children)
- Driving behaviour (experience, annual mileage, speeding violations, DUIs)
- Vehicle details (ownership, year, type, postal code)
- Credit score, past accidents, and claim outcome.

## Project Steps
1. **Data Loading & Cleaning**  
   - Removed `ID` and `RACE` columns.  
   - Checked for missing values.

2. **Exploratory Data Analysis**  
   - Identified numeric vs. categorical variables.  
   - Checked class balance: dataset was imbalanced (31.33% claims, 68.67% no claims).

3. **Outlier Removal**  
   - Used IQR (boxplot) method. Removed rows with outliers in any numeric feature (except `OUTCOME`).  
   - Removed ~1298 rows, leaving 8702 records.

4. **Balancing**  
   - Downsampled majority class to match minority class → balanced dataset of 5740 rows (50% each class).

5. **Encoding**  
   - One-hot encoding for categorical variables → 23 features.

6. **Train/Test Split**  
   - 75% training, 25% test, stratified by outcome.

7. **Model Training & Evaluation**  
   - **Decision Tree** (default settings)  
   - **Linear SVM** (after scaling features)

8. **Feature Importance** (from Decision Tree)  
   - Top features: `CREDIT_SCORE`, `PAST_ACCIDENTS`, `VEHICLE_OWNERSHIP`, `ANNUAL_MILEAGE`, `VEHICLE_YEAR_before 2015`.

9. **Hyperparameter Tuning (SVM)**  
   - Grid search over `C`, `gamma`, `kernel` (linear, rbf).  
   - Best parameters: `C=100`, `gamma=0.001`, `kernel=linear`.

## Results

| Model               | Accuracy | Precision (class 1) | Recall (class 1) | F1-score |
|---------------------|----------|---------------------|------------------|-----------|
| Decision Tree       | 76.24%   | 0.76                | 0.76             | 0.76      |
| SVM (linear)        | 83.97%   | 0.82                | 0.86             | 0.84      |
| SVM (tuned)         | 83.97%   | 0.82                | 0.86             | 0.84      |

- SVM significantly outperforms Decision Tree.
- Best SVM model achieves 84% accuracy on the test set.

## Key Insights
- Past accidents, speeding violations, and DUIs are strong predictors of claims.
- Credit score and annual mileage also have moderate influence.
- The SVM model provides a reliable tool for identifying high‑risk customers.

## Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- tqdm (for grid search progress)

## Usage
Run the Jupyter Notebook `car_insurance_claim.ipynb` step by step.  
All data preprocessing, balancing, encoding, training, and evaluation are included.

## License
MIT

## Author
Mahdi Farahani
