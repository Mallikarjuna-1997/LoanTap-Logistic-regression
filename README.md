# LoanTap Logistic Regression Case Study

## Overview
This project is focused on predicting whether a loan applicant will default on a loan using logistic regression. The dataset contains detailed loan information, including loan amount, interest rate, employment status, income, credit lines, and more.

The primary goal is to use logistic regression to classify whether a borrower will repay or default on the loan based on their attributes.

## Dataset
The dataset contains **396,030 rows** and **27 columns**. It includes both numerical and categorical data, covering attributes like:
- Loan Amount (`loan_amnt`)
- Interest Rate (`int_rate`)
- Employment Title (`emp_title`)
- Home Ownership (`home_ownership`)
- Annual Income (`annual_inc`)
- Loan Status (`loan_status` - Target Variable)

**Target Variable**: The loan status is binary with two classes:
- **Fully Paid**: Borrower has fully repaid the loan.
- **Charged Off**: Borrower defaulted on the loan.

## Steps Involved

### 1. Data Preprocessing
- Handled missing values using techniques like KNN imputation for numerical columns.
- Converted categorical variables into numerical form using techniques like **Label Encoding** and **Target Encoding**.
- Created new features such as credit age, derived from loan issue dates and earliest credit line dates.
- Removed high cardinality features like `address`, `title`, and other unnecessary columns.

### 2. Feature Engineering
- Added features such as `credit_age_years` and `issue_year`.
- Identified and treated outliers using **IQR method** to enhance model performance.
- Standardized numerical features using **StandardScaler** for better logistic regression performance.

### 3. Model Building
- Built multiple logistic regression models:
  - **Base Logistic Regression** without handling class imbalance.
  - **Logistic Regression with Class Weights** to handle class imbalance.
  - **Logistic Regression with SMOTE (Synthetic Minority Over-sampling Technique)** to handle severe class imbalance by generating synthetic data for the minority class.

### 4. Model Evaluation
- Evaluated models using accuracy, precision, recall, F1-score, and confusion matrix.
- Noted significant class imbalance: majority of loans were fully paid, making it important to balance the dataset.
- The final model using **SMOTE** achieved a balanced performance across both classes with an accuracy of **63.33%** on the resampled dataset.

## Key Findings
- There was significant class imbalance, with many more fully paid loans than defaulted ones.
- Features like `loan_amnt`, `annual_inc`, and `total_acc` had the highest impact on predicting loan defaults.
- After applying SMOTE, the model was able to balance precision and recall for both loan approval and loan defaults.

## Conclusion
The logistic regression model, combined with techniques like SMOTE and class weighting, was effective in predicting loan defaults. However, improvements could be made by using more advanced models or additional feature engineering.

## Dependencies
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn (for SMOTE)

## How to Run
1. Clone this repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook to see the entire workflow.

## Future Work
- Explore more complex models like **Random Forest** or **Gradient Boosting**.
- Further investigate and treat outliers.
- Implement additional feature engineering, especially in terms of loan purpose and repayment behavior.
