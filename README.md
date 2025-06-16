AI Job Salary Predictor 2025

This project builds a machine learning model to predict the salaries of AI-related jobs using features like job title, location, experience level, remote ratio, and required skills. The entire pipeline includes data cleaning, feature engineering, model training, evaluation, and deployment using Streamlit.

Dataset: [Global AI Job Market and Salary Trends 2025 (Kaggle)](https://www.kaggle.com/datasets/bismasajjad/global-ai-job-market-and-salary-trends-2025/data)

---

## 1. Exploratory Data Analysis (EDA)

We explore the dataset to understand its structure and content:
- Checked shape and data types
- Visualized salary distribution
- Detected outliers with boxplot
- Counted unique values in categorical columns
- Plotted correlation heatmap for numerical features

---

## 2. Data Preparation

### 2.1 Data Cleaning
- Dropped irrelevant columns (IDs, currency)
- Removed duplicate rows
- Converted `posting_date` to datetime format

### 2.2 Feature Engineering
- Encoded categorical features:
  - Label Encoding (for tree-based models)
  - One-Hot Encoding (for linear models)
- Handled `required_skills`:
  - Created `skill_count` feature
  - Multi-hot encoded skills using `MultiLabelBinarizer`
- Created new features:
  - `job_country_match`
  - `is_junior`
  - `days_since_posted`

### 2.3 Feature Transform and Analysis
- Log-transformed salary to reduce skewness
- Scaled features (for linear regression)
- Used Lasso for feature selection (optional)

---

## 3. Modeling

### 3.1 Train/Test Split
- Separated data for XGBoost, Random Forest, and Linear Regression

### 3.2 Models Trained
- XGBoost
- Random Forest
- Linear Regression

### 3.3 Evaluation
- Compared model performance using MAE, RMSE, and R²

---

## 4. Evaluation

- Plotted **actual vs predicted salaries** (log scale)
- Analyzed **residuals** to check model bias
- Performed **5-fold cross-validated R²** (Linear R² ≈ 0.92)
- Visualized **feature importance** from Linear Regression

---

## 5. Deployment

### 5.1 Save Model and Encoders
Used `joblib` to save:
- Trained model
- OneHot encoder
- MultiLabel encoder
- Feature column list

### 5.2 Build Streamlit App
Created a UI for users to input job info and receive real-time salary predictions.

### 5.3 Streamlit Interface
Final app interface allows users to:
- Select job title, experience level, etc.
- See predicted salary instantly

---

## Result

- Successfully created an interactive salary prediction tool
- Achieved R² score over 0.92 on log-transformed salaries
- Ready for deployment and real-world usage

---

## Tech Stack

- Python (pandas, scikit-learn, XGBoost, matplotlib, seaborn)
- Streamlit
- Joblib
- Jupyter Notebook

---

## Author
Akkharaphon Xu
