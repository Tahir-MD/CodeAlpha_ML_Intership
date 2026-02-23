# 🏦 CodeAlpha_Credit_Scoring_Model

## 📋 Project Overview
A machine learning model that predicts an individual's creditworthiness using historical financial data. This project helps financial institutions assess the risk of loan default before approving credit.

## 🎯 Objective
Build a classification model to predict whether a person will default on a loan (binary classification: 0 = No Default, 1 = Default) using features like income, debt, payment history, and loan details.

## 📊 Dataset
- **Source:** Kaggle Credit Risk Dataset
- **Records:** 32,581 individuals
- **Features:** 12 input features
- **Target:** `loan_status` (0 = Good Credit, 1 = Default)

### Feature Description
| Column | Description |
|--------|-------------|
| `person_age` | Age of the person |
| `person_income` | Annual income |
| `person_home_ownership` | Home ownership status (RENT, OWN, MORTGAGE) |
| `person_emp_length` | Employment length in years |
| `loan_intent` | Purpose of loan |
| `loan_grade` | Loan grade |
| `loan_amnt` | Loan amount |
| `loan_int_rate` | Interest rate |
| `loan_status` | **Target:** 0 = paid, 1 = defaulted |
| `loan_percent_income` | Loan amount as percentage of income |
| `cb_person_default_on_file` | Historical default (Y/N) |
| `cb_person_cred_hist_length` | Credit history length |

## 🛠️ Technologies Used
| Library | Purpose |
|---------|---------|
| Python 3.8+ | Core programming language |
| Pandas, NumPy | Data manipulation and analysis |
| Scikit-learn | Machine learning models and preprocessing |
| Matplotlib, Seaborn | Data visualization |
| Joblib | Model saving and loading |

## 🔧 Data Preprocessing

### 1. Handling Missing Values
- Filled missing `loan_int_rate` with median
- Filled missing `person_emp_length` with median

### 2. Encoding Categorical Variables
- `person_home_ownership`
- `loan_intent`
- `loan_grade`
- `cb_person_default_on_file`

### 3. Feature Engineering
Created new features to improve model performance:
- **Debt to Income Ratio:** `loan_amnt / person_income`
- **Payment to Income Ratio:** `(loan_amnt * loan_int_rate) / (12 * 100 * person_income)`

### 4. Feature Scaling
- StandardScaler used to normalize numerical features

## 🤖 Models Implemented

### 1. Logistic Regression
- Baseline model
- Fast training and inference
- Good interpretability

### 2. Decision Tree Classifier
- Non-linear relationships
- Easy to visualize and explain

### 3. Random Forest Classifier
- Ensemble of decision trees
- Handles overfitting better
- Feature importance analysis

### 4. Tuned Random Forest
- Hyperparameter optimization using GridSearchCV
- Best performing model

## 📈 Model Performance

### Comparison Table
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.85 | 0.74 | 0.58 | 0.65 |
| Decision Tree | 0.89 | 0.76 | 0.73 | 0.74 |
| Random Forest | 0.92 | 0.86 | 0.77 | 0.81 |
| **Tuned Random Forest** | **0.93** | **0.87** | **0.79** | **0.83** |

### Key Metrics Explained
- **Accuracy:** Overall correctness of predictions
- **Precision:** When model predicts default, how often is it correct?
- **Recall:** What proportion of actual defaults did model catch?
- **F1-Score:** Harmonic mean of Precision and Recall

## 📊 Key Insights

### Top 5 Important Features
1. **Loan Interest Rate** - Most significant predictor
2. **Loan Amount** - Higher amounts increase risk
3. **Debt to Income Ratio** - Key financial health indicator
4. **Person Income** - Ability to repay
5. **Credit History Length** - Longer history usually better

### Business Implications
- **False Negatives** (predicting no default but actual default): Costly for bank
- **False Positives** (predicting default but actual payment): Lost business opportunity
- Random Forest provides best balance between catching defaults and approving good loans

## 🚀 How to Run the Project

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
