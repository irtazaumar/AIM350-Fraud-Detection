# Real-Time Credit Card Fraud Detection

This is our AIM 350 Programming for AI term group project.

The goal of the project is to build a machine learning system that can identify possible fraudulent credit card transactions.

The project will compare:

- Logistic Regression
- Random Forest

# Dataset
We are using the Credit Card Transactions Fraud Detection Dataset by Kartik Shenoy from Kaggle.
Dataset link:
https://www.kaggle.com/datasets/kartik2112/fraud-detection
The dataset includes separate training and testing files.
Training data:
- 1,296,675 transactions
- 7,506 fraud transactions
Testing data:
- 555,719 transactions
- 2,145 fraud transactions

The target column is `is_fraud`.

## Data Preparation
The data was checked for missing values, duplicate rows, and class imbalance.
New features were also created:
- customer_age
- transaction_hour
- day_of_week
- is_weekend
- distance_miles

Processed data is saved in:
- `data/processed/train_processed.csv`
- `data/processed/test_processed.csv`

The CSV files are not uploaded to GitHub.
## Exploratory Data Analysis
The project includes basic analysis of:
- Fraud by merchant category
- Fraud by transaction hour
- Average transaction amount for fraud and legitimate transactions
- Fraud distribution
The visualizations are saved in the `figures` folder.

## Project Structure
```text
AIM350_Fraud_Project/
├── app/
├── archive/
├── data/
├── docs/
├── figures/
├── models/
├── notebooks/
├── src/
├── .gitignore
├── README.md
└── requirements.txt