# Dataset

This project uses the Credit Card Transactions Fraud Detection Dataset by Kartik Shenoy from Kaggle.

Dataset link:

https://www.kaggle.com/datasets/kartik2112/fraud-detection

## Raw Data Files

The dataset contains two original CSV files:

- `fraudTrain.csv`
- `fraudTest.csv`

Place both files directly inside the `data` folder:

```text
data/fraudTrain.csv
data/fraudTest.csv
```

The training file contains 1,296,675 transactions.
The test file contains 555,719 transactions.

The target column is `is_fraud`.
- 0 = legitimate transaction
- 1 = fraudulent transaction

The dataset is highly imbalanced because fraud makes up less than 1% of the transactions.
## Data Preparation

The original data was inspected for:
- Missing values
- Duplicate rows
- Fraud distribution
- Available columns
No missing values or duplicate rows were found.

No missing values or duplicate rows were found.

The inspection script can be run with:

```powershell
py -3.14 src\inspect_data.py
```

## Exploratory Data Analysis

Exploratory data analysis was performed on the training dataset to identify patterns related to fraudulent transactions.

The analysis includes:

- Fraud rate by merchant category
- Fraud rate by transaction hour
- Average and median transaction amounts
- Fraud patterns by gender
- Fraud patterns by state

Run the analysis:

```powershell
py -3.14 src\explore_data.py
```

## Data Visualizations

Three graphs were created to show the main findings:

- Fraud rate by merchant category
- Fraud rate by transaction hour
- Average transaction amount for legitimate and fraudulent transactions

Run:

```powershell
py -3.14 src\create_visualizations.py
```

The graphs are saved in the `figures` folder:

- `fraud_by_category.png`
- `fraud_by_hour.png`
- `average_transaction_amount.png`

## Logistic Regression Data Preparation

The Logistic Regression model uses processed data created from the original Sparkov dataset.

The following features were selected:

- `category`
- `state`
- `amt`
- `city_pop`
- `customer_age`
- `transaction_hour`
- `day_of_week`
- `is_weekend`
- `distance_miles`

The target variable is `is_fraud`.

Some of these features were created from the original data, including customer age, transaction hour, day of week, weekend status, and distance between the customer and merchant.

Personal information and identifiers such as names, card numbers, street addresses, and transaction IDs were excluded from the processed files.

### Generate Processed Data

Run this command from the main project folder:

```powershell
py -3.14 src\prepare_data.py
```

This creates two files:

```text
data/processed/train_processed.csv
data/processed/test_processed.csv
```

The processed files contain nine input features and the fraud label.

Logistic Regression uses these processed files for training and testing.

The processed CSV files are not uploaded to GitHub because they can be recreated using `prepare_data.py`.


## Random Forest Data Preparation

The Random Forest V3 model uses the original Sparkov dataset files instead of the processed files used by Logistic Regression.

The required files are:

- `data/fraudTrain.csv`
- `data/fraudTest.csv`

Andrew's Random Forest model uses a larger set of 36 features, including transaction information, time-related features, location data, and previous transaction history.

The evaluation script creates these features from the original CSV files. The saved Random Forest model is then used to make predictions.

### Run Random Forest Evaluation

Make sure the trained model is saved in:

```text
models/random_forest.joblib
```

Run this command from the main project folder:

```powershell
py -3.14 src\evaluate_saved_model.py data\fraudTest.csv --train-data data\fraudTrain.csv
```

Both raw CSV files are required because the evaluation script uses earlier transactions to calculate historical features.

The script loads the existing trained model and evaluates it on the test data. It does not retrain the Random Forest model.

### Expected Results

The full test dataset contains 555,719 transactions.

The Random Forest V3 model produced:

- Precision: 79.19%
- Recall: 80.56%
- F1 Score: 79.87%
- ROC-AUC: 0.9841
- PR-AUC: 0.8534

These results were reproduced using the original training and testing files.

## Data File Notes

- Raw CSV files are downloaded separately from Kaggle.
- Processed Logistic Regression files are generated using `src/prepare_data.py`.
- Random Forest V3 uses the original CSV files and performs additional feature engineering.
- Dataset CSV files are excluded from GitHub because of their size.
