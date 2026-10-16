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
Some new features were created from the original data:

- `customer_age`
- `transaction_hour`
- `day_of_week`
- `is_weekend`
- `distance_miles`

The distance feature measures the distance between the customer location and merchant location.
## Features Selected

The prepared data currently contains:
- category
- state
- amt
- city_pop
- customer_age
- transaction_hour
- day_of_week
- is_weekend
- distance_miles
- is_fraud

Identifier fields such as customer names, card numbers, street addresses, and transaction IDs were not included in the processed data.
## Processed Data
The prepared files are saved inside the `processed` folder:
- `train_processed.csv`
- `test_processed.csv`
The processed CSV files are not uploaded to GitHub because dataset files are excluded through `.gitignore`.