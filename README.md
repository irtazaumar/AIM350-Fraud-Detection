# Real-Time Credit Card Fraud Detection

This repository contains our AIM 350 Programming for AI term group project.

The project compares two machine learning models for detecting fraudulent credit card transactions:

- Logistic Regression
- Random Forest

The stronger model is integrated into a Streamlit application for interactive fraud prediction and transaction simulation.

## Team Members

- Muhammad Umar — Data Preparation, Preprocessing, EDA, and Visualizations
- Alyssa Lee — Logistic Regression Baseline and Evaluation
- Andrew King — Random Forest Model and Model Evaluation
- Tyler Trapani — Streamlit Application and System Integration

## Dataset

This project uses the Credit Card Transactions Fraud Detection Dataset by Kartik Shenoy from Kaggle:

https://www.kaggle.com/datasets/kartik2112/fraud-detection

The dataset contains two raw CSV files:

- `fraudTrain.csv`
- `fraudTest.csv`

Training data:
- 1,296,675 transactions
- 7,506 fraudulent transactions

Testing data:
- 555,719 transactions
- 2,145 fraudulent transactions

The target column is `is_fraud`:

- `0` = legitimate transaction
- `1` = fraudulent transaction

The raw CSV files are not stored in this repository because of their size.


## Project Structure

```text
AIM350-Fraud-Detection/
├── data/
│   └── README.md
├── figures/
├── models/
│   └── random_forest.joblib
├── src/
│   ├── inspect_data.py
│   ├── explore_data.py
│   ├── prepare_data.py
│   ├── create_visualizations.py
│   ├── logistic_regression.py
│   ├── logistic_regression_README.md
│   ├── evaluate_saved_model.py
│   └── dashboard_backend.py
├── app.py
├── .gitignore
├── README.md
└── requirements.txt

```

## Setup
#1. Clone the Repository
git clone https://github.com/irtazaumar/AIM350-Fraud-Detection.git
cd AIM350-Fraud-Detection

#2. Install the Required Python Packages
Python 3.14 was used for this project.
py -3.14 -m pip install -r requirements.txt

The project uses:
- pandas
- NumPy
- scikit-learn
- joblib
- matplotlib
- Streamlit
#Download the Dataset
Download the Credit Card Transactions Fraud Detection Dataset from Kaggle:
https://www.kaggle.com/datasets/kartik2112/fraud-detection
After downloading the dataset, place the two raw CSV files inside the data folder.
The folder should contain:
data/
├── fraudTrain.csv
├── fraudTest.csv
└── README.md

The scripts use relative project paths, so the repository can be cloned and run from another computer without changing file locations in the source code.
#Data Inspection
The raw training and testing files can be inspected using:
py -3.14 src\inspect_data.py

The inspection script reports:
- Dataset dimensions
- Available columns
- Missing values
- Duplicate rows
- Fraud counts
- Fraud percentages
The dataset was found to contain no missing values and no duplicate rows.
#Exploratory Data Analysis
Run the exploratory analysis with:
py -3.14 src\explore_data.py

The analysis examines:
- Fraud rate by merchant category
- Average transaction amount
- Median transaction amount
- Fraud rate by transaction hour
- Fraud rate by gender
- Fraud rate by state
The project also includes visualizations showing several of these patterns.
To generate the figures:
py -3.14 src\create_visualizations.py

The figures are saved in the figures folder.
Current EDA figures include:
- figures/fraud_by_category.png
- figures/fraud_by_hour.png
- figures/average_transaction_amount.png
#Base Feature Engineering and Data Preparation
The Logistic Regression model uses a processed version of the original Sparkov dataset.
Run:
py -3.14 src\prepare_data.py

This script creates the following engineered features:
- customer_age
- transaction_hour
- day_of_week
- is_weekend
- distance_miles
The distance feature represents the distance between the customer's location and the merchant's location.
The prepared dataset contains:
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
Identifier fields such as customer names, card numbers, street addresses, and transaction IDs are not included in the processed dataset.
The script creates:
data/processed/train_processed.csv
data/processed/test_processed.csv

The processed CSV files are generated locally and are not stored in the GitHub repository.
#Logistic Regression Baseline
The Logistic Regression model uses the processed training and testing files created by src/prepare_data.py.
Run:
py -3.14 src\logistic_regression.py

The script:
- Loads the processed training and testing datasets
- One-hot encodes categorical features
- Standardizes numerical features
- Trains a Logistic Regression classifier
- Uses balanced class weights to account for class imbalance
- Evaluates the model on the testing dataset
- Saves the trained model locally
- Generates a model-performance visualization
The trained model is saved locally as:
models/logistic_regression_model.joblib

The Logistic Regression model file is generated when the script is run and does not need to be stored in the repository.
#Logistic Regression Results
- Accuracy: 88.19%
- Precision: 2.38%
- Recall: 73.89%
- F1 Score: 4.61%
#Logistic Regression Confusion Matrix
- True Negatives: 488,505
- False Positives: 65,069
- False Negatives: 560
- True Positives: 1,585
The model achieved relatively high recall but produced a large number of false positives. This makes Logistic Regression useful as a baseline but less practical as the final fraud-detection model.
The Logistic Regression performance visualization is saved as:
figures/logistic_regression_metrics.png

#Random Forest Model
The final Random Forest V3 model uses the original raw Sparkov training and testing files.
Unlike the Logistic Regression model, the Random Forest does not use the 9-feature processed CSV files created by prepare_data.py.
Instead, the Random Forest evaluation pipeline performs its own model-specific feature engineering using:
data/fraudTrain.csv
data/fraudTest.csv

The saved Random Forest model is stored as:
models/random_forest.joblib

The model-specific pipeline creates a larger set of features, including:
- Transaction amount
- Customer age
- Transaction hour
- Day of week
- Weekend indicator
- Customer-to-merchant distance
- Geographic information
- Transaction date components
- Cyclical time features
- Previous transaction counts
- Previous average transaction amounts
- Previous transaction variability
- Time since previous transaction
- Previous customer-merchant activity
- Previous merchant transaction activity
The feature-engineering process only uses transactions that occurred earlier in time when constructing historical features.
#Random Forest Evaluation
Run the final Random Forest evaluation with:
py -3.14 src\evaluate_saved_model.py data\fraudTest.csv --train-data data\fraudTrain.csv

This command:
- Loads the raw training data for transaction history
- Loads the raw testing data
- Recreates the Random Forest feature set
- Loads the saved Random Forest model
- Generates fraud probabilities
- Applies the selected fraud-classification threshold
- Calculates evaluation metrics
- Displays sample scored transactions
The selected Random Forest classification threshold is approximately:
0.224289

#Random Forest Results
- Precision: 79.19%
- Recall: 80.56%
- F1 Score: 79.87%
- F2 Score: 80.28%
- ROC-AUC: 0.9841
- PR-AUC: approximately 0.8534
- Average Precision: approximately 0.8534
- Fraud-Dollar Recall: 96.75%
#Random Forest Confusion Matrix
- True Negatives: 553,120
- False Positives: 454
- False Negatives: 417
- True Positives: 1,728
The Random Forest substantially reduced false positives while also improving recall and F1 score compared with the Logistic Regression baseline.
#Model Comparison
Metric	Logistic Regression	Random Forest
Precision	2.38%	79.19%
Recall	73.89%	80.56%
F1 Score	4.61%	79.87%
True Positives	1,585	1,728
False Positives	65,069	454
False Negatives	560	417


The Logistic Regression model detected a large portion of fraudulent transactions but incorrectly flagged many legitimate transactions.
The Random Forest model achieved higher recall while dramatically improving precision and reducing false positives. Based on these results, the Random Forest was selected as the final model for the Streamlit application.
#Streamlit Application
The final Random Forest model is integrated into an interactive Streamlit application.
Run the application with:
py -3.14 -m streamlit run app.py

The Streamlit application uses:
models/random_forest.joblib
data/fraudTrain.csv
data/fraudTest.csv

The application allows the user to:
- Select a real transaction from the Sparkov test dataset
- Generate a fraud prediction
- Display a fraud risk score
- Compare the prediction with the actual dataset label
- Find an example transaction predicted as fraud
- Process transactions one at a time in a simulated live stream
- Reset the transaction simulation
- Display model performance metrics
- Display the confusion matrix
- Display the fraud-risk distribution
The dashboard reports:
- Precision
- Recall
- F1 Score
- ROC-AUC
#Reproducing the Full Project
After cloning the repository and downloading the two raw CSV files, the main project can be reproduced in the following order.
Step 1: Install Dependencies
py -3.14 -m pip install -r requirements.txt

Step 2: Inspect the Raw Data
py -3.14 src\inspect_data.py

Step 3: Run Exploratory Data Analysis
py -3.14 src\explore_data.py

Step 4: Generate EDA Figures
py -3.14 src\create_visualizations.py

Step 5: Create the Processed Logistic Regression Data
py -3.14 src\prepare_data.py

Step 6: Train and Evaluate Logistic Regression
py -3.14 src\logistic_regression.py

Step 7: Evaluate the Saved Random Forest Model
py -3.14 src\evaluate_saved_model.py data\fraudTest.csv --train-data data\fraudTrain.csv

Step 8: Run the Streamlit Application
py -3.14 -m streamlit run app.py

#Notes on Reproducibility
- Python 3.14 was used during development and testing.
- Raw Kaggle CSV files are not included in the repository because of their size.
- The user must download fraudTrain.csv and fraudTest.csv and place them inside the data folder.
- The Logistic Regression processed datasets are generated locally using src/prepare_data.py.
- Logistic Regression uses the processed 9-feature dataset.
- Random Forest V3 uses the original raw Sparkov files and performs its own model-specific feature engineering.
- The final Random Forest model is stored as models/random_forest.joblib.
- The Logistic Regression model is generated locally when src/logistic_regression.py is run.
- Project scripts use relative paths rather than computer-specific absolute paths.
- Model results were independently reproduced on another group member's computer before final submission.
Repository
GitHub Repository:
https://github.com/irtazaumar/AIM350-Fraud-Detection
Course
AIM 350 — Programming for AI
Instructor: Bruno G. Kamdem, Ph.D.
Fall 2026