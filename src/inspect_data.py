from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent

train_file = ROOT / "data" / "fraudTrain.csv"
test_file = ROOT / "data" / "fraudTest.csv"


print("Loading data...")

train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)


print("\nTRAIN DATA")
print("Rows and columns:", train_data.shape)

print("\nColumns:")
print(train_data.columns.tolist())

print("\nMissing values:")
print(train_data.isnull().sum())

print("\nDuplicate rows:")
print(train_data.duplicated().sum())

print("\nFraud counts:")
print(train_data["is_fraud"].value_counts())

print("\nFraud percentage:")
print(train_data["is_fraud"].value_counts(normalize=True) * 100)


print("\n------------------------------")


print("\nTEST DATA")
print("Rows and columns:", test_data.shape)

print("\nMissing values:")
print(test_data.isnull().sum())

print("\nDuplicate rows:")
print(test_data.duplicated().sum())

print("\nFraud counts:")
print(test_data["is_fraud"].value_counts())

print("\nFraud percentage:")
print(test_data["is_fraud"].value_counts(normalize=True) * 100)