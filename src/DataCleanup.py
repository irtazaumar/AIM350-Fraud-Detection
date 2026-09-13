import pandas as pd

df = pd.read_csv(
    r"C:\Users\umarm\OneDrive\Documents\AIM350 programming for ai\PS_20174392719_1491204439457_log.csv\PS_20174392719_1491204439457_log.csv"
)

print("Shape:")
print(df.shape)

print("\nFraud counts:")
print(df["isFraud"].value_counts())

print("\nFraud percentages:")
print(df["isFraud"].value_counts(normalize=True))

print("\nTransaction types:")
print(df["type"].value_counts())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nFraud by transaction type:")
print(pd.crosstab(df["type"], df["isFraud"]))

print("\nFraud rate by transaction type:")
print(
    df.groupby("type")["isFraud"]
      .mean()
      .sort_values(ascending=False)
)

print("\nisFlaggedFraud counts:")
print(df["isFlaggedFraud"].value_counts())

print("\nisFlaggedFraud vs actual fraud:")
print(pd.crosstab(df["isFlaggedFraud"], df["isFraud"]))