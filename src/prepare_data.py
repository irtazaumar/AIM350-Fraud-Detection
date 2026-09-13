import pandas as pd

df = pd.read_csv(
    r"C:\Users\umarm\OneDrive\Documents\AIM350 programming for ai\PS_20174392719_1491204439457_log.csv\PS_20174392719_1491204439457_log.csv"
)

print("Original dataset shape:")
print(df.shape)

# 2. Keep only transaction types where fraud actually occurs
model_df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()

print("\nFiltered dataset shape:")
print(model_df.shape)

print("\nFraud counts in filtered dataset:")
print(model_df["isFraud"].value_counts())

print("\nFraud percentages in filtered dataset:")
print(model_df["isFraud"].value_counts(normalize=True))

# 3. Remove columns we do not want to use as model features
model_df = model_df.drop(
    columns=["nameOrig", "nameDest", "isFlaggedFraud"]
)

print("\nColumns after removing identifiers/leakage:")
print(model_df.columns)

# 4. Separate fraud and legitimate transactions
fraud_df = model_df[model_df["isFraud"] == 1]
legit_df = model_df[model_df["isFraud"] == 0]

print("\nFraud rows available:")
print(len(fraud_df))

print("\nLegitimate rows available:")
print(len(legit_df))

# 5. Create a smaller development dataset
# Keep ALL fraud rows and randomly sample 200,000 legitimate rows
legit_sample = legit_df.sample(
    n=200000,
    random_state=42
)

dev_df = pd.concat([fraud_df, legit_sample])

# Shuffle the rows
dev_df = dev_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\nDevelopment dataset shape:")
print(dev_df.shape)

print("\nDevelopment dataset fraud counts:")
print(dev_df["isFraud"].value_counts())

print("\nDevelopment dataset fraud percentages:")
print(dev_df["isFraud"].value_counts(normalize=True))

# 6. Save the development dataset
dev_df.to_csv(
    "paysim_development.csv",
    index=False
)

print("\nSaved development dataset as:")
print("paysim_development.csv")