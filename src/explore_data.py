import pandas as pd


train_file = r"C:\Users\umarm\OneDrive\Documents\AIM350_Fraud_Project\data\fraudTrain.csv"

data = pd.read_csv(train_file)


print("DATA SIZE")
print(data.shape)


print("\nFRAUD BY CATEGORY")
category_fraud = data.groupby("category")["is_fraud"].mean() * 100
print(category_fraud.sort_values(ascending=False))


print("\nAVERAGE TRANSACTION AMOUNT")
print(data.groupby("is_fraud")["amt"].mean())


print("\nMEDIAN TRANSACTION AMOUNT")
print(data.groupby("is_fraud")["amt"].median())


# convert transaction date/time
data["trans_date_trans_time"] = pd.to_datetime(
    data["trans_date_trans_time"]
)

# get hour from transaction time
data["transaction_hour"] = data["trans_date_trans_time"].dt.hour


print("\nFRAUD BY TRANSACTION HOUR")
hour_fraud = data.groupby("transaction_hour")["is_fraud"].mean() * 100
print(hour_fraud)


print("\nFRAUD BY GENDER")
gender_fraud = data.groupby("gender")["is_fraud"].mean() * 100
print(gender_fraud)


print("\nFRAUD BY STATE")
state_fraud = data.groupby("state")["is_fraud"].agg(["count", "sum", "mean"])

state_fraud["mean"] = state_fraud["mean"] * 100

print(
    state_fraud.sort_values(
        "mean",
        ascending=False
    ).head(15)
)