import pandas as pd
import matplotlib.pyplot as plt


file = r"C:\Users\umarm\OneDrive\Documents\AIM350_Fraud_Project\data\fraudTrain.csv"

data = pd.read_csv(file)


# fraud by category

category_fraud = data.groupby("category")["is_fraud"].mean() * 100

category_fraud.plot(kind="bar")

plt.title("Fraud Rate by Category")
plt.xlabel("Category")
plt.ylabel("Fraud Rate (%)")

plt.tight_layout()

plt.savefig(
    r"C:\Users\umarm\OneDrive\Documents\AIM350_Fraud_Project\figures\fraud_by_category.png"
)

plt.close()


# fraud by hour

data["trans_date_trans_time"] = pd.to_datetime(
    data["trans_date_trans_time"]
)

data["transaction_hour"] = data["trans_date_trans_time"].dt.hour

hour_fraud = data.groupby("transaction_hour")["is_fraud"].mean() * 100

hour_fraud.plot(kind="bar")

plt.title("Fraud Rate by Hour")
plt.xlabel("Hour")
plt.ylabel("Fraud Rate (%)")

plt.tight_layout()

plt.savefig(
    r"C:\Users\umarm\OneDrive\Documents\AIM350_Fraud_Project\figures\fraud_by_hour.png"
)

plt.close()


# average transaction amount

average_amount = data.groupby("is_fraud")["amt"].mean()

average_amount.plot(kind="bar")

plt.title("Average Transaction Amount")
plt.xlabel("Transaction Type")
plt.ylabel("Average Amount ($)")

plt.xticks(
    [0, 1],
    ["Legitimate", "Fraud"],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    r"C:\Users\umarm\OneDrive\Documents\AIM350_Fraud_Project\figures\average_transaction_amount.png"
)

plt.close()


print("Graphs finished")