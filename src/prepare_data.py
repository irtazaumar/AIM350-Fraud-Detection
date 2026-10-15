from pathlib import Path
import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parent.parent

train_file = ROOT / "data" / "fraudTrain.csv"
test_file = ROOT / "data" / "fraudTest.csv"


print("...")
train_data = pd.read_csv(train_file)
test_data = pd.read_csv(test_file)

# convert date columns
train_data["trans_date_trans_time"] = pd.to_datetime(
    train_data["trans_date_trans_time"]
)
test_data["trans_date_trans_time"] = pd.to_datetime(
    test_data["trans_date_trans_time"]
)
train_data["dob"] = pd.to_datetime(train_data["dob"])
test_data["dob"] = pd.to_datetime(test_data["dob"])

# transaction hour
train_data["transaction_hour"] = train_data["trans_date_trans_time"].dt.hour
test_data["transaction_hour"] = test_data["trans_date_trans_time"].dt.hour

# day of week
train_data["day_of_week"] = train_data["trans_date_trans_time"].dt.dayofweek
test_data["day_of_week"] = test_data["trans_date_trans_time"].dt.dayofweek

# weekend
train_data["is_weekend"] = (
    train_data["day_of_week"] >= 5
).astype(int)
test_data["is_weekend"] = (
    test_data["day_of_week"] >= 5
).astype(int)

# customer age
train_data["customer_age"] = (
    (train_data["trans_date_trans_time"] - train_data["dob"]).dt.days
    / 365.25
).astype(int)
test_data["customer_age"] = (
    (test_data["trans_date_trans_time"] - test_data["dob"]).dt.days
    / 365.25
).astype(int)

# distance between customer and merchant
train_lat = np.radians(train_data["lat"])
train_long = np.radians(train_data["long"])
train_merch_lat = np.radians(train_data["merch_lat"])
train_merch_long = np.radians(train_data["merch_long"])

train_lat_diff = train_merch_lat - train_lat
train_long_diff = train_merch_long - train_long

train_a = (
    np.sin(train_lat_diff / 2) ** 2
    + np.cos(train_lat)
    * np.cos(train_merch_lat)
    * np.sin(train_long_diff / 2) ** 2
)

train_c = 2 * np.arcsin(np.sqrt(train_a))

train_data["distance_miles"] = 3958.8 * train_c

test_lat = np.radians(test_data["lat"])
test_long = np.radians(test_data["long"])
test_merch_lat = np.radians(test_data["merch_lat"])
test_merch_long = np.radians(test_data["merch_long"])

test_lat_diff = test_merch_lat - test_lat
test_long_diff = test_merch_long - test_long

test_a = (
    np.sin(test_lat_diff / 2) ** 2
    + np.cos(test_lat)
    * np.cos(test_merch_lat)
    * np.sin(test_long_diff / 2) ** 2
)

test_c = 2 * np.arcsin(np.sqrt(test_a))

test_data["distance_miles"] = 3958.8 * test_c

# features we plan to use
features = [
    "category",
    "state",
    "amt",
    "city_pop",
    "customer_age",
    "transaction_hour",
    "day_of_week",
    "is_weekend",
    "distance_miles"
]


X_train = train_data[features]
y_train = train_data["is_fraud"]
X_test = test_data[features]
y_test = test_data["is_fraud"]


print("\nFEATURES")
print(features)
print("\nTRAINING DATA SHAPE")
print(X_train.shape)
print("\nTEST DATA SHAPE")
print(X_test.shape)
print("\nFIRST 5 TRAINING ROWS")
print(X_train.head())
print("\nTRAINING TARGET COUNTS")
print(y_train.value_counts())
print("\nTEST TARGET COUNTS")
print(y_test.value_counts())


# save the prepared data

train_ready = train_data[features + ["is_fraud"]]
test_ready = test_data[features + ["is_fraud"]]


processed_dir = ROOT / "data" / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

train_ready.to_csv(
    processed_dir / "train_processed.csv",
    index=False
)

test_ready.to_csv(
    processed_dir / "test_processed.csv",
    index=False
)


print("\nProcessed files saved.")