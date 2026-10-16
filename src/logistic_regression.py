from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent


# Load the processed training and test data
train_data = pd.read_csv(
    ROOT / "data" / "processed" / "train_processed.csv"
)

test_data = pd.read_csv(
    ROOT / "data" / "processed" / "test_processed.csv"
)


print("Training data shape:", train_data.shape)
print("Test data shape:", test_data.shape)

print("\nTraining columns:")
print(train_data.columns.tolist())

print("\nFraud counts:")
print(train_data["is_fraud"].value_counts())


# Separate the features from the target
X_train = train_data.drop("is_fraud", axis=1)
y_train = train_data["is_fraud"]

X_test = test_data.drop("is_fraud", axis=1)
y_test = test_data["is_fraud"]


# Text columns
categorical_features = [
    "category",
    "state",
]


# Number columns
numeric_features = [
    "amt",
    "city_pop",
    "customer_age",
    "transaction_hour",
    "day_of_week",
    "is_weekend",
    "distance_miles",
]


# Prepare the data for Logistic Regression
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "num",
            StandardScaler(),
            numeric_features,
        ),
    ]
)


# Create the Logistic Regression model
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
            ),
        ),
    ]
)


print("\nTraining Logistic Regression model...")

# Train the model
model.fit(X_train, y_train)

print("Training complete.")


# Save the trained model
models_dir = ROOT / "models"
models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    model,
    models_dir / "logistic_regression_model.joblib",
)

print("Model saved.")


# Make predictions using the test data
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)


print("\nLOGISTIC REGRESSION RESULTS")
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1 Score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))
print("PR-AUC:", round(pr_auc, 4))


print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, y_pred))


# Graph the evaluation results
metrics = [
    "Precision",
    "Recall",
    "F1 Score",
]

scores = [
    precision,
    recall,
    f1,
]

plt.bar(metrics, scores)
plt.title("Logistic Regression Performance")
plt.ylabel("Score")
plt.ylim(0, 1)

figures_dir = ROOT / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

plt.savefig(
    figures_dir / "logistic_regression_metrics.png"
)

plt.show()