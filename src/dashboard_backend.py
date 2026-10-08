from pathlib import Path
import sys

import joblib
import pandas as pd

# Make the evaluation code in src available
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from evaluate_saved_model import (
    RAW_INPUTS,
    FEATURES,
    THRESHOLD,
    prepare_pair,
)

@__import__("streamlit").cache_resource
def load_model():
    return joblib.load(ROOT / "models" / "random_forest.joblib")


@__import__("streamlit").cache_data
def load_transactions():
    train = pd.read_csv(
        ROOT / "data" / "fraudTrain.csv",
        usecols=RAW_INPUTS,
        dtype={"cc_num": str, "zip": str},
    )

    test = pd.read_csv(
        ROOT / "data" / "fraudTest.csv",
        dtype={"cc_num": str, "zip": str},
    )

    _, processed_test = prepare_pair(train, test)

    return test, processed_test


@__import__("streamlit").cache_data
def get_predictions():
    raw_test, processed_test = load_transactions()
    model = load_model()

    probabilities = model.predict_proba(
        processed_test[FEATURES]
    )[:, 1]

    results = raw_test[
        ["trans_date_trans_time", "amt", "category", "is_fraud"]
    ].copy()

    results["fraud_probability"] = probabilities
    results["risk_score"] = probabilities * 100
    results["prediction"] = (
        probabilities >= THRESHOLD
    ).astype(int)

    return results
