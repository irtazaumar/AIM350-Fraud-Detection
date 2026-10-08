"""Evaluate Andrew King's saved Random Forest V3; no training is performed."""

import argparse
import importlib.util
from importlib import metadata
import os
from pathlib import Path
import subprocess
import sys


REQUIRED_LIBRARIES = {
    "pandas": "3.0.3", "numpy": "2.5.1",
    "scikit-learn": "1.9.1", "joblib": "1.6.0",
}


def ensure_dependencies():
    """Install absent or mismatched run-library versions in this interpreter."""
    needed = []
    for package, expected in REQUIRED_LIBRARIES.items():
        try:
            installed = metadata.version(package)
        except metadata.PackageNotFoundError:
            installed = None
        if installed != expected:
            needed.append(f"{package}=={expected}")
    if not needed:
        return
    print(f"Python interpreter: {sys.executable}", flush=True)
    print("Installing required model libraries: " + ", ".join(needed), flush=True)
    try:
        if importlib.util.find_spec("pip") is None:
            subprocess.run([sys.executable, "-m", "ensurepip", "--upgrade"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", *needed], check=True)
    except (subprocess.CalledProcessError, OSError) as error:
        raise RuntimeError(
            "Dependency installation failed. Check internet access and write permissions "
            "for this Python installation. Installation command: "
            + subprocess.list2cmdline([sys.executable, "-m", "pip", "install", *needed])
        ) from error
    if __name__ == "__main__":
        # Restart so newly installed versions, including compiled libraries, load cleanly.
        os.execv(sys.executable, [sys.executable, *sys.argv])
    raise RuntimeError("Required libraries were installed. Restart your Python or Streamlit "
                       "process, then import evaluate_saved_model again.")


ensure_dependencies()

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc, average_precision_score, confusion_matrix, f1_score, fbeta_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)

TARGET = "is_fraud"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "random_forest.joblib"
THRESHOLD = 0.22428927926184028

CATEGORIES = ["category", "state", "merchant", "gender", "city", "zip", "job"]

NUMBERS = [
    "amt", "city_pop", "customer_age", "transaction_hour", "day_of_week",
    "is_weekend", "distance_miles", "lat", "long", "merch_lat", "merch_long",
    "unix_time", "transaction_year", "transaction_month", "transaction_day",
    "transaction_minute", "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    "card_prior_count", "card_prior_mean_amt", "card_prior_std_amt",
    "amt_to_card_prior_mean", "seconds_since_card_previous",
    "card_merchant_prior_count", "merchant_prior_count", "merchant_prior_mean_amt",
    "amt_to_merchant_prior_mean",
]

FEATURES = CATEGORIES + NUMBERS

RAW_INPUTS = ["trans_date_trans_time", "cc_num", "merchant", "category", "amt",
              "gender", "city", "state", "zip", "lat", "long", "city_pop",
              "job", "dob", "unix_time", "merch_lat", "merch_long"]


def base_features(raw):
    dates = pd.to_datetime(raw["trans_date_trans_time"], errors="raise")
    births = pd.to_datetime(raw["dob"], errors="raise")
    result = raw[CATEGORIES + ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long", "unix_time"]].copy()
    result["customer_age"] = ((dates - births).dt.days / 365.25).astype(int)
    result["transaction_hour"] = dates.dt.hour
    result["day_of_week"] = dates.dt.dayofweek
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    lat, lon, mlat, mlon = [np.radians(raw[name]) for name in ["lat", "long", "merch_lat", "merch_long"]]
    a = np.sin((mlat - lat) / 2) ** 2 + np.cos(lat) * np.cos(mlat) * np.sin((mlon - lon) / 2) ** 2
    result["distance_miles"] = 3958.8 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    for name, values in [("transaction_year", dates.dt.year), ("transaction_month", dates.dt.month),
                         ("transaction_day", dates.dt.day), ("transaction_minute", dates.dt.minute)]:
        result[name] = values
    hour_angle = 2 * np.pi * (dates.dt.hour + dates.dt.minute / 60) / 24
    weekday_angle = 2 * np.pi * dates.dt.dayofweek / 7
    result["hour_sin"], result["hour_cos"] = np.sin(hour_angle), np.cos(hour_angle)
    result["weekday_sin"], result["weekday_cos"] = np.sin(weekday_angle), np.cos(weekday_angle)
    return result


def history_features(raw):
    """Only prior timestamps contribute; target labels and identity text are never read."""
    history = raw[["cc_num", "merchant", "amt", "trans_date_trans_time"]].copy()
    history["_time"] = pd.to_datetime(history.pop("trans_date_trans_time"))
    history = history.sort_values("_time", kind="stable")
    result = pd.DataFrame(index=history.index)
    for key, prefix in [("cc_num", "card"), ("merchant", "merchant")]:
        group = history.groupby(key, sort=False)
        ties = [history[key], history["_time"]]
        # Equal-time records cannot see each other, regardless of file row order.
        count = group.cumcount().groupby(ties, sort=False).transform("first")
        total = (group["amt"].cumsum() - history["amt"]).groupby(ties, sort=False).transform("first")
        mean = total.div(count.replace(0, np.nan)).fillna(0)
        result[f"{prefix}_prior_count"] = count
        result[f"{prefix}_prior_mean_amt"] = mean
        result[f"amt_to_{prefix}_prior_mean"] = history["amt"].div(mean.replace(0, np.nan)).fillna(-1)
        if prefix == "card":
            squared = history["amt"].pow(2).groupby(history[key], sort=False).cumsum()
            prior_squared = (squared - history["amt"].pow(2)).groupby(ties, sort=False).transform("first")
            variance = prior_squared.div(count.replace(0, np.nan)) - mean.pow(2)
            result["card_prior_std_amt"] = np.sqrt(variance.clip(lower=0).fillna(0))
            gaps = group["_time"].diff().dt.total_seconds()
            result["seconds_since_card_previous"] = gaps.groupby(ties, sort=False).transform("first").fillna(-1)
    pair_count = history.groupby(["cc_num", "merchant"], sort=False).cumcount()
    result["card_merchant_prior_count"] = pair_count.groupby(
        [history["cc_num"], history["merchant"], history["_time"]], sort=False,
    ).transform("first")
    return result.sort_index()


def prepare_pair(train_raw, test_raw):
    if pd.to_datetime(train_raw["trans_date_trans_time"]).max() >= pd.to_datetime(test_raw["trans_date_trans_time"]).min():
        raise ValueError("This history protocol requires the test file to follow the training period.")
    # Prior unlabeled transactions are observable; no test targets enter engineering.
    combined = pd.concat([train_raw[RAW_INPUTS], test_raw[RAW_INPUTS]], ignore_index=True)
    history = history_features(combined)
    outputs = []
    for raw, start in [(train_raw, 0), (test_raw, len(train_raw))]:
        engineered = base_features(raw).reset_index(drop=True)
        past = history.iloc[start:start + len(raw)].reset_index(drop=True)
        engineered = pd.concat([engineered, past], axis=1)[FEATURES]
        if engineered.isna().any().any() or not np.isfinite(engineered[NUMBERS].to_numpy()).all():
            raise ValueError("Feature engineering produced invalid values.")
        outputs.append(engineered)
    return outputs


def evaluate(y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    return {
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(auc(recall, precision)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
        "false_positive_rate": float(fp / (fp + tn)),
        "threshold": float(threshold),
    }


def score(data, probabilities, threshold=0.5):
    scores = evaluate(data[TARGET], probabilities, threshold)
    scores["f2"] = float(fbeta_score(data[TARGET], probabilities >= threshold, beta=2, zero_division=0))
    fraud_amount = data.loc[data[TARGET] == 1, "amt"].sum()
    caught_amount = data.loc[(data[TARGET] == 1) & (probabilities >= threshold), "amt"].sum()
    scores["fraud_dollar_recall"] = float(caught_amount / fraud_amount)
    return scores

def evaluate_model(test_csv, training_csv=None, rows=0):
    """Return (metrics, predictions) from a prepared CSV or an original raw pair.

    training_csv is needed only for raw fraudTest.csv, to construct strict-past
    history. Labels are used for scoring only. rows=0 evaluates the full test.
    """
    if not isinstance(rows, int) or rows < 0:
        raise ValueError("rows must be a nonnegative integer; use 0 for all rows.")
    test_csv = Path(test_csv).expanduser()
    if training_csv is not None:
        # Full raw history is retained even when only a preview is scored.
        test = pd.read_csv(test_csv, dtype={"cc_num": str, "zip": str})
        train = pd.read_csv(Path(training_csv).expanduser(), usecols=RAW_INPUTS,
                            dtype={"cc_num": str, "zip": str})
        missing = sorted(set(RAW_INPUTS + [TARGET]) - set(test.columns))
        if missing:
            raise ValueError(f"Raw test CSV is missing fields: {missing}")
        if train[RAW_INPUTS].isna().any().any() or test[RAW_INPUTS + [TARGET]].isna().any().any():
            raise ValueError("Raw CSVs contain missing input values or labels.")
        _, data = prepare_pair(train, test)
        data[TARGET] = test[TARGET].to_numpy()
        if rows:
            data = data.head(rows).copy()
    else:
        data = pd.read_csv(test_csv, dtype={"zip": str}, nrows=rows or None)
    missing = sorted(set(FEATURES + [TARGET]) - set(data.columns))
    if missing:
        raise ValueError(f"Prepared CSV is missing fields: {missing}. "
                         "For original fraudTest.csv, also provide --train-data fraudTrain.csv.")
    if data[FEATURES + [TARGET]].isna().any().any() or not np.isfinite(data[NUMBERS].to_numpy()).all():
        raise ValueError("Test inputs contain missing or non-finite values.")
    if set(data[TARGET].unique()) != {0, 1}:
        raise ValueError("Evaluation needs both legitimate (0) and fraud (1) labels. "
                         "Use the full test file or a larger preview.")
    model = joblib.load(MODEL_PATH)
    if list(model.feature_names_in_) != FEATURES:
        raise ValueError("The saved model's feature list does not match this V3 evaluator.")
    probabilities = model.predict_proba(data[FEATURES])[:, 1]
    metrics = {"rows": len(data), "fraud_rows": int(data[TARGET].sum()),
               **score(data, probabilities, THRESHOLD)}
    predictions = data[["amt", "transaction_hour", TARGET]].reset_index(drop=True).copy()
    predictions.insert(0, "test_csv_row", np.arange(len(data)))
    predictions["fraud_probability"] = probabilities
    predictions["fraud_risk_score"] = probabilities * 100
    predictions["prediction"] = (probabilities >= THRESHOLD).astype(int)
    return metrics, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_csv", type=Path, help="Prepared V3 testing CSV, or original fraudTest.csv.")
    parser.add_argument("--train-data", type=Path, help="Original fraudTrain.csv; required for raw test data.")
    parser.add_argument("--rows", type=int, default=0, help="0 scores all rows; positive values score a labeled preview.")
    args = parser.parse_args()
    try:
        metrics, predictions = evaluate_model(args.test_csv, args.train_data, args.rows)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    if args.rows:
        print("PREVIEW: results describe only the selected test rows.")
    print(f"Real test rows evaluated: {metrics['rows']:,}; fraud: {metrics['fraud_rows']:,}")
    print(f"Saved-model cutoff: {THRESHOLD:.12f}")
    for name in ["precision", "recall", "f1", "f2", "roc_auc", "pr_auc", "average_precision", "fraud_dollar_recall"]:
        print(f"{name}: {metrics[name]:.6f}")
    print(f"Confusion counts: TN={metrics['true_negatives']:,}, FP={metrics['false_positives']:,}, "
          f"FN={metrics['false_negatives']:,}, TP={metrics['true_positives']:,}")
    print("\nExample scored transactions:")
    print(predictions.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
