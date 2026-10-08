import pandas as pd
import streamlit as st
from src.dashboard_backend import get_predictions

st.set_page_config(
    page_title="AIM350 Fraud Detection",
    page_icon="💳",
    layout="wide",
)

st.title("Real-Time Credit Card Fraud Detection")
st.write("AIM 350 — Machine Learning Group Project")

st.divider()

st.subheader("Transaction Fraud Prediction")
st.write(
    "Select a real transaction from the Sparkov test dataset "
    "to see its fraud prediction."
)

with st.spinner("Loading real transaction data and model..."):
    results = get_predictions()
if st.button("Find Example Fraud Prediction"):
    fraud_rows = results.index[
        results["prediction"] == 1
    ]

    if len(fraud_rows) > 0:
        st.info(
            f"Try transaction row: {fraud_rows[0]}"
        )
    else:
        st.warning("No predicted fraud rows found.")

transaction_id = st.number_input(
    "Select Transaction Row",
    min_value=0,
    max_value=len(results) - 1,
    value=0,
    step=1,
)

transaction = results.iloc[int(transaction_id)]

st.write(f"**Transaction Amount:** ${transaction['amt']:,.2f}")
st.write(f"**Merchant Category:** {transaction['category']}")
st.write(
    f"**Transaction Time:** "
    f"{transaction['trans_date_trans_time']}"
)

if st.button("Predict Fraud"):
    prediction = int(transaction["prediction"])
    risk = float(transaction["risk_score"])

    if prediction == 1:
        st.error("Fraud Detected")
    else:
        st.success("Transaction Predicted Legitimate")

    st.metric("Fraud Risk Score", f"{risk:.2f}%")

    st.write(
        "**Actual Dataset Label:** "
        + ("Fraud" if transaction["is_fraud"] == 1
           else "Legitimate")
    )
st.divider()

st.subheader("Live Transaction Simulation")

st.write(
    "Process real credit card transactions one at a time "
    "using the trained Random Forest model."
)

# Keep track of simulation progress
if "stream_position" not in st.session_state:
    st.session_state.stream_position = 0

if "stream_history" not in st.session_state:
    st.session_state.stream_history = []

col1, col2 = st.columns(2)

with col1:
    if st.button("Process Next Transaction"):
        position = st.session_state.stream_position

        if position < len(results):
            transaction = results.iloc[position]

            st.session_state.stream_history.append({
                "Transaction": position,
                "Amount ($)": float(transaction["amt"]),
                "Fraud Risk (%)": round(
                    float(transaction["risk_score"]), 2
                ),
                "Prediction": (
                    "Fraud"
                    if int(transaction["prediction"]) == 1
                    else "Legitimate"
                ),
            })

            st.session_state.stream_position += 1
        else:
            st.warning("All transactions have been processed.")

with col2:
    if st.button("Reset Simulation"):
        st.session_state.stream_position = 0
        st.session_state.stream_history = []

history = st.session_state.stream_history

st.metric("Transactions Processed", len(history))

if history:
    st.dataframe(
        pd.DataFrame(history),
        use_container_width=True,
        hide_index=True,
    )
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

st.divider()

st.subheader("Random Forest Model Performance")

y_true = results["is_fraud"]
y_pred = results["prediction"]
y_prob = results["fraud_probability"]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_prob)

m1, m2, m3, m4 = st.columns(4)

m1.metric("Precision", f"{precision:.2%}")
m2.metric("Recall", f"{recall:.2%}")
m3.metric("F1 Score", f"{f1:.2%}")
m4.metric("ROC-AUC", f"{roc_auc:.4f}")

st.write("### Confusion Matrix")

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1],
)

matrix_df = pd.DataFrame(
    matrix,
    index=["Actual Legitimate", "Actual Fraud"],
    columns=["Predicted Legitimate", "Predicted Fraud"],
)

st.dataframe(
    matrix_df,
    use_container_width=True,
)

st.write("### Fraud Risk Distribution")

st.bar_chart(
    results["risk_score"].value_counts(
        bins=10,
        sort=False,
    )
)
