import streamlit as st
import requests

st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

st.markdown("""
<style>
.block-container{
    padding-top:2rem;
    max-width:1100px;
}
.stButton > button{
    width:100%;
    height:52px;
    border-radius:12px;
    border:none;
    font-size:18px;
    font-weight:600;
    background:linear-gradient(90deg,#2563eb,#06b6d4);
    color:white;
}
</style>
""", unsafe_allow_html=True)

st.title("Fraud Detection Dashboard")
st.caption("AI-powered fraud risk analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Transaction Info")

    step = st.number_input("Step (hour)", min_value=0, value=1)

    type_ = st.selectbox(
        "Transaction Type",
        ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    )

    amount = st.number_input("Amount", min_value=0.0, value=1000.0)

with col2:
    st.subheader("Balances")

    oldbalanceOrg = st.number_input("Sender Old Balance", min_value=0.0, value=5000.0)

    newbalanceOrig = st.number_input("Sender New Balance", min_value=0.0, value=4000.0)

    oldbalanceDest = st.number_input("Receiver Old Balance", min_value=0.0, value=1000.0)

    newbalanceDest = st.number_input("Receiver New Balance", min_value=0.0, value=2000.0)

type_map = {
    "PAYMENT": 0,
    "TRANSFER": 1,
    "CASH_OUT": 2,
    "DEBIT": 3,
    "CASH_IN": 4
}

st.divider()

if st.button("🔍 Analyze Transaction"):

    data = {
        "step": int(step),
        "type": int(type_map[type_]),
        "amount": float(amount),
        "oldbalanceOrg": float(oldbalanceOrg),
        "newbalanceOrig": float(newbalanceOrig),
        "oldbalanceDest": float(oldbalanceDest),
        "newbalanceDest": float(newbalanceDest)
    }

    try:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            json=data,
            timeout=10
        )

        result = response.json()

        st.write("Response Status Code:", response.status_code)
        st.json(result)

        if result.get("status") == "success":

            prob = result["fraud_probability"] * 100

            if result["prediction"] == 1:
                st.error("⚠ Fraud Detected")
            else:
                st.success("✅ Safe Transaction")

            st.metric("Fraud Probability", f"{prob:.2f}%")

        else:
            st.error("Backend Error")
            st.json(result)

    except Exception as e:
        st.error(f"Request failed: {e}")