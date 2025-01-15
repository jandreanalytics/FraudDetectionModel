import streamlit as st
import requests
import pandas as pd
from datetime import datetime
import os
import plotly.express as px

st.set_page_config(page_title="Fraud Detection System", layout="wide")

def main():
    st.title("Fraud Detection System")
    
    # Initialize session state for transaction history
    if 'transactions' not in st.session_state:
        st.session_state.transactions = []

    # Create two columns
    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Test Transaction")
        with st.form("transaction_form"):
            amount = st.number_input("Amount ($)", min_value=0.01, value=100.00, step=0.01)
            location = st.selectbox(
                "Location",
                ["NY", "LA", "CH", "HO", "SF", "UK", "RU", "BR", "CN"]
            )
            transaction_type = st.selectbox(
                "Transaction Type",
                ["pos", "online", "atm", "recurring"]
            )
            merchant_category = st.selectbox(
                "Merchant Category",
                ["retail", "dining", "travel", "online", "entertainment"]
            )
            
            submitted = st.form_submit_button("Check Transaction")
            
            if submitted:
                # Prepare transaction data
                data = {
                    "transaction_id": int(datetime.now().timestamp()),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user_id": 1234,
                    "amount": amount,
                    "location": location,
                    "transaction_type": transaction_type,
                    "merchant_category": merchant_category
                }

                # Call API
                api_url = os.getenv('API_URL', 'http://localhost:5000')
                try:
                    response = requests.post(
                        f"{api_url}/api/v1/predict",
                        json=data
                    )
                    result = response.json()
                    
                    # Add to transaction history
                    st.session_state.transactions.append({
                        **data,
                        **result
                    })
                    
                    # Display result
                    risk_color = {
                        "high": "🔴",
                        "medium": "🟡",
                        "low": "🟢",
                        "none": "⚪"
                    }
                    
                    st.markdown("### Analysis Result")
                    st.markdown(f"**Risk Level:** {risk_color[result['fraud_level']]} {result['fraud_level'].upper()}")
                    st.markdown(f"**Fraud Probability:** {result['fraud_probability']*100:.1f}%")
                    st.markdown(f"**Decision:** {'🚫 SUSPICIOUS' if result['is_fraud'] else '✅ LEGITIMATE'}")
                    
                    if result['risk_patterns']:
                        st.markdown("**Risk Patterns Detected:**")
                        for pattern, is_risky in result['risk_patterns'].items():
                            if is_risky:
                                st.markdown(f"- {pattern.replace('_', ' ').title()}")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    with col2:
        if st.session_state.transactions:
            st.subheader("Transaction History")
            
            # Convert transactions to DataFrame
            df = pd.DataFrame(st.session_state.transactions)
            
            # Create visualization
            fig = px.scatter(
                df,
                x="amount",
                y="fraud_probability",
                color="fraud_level",
                size="amount",
                hover_data=["location", "transaction_type", "merchant_category"],
                title="Transaction Risk Analysis"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show transactions table
            st.dataframe(
                df[["timestamp", "amount", "location", "fraud_level", "fraud_probability"]]
                .sort_values("timestamp", ascending=False)
                .style.format({
                    "amount": "${:.2f}",
                    "fraud_probability": "{:.1%}"
                })
            )

if __name__ == "__main__":
    main()
