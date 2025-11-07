from churn_custom_transformers import Winsorizer, TopNCategories, DistributionPreservingImputer
import streamlit as st
import pandas as pd
import joblib

# Load the saved model pipeline
#with open("model_pipeline2.pkl", "rb") as f:
#    model = pickle.load(f)

model = joblib.load("model_pipeline2.pkl")

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("🧠 Customer Churn Prediction")

st.markdown("Fill in the customer details below:")

# Numerical inputs
monthly_spend_inr = st.number_input("Monthly Spend (INR)", min_value=0.0, step=100.0)
transactions_last_90d = st.number_input("Transactions in Last 90 Days", min_value=0, step=1)
avg_session_duration_min = st.number_input("Avg. Session Duration (mins)", min_value=0.0, step=1.0)
support_tickets_last_year = st.number_input("Support Tickets Last Year", min_value=0, step=1)
account_vintage = st.number_input("Account Vintage (in days)", min_value=0, step=1)
activity_vintage = st.number_input("Activity Vintage (in days)", min_value=0, step=1)

# Categorical dropdowns
city = st.selectbox("City", [
    'Delhi', 'Bengaluru', 'Mumbai', 'Hyderabad', 'Pune', 'Nagpur',
    'Kochi', 'Lucknow', 'Ahmedabad', 'Kanpur', 'Surat', 'Chennai',
    'Kolkata', 'Varanasi', 'Visakhapatnam', 'Indore', 'Noida', 'Mysuru',
    'Ghaziabad', 'Gurugram', 'Patna', 'Chandigarh', 'Ranchi', 'Jaipur',
    'Bhubaneswar', 'Thane', 'Vijayawada', 'Bhopal', 'Guwahati', 'Coimbatore'
])

preferred_payment_method = st.selectbox("Preferred Payment Method", [
    'UPI', 'Credit Card', 'Debit Card', 'NetBanking', 'Wallet',
    'Cash on Delivery', 'EMI'
])

referral_source = st.selectbox("Referral Source", [
    'Organic', 'Twitter/X', 'Direct', 'Affiliate', 'Google Ads',
    'WhatsApp', 'Friend Referral', 'LinkedIn', 'Facebook Ads',
    'Instagram Ads', 'Reddit', 'Push', 'SMS', 'Blog', 'Email',
    'Radio', 'OOH', 'Quora', 'YouTube', 'TV'
])

tenure_bucket = st.selectbox("Tenure Bucket", [
    '3y+', '2-3y', '6-12m', '1-2y', '3-6m', '1-3m'
])

satisfaction_level = st.selectbox("Satisfaction Level", [
    'High', 'Medium', 'Very Low', 'Very High', 'Low'
])

risk_segment = st.selectbox("Risk Segment", [
    'Medium', 'Low', 'High'
])

# Prepare data as DataFrame
input_data = pd.DataFrame([{
    "monthly_spend_inr": monthly_spend_inr,
    "transactions_last_90d": transactions_last_90d,
    "avg_session_duration_min": avg_session_duration_min,
    "support_tickets_last_year": support_tickets_last_year,
    "account_vintage": account_vintage,
    "activity_vintage": activity_vintage,
    "city": city,
    "preferred_payment_method": preferred_payment_method,
    "referral_source": referral_source,
    "tenure_bucket": tenure_bucket,
    "satisfaction_level": satisfaction_level,
    "risk_segment": risk_segment
}])

# Prediction
if st.button("🔍 Predict Churn"):
    try:
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.error("⚠️ The customer is likely to churn.")
        else:
            st.success("✅ The customer is likely to stay.")
    except Exception as e:
        st.warning(f"Prediction failed: {e}")
