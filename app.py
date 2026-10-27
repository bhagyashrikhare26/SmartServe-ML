
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="SmartServe ML",
    page_icon="🍽️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("models/knn_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/train.csv")

model = load_model()
df = load_data()

st.title("🍽️ SmartServe ML")
st.subheader("Smart College Canteen Food Demand Predictor")

st.write(
    "Predict food demand using Machine Learning "
    "and reduce food wastage."
)

tab1, tab2, tab3 = st.tabs([
    "🔮 Demand Prediction",
    "📊 Dataset Insights",
    "🤖 Model Comparison"
])

with tab1:
    st.header("Predict Food Orders")

    col1, col2 = st.columns(2)

    with col1:
        week = st.number_input(
            "Week Number",
            min_value=1,
            max_value=200,
            value=145
        )

        center_id = st.selectbox(
            "Canteen / Center ID",
            sorted(df["center_id"].unique())
        )

        meal_id = st.selectbox(
            "Meal ID",
            sorted(df["meal_id"].unique())
        )

        checkout_price = st.number_input(
            "Selling Price (₹)",
            min_value=0.0,
            value=150.0
        )

    with col2:
        base_price = st.number_input(
            "Base Price (₹)",
            min_value=0.0,
            value=180.0
        )

        emailer = st.selectbox(
            "Promotion Email Sent?",
            ["No", "Yes"]
        )

        featured = st.selectbox(
            "Featured on Homepage?",
            ["No", "Yes"]
        )

    if st.button("Predict Demand", type="primary"):
        input_data = pd.DataFrame([{
            "week": week,
            "center_id": center_id,
            "meal_id": meal_id,
            "checkout_price": checkout_price,
            "base_price": base_price,
            "emailer_for_promotion": int(emailer == "Yes"),
            "homepage_featured": int(featured == "Yes")
        }])

        prediction = model.predict(input_data)[0]
        predicted_orders = max(0, round(prediction))

        st.success("Prediction completed!")
        st.metric(
            "Estimated Food Orders",
            f"{predicted_orders:,}"
        )

        st.info(
            "This is an estimate based on historical "
            "food demand data, not a guaranteed count."
        )

with tab2:
    st.header("Dataset Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Unique Meals", df["meal_id"].nunique())
    c3.metric("Unique Centers", df["center_id"].nunique())

    st.subheader("Food Orders by Week")

    weekly_orders = (
        df.groupby("week")["num_orders"]
        .sum()
    )

    st.line_chart(weekly_orders)

    st.subheader("Top 10 Meals by Total Orders")

    top_meals = (
        df.groupby("meal_id")["num_orders"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )

    st.bar_chart(top_meals)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), width="stretch")

with tab3:
    st.header("Machine Learning Model Comparison")

    results = pd.DataFrame({
        "Algorithm": [
            "Linear Regression",
            "Decision Tree",
            "Random Forest",
            "KNN"
        ],
        "MAE": [149.31, 136.92, 131.73, 88.17],
        "RMSE": [260.16, 269.71, 249.17, 197.37],
        "R2 Score": [0.4925, 0.4546, 0.5345, 0.7080]
    })

    st.dataframe(results, width="stretch")

    st.subheader("R² Score Comparison")
    st.bar_chart(
        results.set_index("Algorithm")["R2 Score"]
    )

    st.success("Best Tested Model: KNN Regressor")

st.divider()
st.caption(
    "SmartServe ML | Machine Learning Mini Project | "
    "Developed by Bhagyashri Vinod Khare"
)
