import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="AI Production Cost System", layout="wide")

st.title("🤖 AI-Based Production Cost Optimization System")

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("📂 Upload Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # ==============================
    # DATA PREPROCESSING
    # ==============================
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # Feature Engineering
    df['Date_Ordinal'] = df['Date'].map(pd.Timestamp.toordinal)

    # Features (SMART MODEL)
    features = ['Date_Ordinal']

    extra_features = ['Raw_Material_Cost', 'Labour_Cost', 'Units_Produced']
    for col in extra_features:
        if col in df.columns:
            features.append(col)

    X = df[features]
    y = df['Total_Production_Cost']

    # ==============================
    # MODEL TRAINING (ADVANCED)
    # ==============================
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)

    # Accuracy
    y_pred = model.predict(X)
    accuracy = r2_score(y, y_pred)

    # ==============================
    # FUTURE PREDICTION
    # ==============================
    future_dates = pd.date_range(start=df['Date'].max(), periods=60)

    future_df = pd.DataFrame({
        'Date': future_dates
    })

    future_df['Date_Ordinal'] = future_df['Date'].map(pd.Timestamp.toordinal)

    # Fill missing features with mean
    for col in extra_features:
        if col in df.columns:
            future_df[col] = df[col].mean()

    future_X = future_df[features]
    predictions = model.predict(future_X)

    # ==============================
    # 📊 METRICS
    # ==============================
    st.subheader("📌 Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(df))
    col2.metric("Latest Cost", f"{y.iloc[-1]:,.2f}")
    col3.metric("Predicted Next Day", f"{predictions[0]:,.2f}")
    col4.metric("Model Accuracy (R²)", f"{accuracy:.2f}")

    # ==============================
    # 📈 GRAPH
    # ==============================
    st.subheader("📈 Cost Analysis & Prediction")

    fig, ax = plt.subplots(figsize=(12, 5))

    # Actual
    ax.plot(df['Date'], y, label="Actual Cost", linewidth=2)

    # Trend Line
    df['Moving_Avg'] = df['Total_Production_Cost'].rolling(10).mean()
    ax.plot(df['Date'], df['Moving_Avg'], label="Trend Line", linewidth=3)

    # Prediction
    ax.plot(future_dates, predictions, linestyle='dashed', linewidth=2, label="Predicted Cost")

    ax.set_title("Production Cost Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cost")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # ==============================
    # 🔮 FUTURE TABLE
    # ==============================
    pred_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted Cost': predictions
    })

    st.subheader("🔮 Future Predictions")
    st.dataframe(pred_df)

    # ==============================
    # 🎯 USER INPUT PREDICTION
    # ==============================
    st.sidebar.header("🔢 Predict Your Own Cost")

    units = st.sidebar.number_input("Units Produced", value=300)
    raw = st.sidebar.number_input("Raw Material Cost", value=20000)
    labour = st.sidebar.number_input("Labour Cost", value=12000)

    if st.sidebar.button("Predict Cost"):
        input_data = [[
            df['Date_Ordinal'].max(),  # current date
            raw,
            labour,
            units
        ]]

        pred_cost = model.predict(input_data)[0]

        st.sidebar.success(f"💰 Predicted Cost: {pred_cost:,.2f}")

        # Alert system
        if pred_cost > 50000:
            st.sidebar.error("⚠️ High Production Cost!")
        else:
            st.sidebar.info("✅ Cost is under control")

    # ==============================
    # 📊 ADDITIONAL VISUALS
    # ==============================
    st.subheader("📊 Cost Relationships")

    fig2, ax2 = plt.subplots()
    ax2.scatter(df['Units_Produced'], y)
    ax2.set_xlabel("Units Produced")
    ax2.set_ylabel("Production Cost")
    ax2.set_title("Cost vs Units Produced")

    st.pyplot(fig2)

    # ==============================
    # 📌 INSIGHTS
    # ==============================
    st.subheader("📌 Insights")

    st.write("• Production cost shows fluctuations based on input factors.")
    st.write("• Raw material cost significantly impacts total cost.")
    st.write("• Increasing units generally increases production cost.")
    st.write("• Model predicts future cost trends using machine learning.")

    # ==============================
    # 📥 DOWNLOAD
    # ==============================
    csv = pred_df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="📥 Download Predictions",
        data=csv,
        file_name='predicted_costs.csv',
        mime='text/csv'
    )

else:
    st.info("👆 Upload a dataset to begin analysis.")
