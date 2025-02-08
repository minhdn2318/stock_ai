import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np

# Cache functions remain the same as in original code
@st.cache_data(ttl=3600)
def visualize1(results, last_price, price_change):
    fig, ax = plt.subplots(figsize=(10, 6))
    predictions = list(results["individual_predictions"].values())
    models = list(results["individual_predictions"].keys())

    # Horizontal bar chart showing predictions
    y_pos = np.arange(len(models))
    ax.barh(y_pos, predictions)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.axvline(x=last_price, color="r", linestyle="--", label="Current Price")
    ax.axvline(
        x=results["prediction"],
        color="g",
        linestyle="--",
        label="Ensemble Prediction",
    )
    ax.set_xlabel("Price (VND*1000)")
    ax.set_title("Model Predictions Comparison")
    ax.legend()

    st.pyplot(fig)

    # Trading signal box
    signal_box = st.container()
    if abs(price_change) > 10:  # For very large changes
        if price_change > 0:
            signal_box.success(f"💹 Strong BUY Signal (+{price_change:.1f}%)")
        else:
            signal_box.error(f"📉 Strong SELL Signal ({price_change:.1f}%)")
    elif abs(price_change) > 3 and results["confidence_score"] > 0.8:
        if price_change > 0:
            signal_box.success(f"💹 BUY Signal (+{price_change:.1f}%)")
        else:
            signal_box.error(f"📉 SELL Signal ({price_change:.1f}%)")
    elif abs(price_change) > 2 and results["confidence_score"] > 0.6:
        if price_change > 0:
            signal_box.warning(f"📈 Moderate BUY Signal (+{price_change:.1f}%)")
        else:
            signal_box.warning(f"📉 Moderate SELL Signal ({price_change:.1f}%)")
    else:
        if abs(price_change) < 1:
            signal_box.info(f"⚖️ HOLD Signal ({price_change:.1f}%)")
        else:
            if price_change > 0:
                signal_box.info(f"📈 Weak BUY Signal (+{price_change:.1f}%)")
            else:
                signal_box.info(f"📉 Weak SELL Signal ({price_change:.1f}%)")

    # Model consensus analysis
    st.subheader("Model Consensus Analysis")
    buy_signals = sum(1 for pred in predictions if pred > last_price)
    sell_signals = sum(1 for pred in predictions if pred < last_price)
    total_models = len(predictions)

    consensus_col1, consensus_col2, consensus_col3 = st.columns(3)
    with consensus_col1:
        st.metric("Buy Signals", f"{buy_signals}/{total_models}")
    with consensus_col2:
        st.metric("Sell Signals", f"{sell_signals}/{total_models}")
    with consensus_col3:
        consensus_strength = abs(buy_signals - sell_signals) / total_models
        st.metric("Consensus Strength", f"{consensus_strength:.1%}")

    # Risk assessment
    st.subheader("Risk Assessment")
    prediction_std = np.std(predictions)
    prediction_range = results["upper_bound"] - results["lower_bound"]
    risk_level = (
        "Low"
        if prediction_std < last_price * 0.02
        else "Medium"
        if prediction_std < last_price * 0.05
        else "High"
    )

    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        currency = "VND"
        # if ".vn" in symbol.lower():
        #     currency = "VND"
        st.metric("Prediction Volatility", f"{prediction_std:.2f} {currency}")
    with risk_col2:
        st.metric("Risk Level", risk_level)

# Cache functions remain the same as in original code
@st.cache_data(ttl=3600)
def visualize2(df):
    # Đảm bảo 'Date' là kiểu datetime
    df["Date"] = pd.to_datetime(df["time"])

    # Định nghĩa fig và ax1 để vẽ biểu đồ.
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Đoạn này vẽ đường giá đóng cửa (Close Price)
    ax1.plot(df["Date"], df["close"], color="blue", label="Close Price", linewidth=2)

    # Tạo trục y thứ hai cho Volume
    ax2 = ax1.twinx()
    ax2.bar(
        df["Date"], df["volume"], color="gray", alpha=0.3, label="Volume", width=0.8
    )

    # Cập nhật tiêu đề và nhãn cho các trục
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price (VND)", color="blue")
    ax2.set_ylabel("Volume", color="gray")

    # Cập nhật tiêu đề biểu đồ
    ax1.set_title("Stock Price and Volume")

    # Thêm legend cho cả hai trục
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Cải thiện hiển thị của trục x (ngày tháng)
    fig.autofmt_xdate(rotation=45)  # Xoay ngày tháng để tránh bị chồng lên

    # Hiển thị biểu đồ trong Streamlit
    st.subheader("Stock Price Chart")
    st.pyplot(fig)
