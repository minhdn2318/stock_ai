import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from vnstock import Vnstock
import yfinance as yf
import logging
import random
import extra_streamlit_components as stx
from models.predictor import MultiAlgorithmStockPredictor
import weight_configurations

def get_cookie(name):
    return cookie_manager.get(name)


def set_cookie(name, value):
    cookie_manager.set(name, value)


# Randomly assign user to a variant (50% chance for A or B)
def get_variant():
    if get_cookie("variant"):
        return get_cookie("variant")
    else:
        variant = random.choice(["A", "B"])
        set_cookie("variant", variant)
        return get_cookie("variant")


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

st.set_page_config(
    page_title="Multi-Algorithm Stock Predictor GROUP 6 - 2425I_INT7024 - VNU",
    layout="wide",
)
st.markdown(
    "<h1 style='text-align: center;'>Multi-Algorithm Stock Predictor GROUP 6 - 2425I_INT7024 - VNU </h1>",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style='text-align: center; color: gray; font-size: 14px;'>
    Disclaimer: This application provides stock predictions based on algorithms and is intended for informational purposes only. 
    Predictions may not be accurate, and users are encouraged to conduct their own research and consider consulting with a 
    financial advisor before making any investment decisions. This is not financial advice, and I am not responsible for any 
    outcomes resulting from the use of this application.
    </p>
    """,
    unsafe_allow_html=True,
)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # In log ra màn hình
    ],
)


# Cookie functionality
cookie_manager = stx.CookieManager(key="my_cookie")

# Check the variant assigned
variant = get_variant()


# Cache functions remain the same as in original code
@st.cache_data(ttl=3600)
def fetch_stock_data_from_yahoo_finance(symbol, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    logging.info(
        f"Fetching stock data for {symbol} from {start_date.date()} to {end_date.date()}..."
    )
    df = yf.download(symbol, start=start_date, end=end_date)

    # In log kết quả
    if df.empty:
        logging.warning("No data was fetched. Please check the symbol or date range.")
    else:
        logging.info(f"Successfully fetched {len(df)} rows of data for {symbol}.")
        logging.info(f"Last 5 rows of the data:\n{df.tail()}")
    return df


# Cache functions remain the same as in original code
@st.cache_data(ttl=3600)
def fetch_stock_data_from_vnstock(symbol, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    logging.info(
        f"Fetching stock data for {symbol} from {start_date.date()} to {end_date.date()}..."
    )
    symbol_vnstock = symbol.split(".")[0]  # Giữ phần trước dấu chấm
    stock = Vnstock().stock(symbol=symbol_vnstock, source="TCBS")
    df = stock.quote.history(
        start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d")
    )

    # In log kết quả
    if df.empty:
        logging.warning("No data was fetched. Please check the symbol or date range.")
    else:
        logging.info(f"Successfully fetched {len(df)} rows of data for {symbol}.")
        logging.info(f"Last 5 rows of the data from Vnstock:\n{df.tail()}")
    return df


# Streamlit interface
show_variant_version = st.write(f"System Variant: {variant}")

if variant == "B":
    st.warning(
        "🚨 You are currently using Variant B of the application. This variant focuses more on LSTM and tree-based algorithms." 
    )

symbol = st.text_input("Enter VietNam Stock Symbol (e.g., VND):", "VND")
display_days = st.slider("Select number of days to display", 30, 3650, 180)

# Define different weight configurations
WEIGHT_CONFIGURATIONS = (
    weight_configurations.WEIGHT_CONFIGURATIONS
    if variant == "A"
    else weight_configurations.WEIGHT_CONFIGURATIONS_BETA
)

WEIGHT_DESCRIPTIONS = {
    "Default": "Original configuration with balanced weights",
    "Trend-Focused": "Best for growth stocks, tech stocks, clear trend patterns",
    "Statistical": "Best for blue chip stocks, utilities, stable dividend stocks",
    "Tree-Ensemble": "Best for stocks with complex relationships to market factors",
    "Balanced": "Best for general purpose, unknown stock characteristics",
    "Volatility-Focused": "Best for small cap stocks, emerging market stocks, crypto-related stocks",
}

col1, col2 = st.columns([2, 1])

with col1:
    selected_weight = st.selectbox(
        "Select Weight Configuration:",
        options=list(WEIGHT_CONFIGURATIONS.keys()),
        help="Choose different weight configurations for the prediction models",
    )


with col2:
    st.info(WEIGHT_DESCRIPTIONS[selected_weight])

try:
    # Display stock price chart
    # st.subheader("Stock Price Chart")
    # st.line_chart(df['close'])

    # show info stock and weight
    col1, col2 = st.columns([1, 1])
    with col1:
        # Hiển thị bảng trọng số
        current_weights = WEIGHT_CONFIGURATIONS[selected_weight]
        weight_df = pd.DataFrame(
            list(current_weights.items()), columns=["Model", "Weight"]
        )
        st.subheader("Weight Configuration")
        st.dataframe(weight_df)
    with col2:
        st.subheader("Stock Price History")
        # Fetch data
        df = fetch_stock_data_from_vnstock(symbol, display_days)
        if df is not None and not df.empty:
            st.write(df.tail())
        else:
            st.warning("No stock data available.")

    # Thêm CSS cho nút đẹp mắt với màu nhã nhặn và hiệu ứng nhấp nháy
    st.markdown(
        """
        <style>
            .custom-button {
                background-color: #4CAF50;  /* Màu xanh nhã nhặn */
                color: white;
                font-size: 18px;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                border-radius: 10px;
                border: none;
                cursor: pointer;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                animation: blink 1s linear infinite;  /* Hiệu ứng nhấp nháy */
            }

            .custom-button:hover {
                background-color: #45a049;  /* Màu đậm hơn khi hover */
                transform: scale(1.05);  /* Phóng to nhẹ khi hover */
            }

            .custom-button:active {
                transform: scale(0.98);  /* Nhấn vào sẽ co lại một chút */
            }

            /* Hiệu ứng nhấp nháy */
            @keyframes blink {
                0% { background-color: #4CAF50; }
                50% { background-color: #80C784; }
                100% { background-color: #4CAF50; }
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # HTML cho nút tùy chỉnh
    button_html = '<button class="custom-button">Generate Predictions</button>'

    # Hiển thị nút với Streamlit
    # st.markdown(button_html, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        if st.button("Generate Predictions"):
            with st.spinner("Training multiple models and generating predictions..."):
                predictor = MultiAlgorithmStockPredictor(
                    symbol, weights=WEIGHT_CONFIGURATIONS[selected_weight]
                )
                results = predictor.predict_with_all_models()

                if results is not None:
                    last_price = float(df["close"].iloc[-1])

                    # Individual model predictions
                    st.subheader("Individual Model Predictions")
                    model_predictions = pd.DataFrame(
                        {
                            "Model": results["individual_predictions"].keys(),
                            "Predicted Price": [
                                v for v in results["individual_predictions"].values()
                            ],
                        }
                    )
                    model_predictions["Deviation from Ensemble"] = model_predictions[
                        "Predicted Price"
                    ] - abs(results["prediction"])
                    model_predictions["Weight"] = WEIGHT_CONFIGURATIONS[
                        selected_weight
                    ].values()
                    model_predictions = model_predictions.sort_values(
                        "Predicted Price", ascending=False
                    )
                    currency = "VND"
                    # if ".vn" in symbol.lower():
                    #     currency = "VND"
                    st.dataframe(
                        model_predictions.style.format(
                            {
                                "Predicted Price": f"{{:.2f}} {currency}",
                                "Deviation from Ensemble": f"{{:.2f}} {currency}",
                                "Weight": "{:.2f}",
                            }
                        )
                    )
                    # Hiển thị tiêu đề
                    st.subheader("🔥 Average Predicted Price")
                    # Lấy dữ liệu chứng khoán và giá hiện tại
                    current_price = df["close"].iloc[-1] if not df.empty else 0
                    # Tính biên lợi nhuận
                    profit_margin = (
                        (abs(results["prediction"] - current_price) / current_price)
                        * 100
                        if current_price != 0
                        else 0
                    )
                    # Hiển thị 3 cột với tiêu đề và giá trị
                    col1, col2, col3 = st.columns(3)

                    # Cột 1: Giá dự đoán
                    with col1:
                        st.metric(
                            label="Predicted Price",
                            value=f"{abs(results['prediction']):.2f} VND",
                        )

                    # Cột 2: Giá hiện tại
                    with col2:
                        st.metric(
                            label="Current Price", value=f"{current_price:.2f} VND"
                        )

                    # Cột 3: Phần trăm biên lợi nhuận
                    with col3:
                        st.metric(
                            label="Profit Margin (%)", value=f"{profit_margin:.2f}%"
                        )

                    # Trading signal with confidence
                    price_change = (
                        (results["prediction"] - last_price) / last_price
                    ) * 100

                    # Create a prediction distribution plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    predictions = list(results["individual_predictions"].values())
                    models = list(results["individual_predictions"].keys())

                    # Horizontal bar chart showing predictions
                    y_pos = np.arange(len(models))
                    ax.barh(y_pos, predictions)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(models)
                    ax.axvline(
                        x=last_price, color="r", linestyle="--", label="Current Price"
                    )
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
                            signal_box.success(
                                f"💹 Strong BUY Signal (+{price_change:.1f}%)"
                            )
                        else:
                            signal_box.error(
                                f"📉 Strong SELL Signal ({price_change:.1f}%)"
                            )
                    elif abs(price_change) > 3 and results["confidence_score"] > 0.8:
                        if price_change > 0:
                            signal_box.success(f"💹 BUY Signal (+{price_change:.1f}%)")
                        else:
                            signal_box.error(f"📉 SELL Signal ({price_change:.1f}%)")
                    elif abs(price_change) > 2 and results["confidence_score"] > 0.6:
                        if price_change > 0:
                            signal_box.warning(
                                f"📈 Moderate BUY Signal (+{price_change:.1f}%)"
                            )
                        else:
                            signal_box.warning(
                                f"📉 Moderate SELL Signal ({price_change:.1f}%)"
                            )
                    else:
                        if abs(price_change) < 1:
                            signal_box.info(f"⚖️ HOLD Signal ({price_change:.1f}%)")
                        else:
                            if price_change > 0:
                                signal_box.info(
                                    f"📈 Weak BUY Signal (+{price_change:.1f}%)"
                                )
                            else:
                                signal_box.info(
                                    f"📉 Weak SELL Signal ({price_change:.1f}%)"
                                )

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
                        consensus_strength = (
                            abs(buy_signals - sell_signals) / total_models
                        )
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
                        st.metric(
                            "Prediction Volatility", f"{prediction_std:.2f} {currency}"
                        )
                    with risk_col2:
                        st.metric("Risk Level", risk_level)

        ####-chart stock-############################################
    # Đảm bảo 'Date' là kiểu datetime
    df["Date"] = pd.to_datetime(df["time"])

    # Vẽ biểu đồ với Matplotlib
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
    
    # Rate variant
    if st.button("Rate Variant"):
        st.write("What do you think about this variant?")
        stx.RatingComponent(key="variant-rating")
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Detailed error information:", str(e))
st.markdown("---")

# Rating
import os

# CSV file path
RATINGS_LOG = "ratings_log.csv"

# Ensure CSV file exists
if not os.path.exists(RATINGS_LOG):
    pd.DataFrame(columns=["User ID", "Rating", "Comment"]).to_csv(RATINGS_LOG, index=False)

st.title("Rate This Prediction")

# User rating input
variant_version = get_variant()
rating = st.slider("Rate the prediction:", min_value=1, max_value=5, step=1)
comment = st.text_area("Optional comment:")

if st.button("Submit Rating"):
    # Append rating to CSV
    new_entry = pd.DataFrame([{"Variant": variant_version, "Rating": rating, "Comment": comment}])
    new_entry.to_csv(RATINGS_LOG, mode="a", header=False, index=False)
    st.success("Thank you for your feedback! 🎉")

st.markdown("---")


