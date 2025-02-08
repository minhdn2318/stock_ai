import pandas as pd
import streamlit as st
import tensorflow as tf
import logging
import extra_streamlit_components as stx
from utils.cookie import get_variant
from models.predictor import MultiAlgorithmStockPredictor, run_shadow_test
from utils.ratings import ratings_function
import models.weight_configurations as weight_configurations
from utils.fetch_data import fetch_stock_data_from_vnstock
from utils.visualize import visualize1, visualize2


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
variant = get_variant(cookie_manager)

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

WEIGHT_DESCRIPTIONS = weight_configurations.WEIGHT_DESCRIPTIONS

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
        if st.checkbox("Run Shadow Test"):
            shadow_test_allowed = True
        else:
            shadow_test_allowed = False

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
                    visualize1(results, last_price, price_change)

                # Shadow testing
                if shadow_test_allowed:
                    run_shadow_test(
                        symbol, selected_weight, variant, results, current_price
                    )

    ####-chart stock-############################################
    # Vẽ biểu đồ với Matplotlib
    visualize2(df)

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Detailed error information:", str(e))
st.markdown("---")

# Rating

ratings_function(variant)

st.markdown("---")
