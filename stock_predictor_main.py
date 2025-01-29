# Sử dụng các thư viện.
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
from newsapi import NewsApiClient
import yfinance as yf
import logging
from weights import WEIGHT_CONFIGURATIONS, WEIGHT_DESCRIPTIONS
from masp import MultiAlgorithmStockPredictor

# Đặt chế độ logging của TensorFlow thành chỉ log lỗi.
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Cấu hình trang web.
## Tiêu đề trang web trên thanh tiêu đề trình duyệt.
st.set_page_config(
    page_title="Multi-Algorithm Stock Predictor GROUP 7 - VNU", layout="wide"
)
## Hiển thị tiêu đề trang web ở giữa trang.
st.markdown(
    "<h1 style='text-align: center;'>Multi-Algorithm Stock Predictor GROUP 7 - VNU </h1>",
    unsafe_allow_html=True,
)
## Hiển thị phần giới thiệu/cảnh báo an toàn.
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

# Thiết lập API cho NewsAPI. TODO: Phần này chưa triển khai gì cả. Nên đặt API key vào file .env để tránh lộ thông tin.
NEWS_API_KEY = "567a5eff35d84d199867208fcbd51f26"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Cấu hình logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # In log ra màn hình.
    ],
)


# Sử dụng decorator để cache dữ liệu cho trang web.
# `ttl` là time-to-live, tức thời gian cache dữ liệu (sau bao lâu thì cache bị hết hạn).
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol: str, days: int) -> pd.DataFrame:
    """Hàm này dùng để lấy dữ liệu giá cổ phiếu từ Yahoo Finance.

    Args
    ----
        symbol: str
            Mã cổ phiếu cần lấy dữ liệu. VD: `VND.VN`.
        days: int
            Số ngày cần lấy dữ liệu.

    Returns
    -------
        df: pd.DataFrame
            DataFrame chứa dữ liệu giá cổ phiếu.
    """
    # Lấy ngày bắt đầu và kết thúc.
    ## Ngày kết thúc là ngày hiện tại (ngày chạy app).
    end_date = datetime.now()
    ## Ngày bắt đầu là ngày kết thúc trừ đi số ngày cần lấy dữ liệu.
    start_date = end_date - timedelta(days=days)

    # Hiển thị log thông tin về việc lấy dữ liệu.
    logging.info(
        f"Fetching stock data for {symbol} from {start_date.date()} to {end_date.date()}..."
    )

    # Lấy dữ liệu từ Yahoo Finance.
    df = yf.download(symbol, start=start_date, end=end_date)

    # In log kết quả.
    ## Nếu không có dữ liệu thì in ra cảnh báo kiểm tra lại mã cổ phiếu hoặc ngày.
    if df.empty:
        logging.warning("No data was fetched. Please check the symbol or date range.")
    ## Nếu có dữ liệu thì in ra số dòng dữ liệu và 5 dòng cuối cùng.
    else:
        logging.info(f"Successfully fetched {len(df)} rows of data for {symbol}.")
        logging.info(f"Last 5 rows of the data:\n{df.tail()}")
    return df


# Sử dụng decorator để cache dữ liệu cho trang web.
@st.cache_data(ttl=3600)
def get_news_headlines(symbol: str):
    """Hàm này dùng để lấy các tin tức liên quan đến mã cổ phiếu từ NewsAPI.

    TODO: Phần này chưa có ý nghĩa gì với hệ thống vì hàm này chưa được sử dụng ở đâu cả.

    Args
    ----
        symbol: str
            Mã cổ phiếu cần lấy tin tức.

    Returns
    -------
        news: List[Tuple[str, str, str]]
            Danh sách các tuple chứa tiêu đề, mô tả và link tin tức.
    """
    # Thử lấy dữ liệu từ NewsAPI.
    try:
        # Lấy tin tức (tiếng Anh) từ NewsAPI với số lượng 5 trang và sắp xếp theo mức độ liên quan.
        news = newsapi.get_everything(
            q=symbol, language="en", sort_by="relevancy", page_size=5
        )

        # Trả về danh sách các tin tức, mỗi tin tức là một tuple chứa tiêu đề, mô tả và link.
        return [
            (article["title"], article["description"], article["url"])
            for article in news["articles"]
        ]

    # Nếu có lỗi thì in ra log và trả về danh sách rỗng.
    except Exception as e:
        print(f"News API error: {str(e)}")
        return []


def calculate_technical_indicators_for_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Hàm này dùng để tính toán các chỉ số kỹ thuật cho cổ phiếu.

    Args
    ----
        df: pd.DataFrame
            DataFrame chứa dữ liệu giá cổ phiếu.

    Returns
    -------
        analysis_df: pd.DataFrame
            DataFrame chứa dữ liệu giá cổ phiếu và các chỉ số kỹ thuật đã tính toán.
    """
    # Tạo một bản sao của DataFrame để không ảnh hưởng đến dữ liệu gốc.
    analysis_df = df.copy()

    # Tính Moving Averages (MA), tiếng Việt là Trung bình động.
    ## MA là đường nối tất cả các giá đóng cửa của cổ phiếu trong n ngày gần nhất.
    # Công thức: MA_n = (Close[0] + Close[1] + ... + Close[n]) / n.
    ## Trong đó, n là số ngày trung bình động.
    analysis_df["MA20"] = analysis_df["Close"].rolling(window=20).mean()
    analysis_df["MA50"] = analysis_df["Close"].rolling(window=50).mean()

    # Tính Relative Strength Index (RSI), tiếng Việt là Chỉ số sức mạnh tương đối.
    ## RSI là một chỉ số đo lường sức mạnh của một cổ phiếu, nó thường được sử dụng để xác định xem cổ phiếu đó đã bị mua quá mức hay bán quá mức.
    # Công thức: RSI = 100 - (100 / (1 + RS)).
    ## Trong đó, RS là sức mạnh tương đối (Relative Strength), tính theo công thức: RS = AvgGain / AvgLoss (lãi trung bình / lỗ trung bình).
    delta = analysis_df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    analysis_df["RSI"] = 100 - (100 / (1 + rs))

    # Tính Volume MA (Moving Average), tiếng Việt là Trung bình động của khối lượng giao dịch.
    ## Về cơ bản, nó giống như MA nhưng áp dụng cho khối lượng giao dịch.
    analysis_df["Volume_MA"] = analysis_df["Volume"].rolling(window=20).mean()

    # Tính Bollinger Bands, tiếng Việt là Dải Bollinger.
    ## Dải Bollinger là một chỉ báo kỹ thuật dùng để đo lường biến động của giá và xác định các mức hỗ trợ và kháng cự tiềm năng.
    # Công thức: BB_upper = MA + (std * 2), BB_middle = MA, BB_lower = MA - (std * 2).
    ## Trong đó, MA là Moving Average, std là độ lệch chuẩn.
    ## Công thức được sử dụng ở đây là dùng MA20 và std20 (trong 20 ngày gần nhất).
    ma20 = analysis_df["Close"].rolling(window=20).mean()
    std20 = analysis_df["Close"].rolling(window=20).std()
    analysis_df["BB_upper"] = ma20 + (std20 * 2)
    analysis_df["BB_middle"] = ma20
    analysis_df["BB_lower"] = ma20 - (std20 * 2)

    # Trả về DataFrame sau khi tính toán xong.
    return analysis_df


def individual_model_predictions(last_price: float, df: pd.DataFrame) -> list[float]:
    # Hiển thị kết quả dự đoán riêng lẻ của từng mô hình.
    st.subheader("Individual Model Predictions")
    model_predictions = pd.DataFrame(
        {
            "Model": results["individual_predictions"].keys(),
            "Predicted Price": [v for v in results["individual_predictions"].values()],
        }
    )
    # Tính toán độ lệch giữa giá dự đoán riêng lẻ và giá dự đoán của mô hình kết hợp.
    model_predictions["Deviation from Ensemble"] = model_predictions[
        "Predicted Price"
    ] - abs(results["prediction"])

    # Thêm trọng số của mô hình vào bảng dữ liệu.
    model_predictions["Weight"] = WEIGHT_CONFIGURATIONS[selected_weight].values()

    # Sắp xếp dữ liệu theo giá dự đoán giảm dần.
    model_predictions = model_predictions.sort_values(
        "Predicted Price", ascending=False
    )

    # Đơn vị tiền tệ (USD hoặc VND).
    currency = "USD"
    if ".vn" in symbol.lower():
        currency = "VND"

    # Định dạng bảng dữ liệu.
    st.dataframe(
        model_predictions.style.format(
            {
                "Predicted Price": f"{{:.0f}} {currency}",
                "Deviation from Ensemble": f"{{:.0f}} {currency}",
                "Weight": "{:.2f}",
            }
        )
    )

    # Tạo biểu đồ.
    fig, ax = plt.subplots(figsize=(10, 6))
    predictions: list[float] = list(results["individual_predictions"].values())
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
    ax.set_xlabel("Price ($)")
    ax.set_title("Model Predictions Comparison")
    ax.legend()

    st.pyplot(fig)

    # Trả về dự đoán của từng mô hình.
    return predictions


def trading_signal_analysis(price_change: float):
    # Tạo một container trên trang web để hiển thị tín hiệu giao dịch.
    signal_box = st.container()

    # Phân tích tín hiệu giao dịch dựa trên sự thay đổi giá.
    ## Các tín hiệu mạnh.
    if abs(price_change) > 10:
        if price_change > 0:  # Tín hiệu mua mạnh.
            signal_box.success(f"💹 Strong BUY Signal (+{price_change:.1f}%)")
        else:  # Tín hiệu bán mạnh.
            signal_box.error(f"📉 Strong SELL Signal ({price_change:.1f}%)")
    ## Các tín hiệu mua/bán có độ tin cậy cao.
    elif abs(price_change) > 3 and results["confidence_score"] > 0.8:
        if price_change > 0:  # Tín hiệu mua.
            signal_box.success(f"💹 BUY Signal (+{price_change:.1f}%)")
        else:  # Tín hiệu bán.
            signal_box.error(f"📉 SELL Signal ({price_change:.1f}%)")
    ## Các tín hiệu mua/bán có độ tin cậy trung bình.
    elif abs(price_change) > 2 and results["confidence_score"] > 0.6:
        if price_change > 0: # Tín hiệu mua.
            signal_box.warning(f"📈 Moderate BUY Signal (+{price_change:.1f}%)")
        else: # Tín hiệu bán.
            signal_box.warning(f"📉 Moderate SELL Signal ({price_change:.1f}%)")
    ## Các tín hiệu mua/bán yếu.
    else:
        # Tín hiệu giữ cổ phiếu..
        if abs(price_change) < 1:
            signal_box.info(f"⚖️ HOLD Signal ({price_change:.1f}%)")
        else:
            if price_change > 0: # Tín hiệu mua yếu.
                signal_box.info(f"📈 Weak BUY Signal (+{price_change:.1f}%)")
            else: # Tín hiệu bán yếu.
                signal_box.info(f"📉 Weak SELL Signal ({price_change:.1f}%)")

def model_consensus_analysis(predictions: list[float], last_price: float):
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

def risk_assessment(predictions: list[float], last_price: float):
    st.subheader("Risk Assessment")
    prediction_std = np.std(predictions)
    risk_level = (
        "Low"
        if prediction_std < last_price * 0.02
        else "Medium"
        if prediction_std < last_price * 0.05
        else "High"
    )

    risk_col1, risk_col2 = st.columns(2)
    with risk_col1:
        currency = "USD"
        if ".vn" in symbol.lower():
            currency = "VND"
        st.metric(
            "Prediction Volatility", f"{prediction_std:.0f} {currency}"
        )
    with risk_col2:
        st.metric("Risk Level", risk_level)

# Giao diện trang web.
## Phần nhập mã cổ phiếu và số ngày gần nhất để tính toán.
symbol: str = st.text_input("Enter Stock Symbol (e.g., VND.VN):", "VND.VN")
display_days: int = st.slider(
    # Giá trị tối thiểu là 30 ngày (tức 1 tháng), tối đa là 3650 ngày (tức 10 năm).
    # Giá trị mặc định là 180 ngày (tức 6 tháng).
    label="Select number of days to calculate",
    min_value=30,
    max_value=3650,
    value=180,
)

# Hiển thị các cột dữ liệu.
col1, col2 = st.columns([2, 1])

# Cột 1: Hiển thị lựa chọn chiến lược trọng số cho các mô hình.
with col1:
    selected_weight = st.selectbox(
        "Select Weight Configuration:",
        options=list(WEIGHT_CONFIGURATIONS.keys()),
        help="Choose different weight configurations for the prediction models",
    )

# Cột 2: Hiển thị mô tả về trọng số các mô hình trong chiến lược.
with col2:
    st.info(WEIGHT_DESCRIPTIONS[selected_weight])

# Xử lý dữ liệu và hiển thị trên trang web.
try:
    # Hiển thị thông tin cổ phiếu.
    col1, col2 = st.columns([1, 1])

    # Với cột 1.
    with col1:
        # Hiển thị bảng trọng số.
        current_weights = WEIGHT_CONFIGURATIONS[selected_weight]
        weight_df = pd.DataFrame(
            list(current_weights.items()), columns=["Model", "Weight"]
        )
        st.subheader("Weight Configuration")
        st.dataframe(weight_df)

    # Hiển thị dữ liệu lịch sử giá cổ phiếu (5 ngày cuối cùng). Nếu không có dữ liệu thì hiển thị cảnh báo.
    with col2:
        st.subheader("Stock Price History")
        # Fetch data
        df = fetch_stock_data(symbol, display_days)
        if df is not None and not df.empty:
            st.write(df.tail())
        else:
            st.warning("No stock data available.")

    # Tính toán các chỉ số kỹ thuật và hiển thị trên trang web.
    col1, col2 = st.columns([1, 1])

    # Với cột 1.
    with col1:
        # Thực hiện dự đoán giá cổ phiếu.
        if st.button("Generate Predictions"):
            with st.spinner("Training multiple models and generating predictions..."):
                # Sử dụng class MultiAlgorithmStockPredictor để dự đoán giá cổ phiếu.
                predictor = MultiAlgorithmStockPredictor(
                    symbol, weights=WEIGHT_CONFIGURATIONS[selected_weight]
                )
                results = predictor.predict_with_all_models()

                # Nếu có kết quả thì hiển thị thông tin dự đoán.
                if results is not None:
                    # Tính toán giá đóng cửa cuối cùng của cổ phiếu.
                    last_price = float(df["Close"].iloc[-1])

                    # Tính toán và trả về các kết quả dự đoán của từng mô hình.
                    predictions: list[float] = individual_model_predictions(
                        last_price, df
                    )

                    # Tính sự thay đổi giá.
                    # Công thức: ((Giá dự đoán - Giá cuối cùng) / Giá cuối cùng) * 100.
                    price_change: float = (
                        (results["prediction"] - last_price) / last_price
                    ) * 100

                    # Xử lý tín hiệu giao dịch.
                    trading_signal_analysis(price_change)

                    # Model consensus analysis
                    model_consensus_analysis(predictions, last_price)

                    # Risk assessment
                    risk_assessment(predictions)

# Hiển thị cảnh báo nếu có lỗi xảy ra.
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Detailed error information:", str(e))

# Footer của trang web.
st.markdown("---")
