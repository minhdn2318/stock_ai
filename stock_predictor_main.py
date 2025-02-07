import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from newsapi import NewsApiClient
from vnstock import Vnstock
import yfinance as yf
import plotly.graph_objects as go
import logging

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

st.set_page_config(page_title="Multi-Algorithm Stock Predictor GROUP 6 - 2425I_INT7024 - VNU", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Multi-Algorithm Stock Predictor GROUP 6 - 2425I_INT7024 - VNU </h1>",
    unsafe_allow_html=True
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
    unsafe_allow_html=True
)
# API setup
# NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
NEWS_API_KEY = "567a5eff35d84d199867208fcbd51f26"
newsapi = NewsApiClient(api_key=NEWS_API_KEY)

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # In log ra màn hình
    ]
)

# Cache functions remain the same as in original code
@st.cache_data(ttl=3600)
def fetch_stock_data_from_yahoo_finance(symbol, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    logging.info(f"Fetching stock data for {symbol} from {start_date.date()} to {end_date.date()}...")
    df = yf.download(symbol, start=start_date, end=end_date)

     # In log kết quả
    if df.empty:
        logging.warning("No data was fetched. Please check the symbol or date range.")
    else:
        logging.info(f"Successfully fetched {len(df)} rows of data for {symbol}.")
        logging.info(f"Last 5 rows of the data:\n{df.tail()}")
        # st.write(df.tail())
    return df

# Cache functions remain the same as in original code
@st.cache_data(ttl=3600)
def fetch_stock_data_from_vnstock(symbol, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    logging.info(f"Fetching stock data for {symbol} from {start_date.date()} to {end_date.date()}...")
    symbol_vnstock = symbol.split('.')[0]  # Giữ phần trước dấu chấm
    stock = Vnstock().stock(symbol=symbol_vnstock, source="TCBS")
    df = stock.quote.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

     # In log kết quả
    if df.empty:
        logging.warning("No data was fetched. Please check the symbol or date range.")
    else:
        logging.info(f"Successfully fetched {len(df)} rows of data for {symbol}.")
        logging.info(f"Last 5 rows of the data from Vnstock:\n{df.tail()}")
        # st.write(df.tail())
    return df


# test api key 
@st.cache_data(ttl=3600)
def get_news_headlines(symbol):
    try:
        news = newsapi.get_everything(
            q=symbol,
            language='en',
            sort_by='relevancy',
            page_size=5
        )
        return [(article['title'], article['description'], article['url']) 
                for article in news['articles']]
    except Exception as e:
        print(f"News API error: {str(e)}")
        return []
def calculate_technical_indicators_for_summary(df):
        analysis_df = df.copy()
        
        # Calculate Moving Averages
        analysis_df['MA20'] = analysis_df['close'].rolling(window=20).mean()
        analysis_df['MA50'] = analysis_df['close'].rolling(window=50).mean()
        
        # Calculate RSI
        delta = analysis_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        analysis_df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Volume MA
        analysis_df['Volume_MA'] = analysis_df['volume'].rolling(window=20).mean()
        
        # Calculate Bollinger Bands
        ma20 = analysis_df['close'].rolling(window=20).mean()
        std20 = analysis_df['close'].rolling(window=20).std()
        analysis_df['BB_upper'] = ma20 + (std20 * 2)
        analysis_df['BB_lower'] = ma20 - (std20 * 2)
        analysis_df['BB_middle'] = ma20
        
        return analysis_df

class MultiAlgorithmStockPredictor:
    def __init__(self, symbol, training_years=5, weights=None):
        self.symbol = symbol
        self.training_years = training_years
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.weights = weights if weights is not None else WEIGHT_CONFIGURATIONS["Default"]
        
    def fetch_historical_data_from_yahoo(self):
        # Same as original EnhancedStockPredictor
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.training_years)
        
        try:
            df = yf.download(self.symbol, start=start_date, end=end_date)
            if df.empty:
                st.warning(f"Data for the last {self.training_years} years is unavailable. Fetching maximum available data instead.")
                df = yf.download(self.symbol, period="max")
            return df
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return yf.download(self.symbol, period="max")

    def fetch_historical_data_from_vnstock(self):
        # Same as original EnhancedStockPredictor
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.training_years)
        
        try:
            symbol_vnstock = self.symbol.split('.')[0]  # Giữ phần trước dấu chấm
            stock = Vnstock().stock(symbol=symbol_vnstock, source="TCBS")
            df = stock.quote.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))

            if df.empty:
                st.warning(f"Data for the last {self.training_years} years is unavailable. Fetching maximum available data instead.")
            return df
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return df
    
    # Technical indicators calculation methods remain the same
    def calculate_technical_indicators(self, df):
        # Original technical indicators remain the same
        df['MA5'] = df['close'].rolling(window=5).mean()
        df['MA20'] = df['close'].rolling(window=20).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA200'] = df['close'].rolling(window=200).mean()
        df['RSI'] = self.calculate_rsi(df['close'])
        df['MACD'] = self.calculate_macd(df['close'])
        df['ROC'] = df['close'].pct_change(periods=10) * 100
        df['ATR'] = self.calculate_atr(df)
        df['BB_upper'], df['BB_lower'] = self.calculate_bollinger_bands(df['close'])
        df['Volume_MA'] = df['volume'].rolling(window=20).mean()
        df['Volume_Rate'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Additional technical indicators
        df['EMA12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['close'].ewm(span=26, adjust=False).mean()
        df['MOM'] = df['close'].diff(10)
        df['STOCH_K'] = self.calculate_stochastic(df)
        df['WILLR'] = self.calculate_williams_r(df)
        
        return df.dropna()
    
    
    @staticmethod
    def calculate_stochastic(df, period=14):
        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()
        k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        return k

    @staticmethod
    def calculate_williams_r(df, period=14):
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        return -100 * ((high_max - df['close']) / (high_max - low_min))

    # Original calculation methods remain the same
    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(prices, slow=26, fast=12, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        return exp1 - exp2
    
    @staticmethod
    def calculate_atr(df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, lower_band

    def prepare_data(self, df, seq_length=60):
        feature_columns = ['close', 'MA5', 'MA20', 'MA50', 'MA200', 'RSI', 'MACD', 
                          'ROC', 'ATR', 'BB_upper', 'BB_lower', 'Volume_Rate',
                          'EMA12', 'EMA26', 'MOM', 'STOCH_K', 'WILLR']
        
        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_columns])
        
        # Prepare sequences for LSTM
        X_lstm, y = [], []
        for i in range(seq_length, len(scaled_data)):
            X_lstm.append(scaled_data[i-seq_length:i])
            y.append(scaled_data[i, 0])  # 0 index represents Close price
            
        # Prepare data for other models
        X_other = scaled_data[seq_length:]
        
        return np.array(X_lstm), X_other, np.array(y)

    def build_lstm_model(self, input_shape):
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(50, return_sequences=True)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dropout(0.1),
            Dense(10, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model

    def train_arima(self, df):
        model = ARIMA(df['close'], order=(5,1,0))
        return model.fit()

    def predict_with_all_models(self, prediction_days=30, sequence_length=60):
        try:
            # Fetch and prepare data
            df = self.fetch_historical_data_from_vnstock()
            logging.info(f"Successfully fetched {len(df)} rows of data for {self.symbol}.")
            # Check if we have enough data
            if len(df) < sequence_length + 20:  # Need extra days for technical indicators
                st.error(f"Insufficient historical data. Need at least {sequence_length + 20} days of data.")
                return None
                
            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)
            
            # Check for NaN values and handle them
            if df.isnull().any().any():
                df = df.fillna(method='ffill').fillna(method='bfill')
                
            # Verify we have enough valid data after cleaning
            if len(df.dropna()) < sequence_length:
                st.error("Insufficient valid data after calculating indicators.")
                return None
                
            # Prepare features
            feature_columns = ['close', 'MA5', 'MA20', 'MA50', 'MA200', 'RSI', 'MACD', 
                            'ROC', 'ATR', 'BB_upper', 'BB_lower', 'Volume_Rate',
                            'EMA12', 'EMA26', 'MOM', 'STOCH_K', 'WILLR']
                            
            # Verify all required features exist
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                st.error(f"Missing required features: {', '.join(missing_features)}")
                return None
                
            # Ensure we have valid data for all features
            df = df[feature_columns].dropna()
            if len(df) < sequence_length:
                st.error(f"Insufficient valid data points after cleaning. Need at least {sequence_length} points.")
                st.write(f"Available data points: {len(df)}")
                return None
                
            try:
                # Scale features
                scaled_data = self.scaler.fit_transform(df[feature_columns])
            except ValueError as e:
                st.error(f"Scaling error: {str(e)}")
                st.write("This usually happens with newly listed stocks or stocks with insufficient trading history.")
                return None
                
            # Prepare sequences for LSTM
            X_lstm, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X_lstm.append(scaled_data[i-sequence_length:i])
                y.append(scaled_data[i, 0])  # 0 index represents Close price
                
            # Verify we have enough sequences
            if len(X_lstm) == 0 or len(y) == 0:
                st.error("Could not create valid sequences for prediction.")
                return None
                
            # Prepare data for other models
            X_other = scaled_data[sequence_length:]
            
            # Convert to numpy arrays
            X_lstm = np.array(X_lstm)
            X_other = np.array(X_other)
            y = np.array(y)
            
            # Split data
            split_idx = int(len(y) * 0.8)
            X_lstm_train, X_lstm_test = X_lstm[:split_idx], X_lstm[split_idx:]
            X_other_train, X_other_test = X_other[:split_idx], X_other[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            predictions = {}
            
            # Train and predict with LSTM
            lstm_model = self.build_lstm_model((sequence_length, X_lstm.shape[2]))
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            lstm_model.fit(X_lstm_train, y_train, epochs=50, batch_size=32,
                          validation_data=(X_lstm_test, y_test),
                          callbacks=[early_stopping], verbose=0)
            lstm_pred = lstm_model.predict(X_lstm_test[-1:], verbose=0)[0][0]
            predictions['LSTM'] = lstm_pred

            # Train and predict with SVR
            svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
            svr_model.fit(X_other_train, y_train)
            svr_pred = svr_model.predict(X_other_test[-1:])
            predictions['SVR'] = svr_pred[0]

            # Train and predict with Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_other_train, y_train)
            rf_pred = rf_model.predict(X_other_test[-1:])
            predictions['Random Forest'] = rf_pred[0]

            # Train and predict with XGBoost
            xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
            xgb_model.fit(X_other_train, y_train)
            xgb_pred = xgb_model.predict(X_other_test[-1:])
            predictions['XGBoost'] = xgb_pred[0]

            # Train and predict with KNN
            knn_model = KNeighborsRegressor(n_neighbors=5)
            knn_model.fit(X_other_train, y_train)
            knn_pred = knn_model.predict(X_other_test[-1:])
            predictions['KNN'] = knn_pred[0]

            # Train and predict with GBM
            gbm_model = GradientBoostingRegressor(random_state=42)
            gbm_model.fit(X_other_train, y_train)
            gbm_pred = gbm_model.predict(X_other_test[-1:])
            predictions['GBM'] = gbm_pred[0]

            # Train and predict with ARIMA
            try:
                close_prices = df['close'].values
                arima_model = ARIMA(close_prices, order=(5,1,0))
                arima_fit = arima_model.fit()
                arima_pred = arima_fit.forecast(steps=1)[0]
                arima_scaled = (arima_pred - df['close'].mean()) / df['close'].std()
                predictions['ARIMA'] = arima_scaled
            except Exception as e:
                st.warning(f"ARIMA prediction failed: {str(e)}")

            weights = self.weights

            ## Adjust weights if some models failed
            # available_models = list(predictions.keys())
            # total_weight = sum(weights[model] for model in available_models)
            # adjusted_weights = {model: weights[model]/total_weight for model in available_models}

            # ensemble_pred = sum(pred * adjusted_weights[model] 
            #                   for model, pred in predictions.items())

        # Adjust weights if some models failed
            valid_predictions = {model: float(pred) for model, pred in predictions.items() if np.isscalar(pred) and pred >= 0}
            if not valid_predictions:
                st.error("All model predictions are negative, unable to compute ensemble prediction.")
            else:
                total_weight = sum(weights[model] for model in valid_predictions.keys())
                adjusted_weights = {model: weights[model]/total_weight for model in valid_predictions.keys()}

                # Tính toán trung bình có trọng số chỉ với các model hợp lệ
                ensemble_pred = sum(pred * adjusted_weights[model] for model, pred in valid_predictions.items())

            
            # Inverse transform predictions
            dummy_array = np.zeros((1, X_other.shape[1]))
            dummy_array[0, 0] = ensemble_pred
            final_prediction = self.scaler.inverse_transform(dummy_array)[0, 0]

            # Calculate prediction range
            individual_predictions = []
            for pred in predictions.values():
                dummy = dummy_array.copy()
                dummy[0, 0] = pred
                individual_predictions.append(
                    self.scaler.inverse_transform(dummy)[0, 0]
                )
            
            std_dev = np.std(individual_predictions)
            
            return {
                'prediction': final_prediction,
                'lower_bound': final_prediction - std_dev,
                'upper_bound': final_prediction + std_dev,
                'confidence_score': 1 / (1 + std_dev / final_prediction),
                'individual_predictions': {
                    model: pred for model, pred in zip(predictions.keys(), individual_predictions)
                }
            }

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None

# Streamlit interface

symbol = st.text_input("Enter VietNam Stock Symbol (e.g., VND):", "VND")
display_days = st.slider("Select number of days to display", 30, 3650, 180)

# Define different weight configurations
WEIGHT_CONFIGURATIONS = {
    "Default": {
        'LSTM': 0.3,
        'XGBoost': 0.15,
        'Random Forest': 0.15,
        'ARIMA': 0.1,
        'SVR': 0.1,
        'GBM': 0.1,
        'KNN': 0.1
    },
    "Trend-Focused": {
        'LSTM': 0.35,
        'XGBoost': 0.20,
        'Random Forest': 0.15,
        'ARIMA': 0.10,
        'SVR': 0.08,
        'GBM': 0.07,
        'KNN': 0.05
    },
    "Statistical": {
        'LSTM': 0.20,
        'XGBoost': 0.15,
        'Random Forest': 0.15,
        'ARIMA': 0.20,
        'SVR': 0.15,
        'GBM': 0.10,
        'KNN': 0.05
    },
    "Tree-Ensemble": {
        'LSTM': 0.25,
        'XGBoost': 0.25,
        'Random Forest': 0.20,
        'ARIMA': 0.10,
        'SVR': 0.08,
        'GBM': 0.07,
        'KNN': 0.05
    },
    "Balanced": {
        'LSTM': 0.25,
        'XGBoost': 0.20,
        'Random Forest': 0.15,
        'ARIMA': 0.15,
        'SVR': 0.10,
        'GBM': 0.10,
        'KNN': 0.05
    },
    "Volatility-Focused": {
        'LSTM': 0.30,
        'XGBoost': 0.25,
        'Random Forest': 0.20,
        'ARIMA': 0.05,
        'SVR': 0.10,
        'GBM': 0.07,
        'KNN': 0.03
    }
}

WEIGHT_DESCRIPTIONS = {
    "Default": "Original configuration with balanced weights",
    "Trend-Focused": "Best for growth stocks, tech stocks, clear trend patterns",
    "Statistical": "Best for blue chip stocks, utilities, stable dividend stocks",
    "Tree-Ensemble": "Best for stocks with complex relationships to market factors",
    "Balanced": "Best for general purpose, unknown stock characteristics",
    "Volatility-Focused": "Best for small cap stocks, emerging market stocks, crypto-related stocks"
}

col1, col2 = st.columns([2, 1])

with col1:
    selected_weight = st.selectbox(
        "Select Weight Configuration:",
        options=list(WEIGHT_CONFIGURATIONS.keys()),
        help="Choose different weight configurations for the prediction models"
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
        weight_df = pd.DataFrame(list(current_weights.items()), columns=["Model", "Weight"])
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
    st.markdown("""
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
    """, unsafe_allow_html=True)

    # HTML cho nút tùy chỉnh
    button_html = '<button class="custom-button">Generate Predictions</button>'

    # Hiển thị nút với Streamlit
    # st.markdown(button_html, unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        
        if st.button("Generate Predictions"):
            with st.spinner("Training multiple models and generating predictions..."):
                predictor = MultiAlgorithmStockPredictor(
                    symbol, 
                    weights=WEIGHT_CONFIGURATIONS[selected_weight]
                )
                results = predictor.predict_with_all_models()
                
                if results is not None:
                    
                    
                    last_price = float(df['close'].iloc[-1])
                    
                    
                    # Individual model predictions
                    st.subheader("Individual Model Predictions")
                    model_predictions = pd.DataFrame({
                        'Model': results['individual_predictions'].keys(),
                        'Predicted Price': [v for v in results['individual_predictions'].values()]
                    })
                    model_predictions['Deviation from Ensemble'] = (
                        model_predictions['Predicted Price'] - abs(results['prediction'])
                    )
                    model_predictions['Weight'] = (
                        WEIGHT_CONFIGURATIONS[selected_weight].values()
                    )
                    model_predictions = model_predictions.sort_values('Predicted Price', ascending=False)
                    currency = "VND"
                    # if ".vn" in symbol.lower(): 
                    #     currency = "VND"
                    st.dataframe(model_predictions.style.format({
                        'Predicted Price': f'{{:.2f}} {currency}',
                        'Deviation from Ensemble': f'{{:.2f}} {currency}',
                        'Weight': f'{{:.2f}}'
                    }))
                    # Hiển thị tiêu đề
                    st.subheader("🔥 Average Predicted Price")
                    # Lấy dữ liệu chứng khoán và giá hiện tại
                    current_price = df['close'].iloc[-1] if not df.empty else 0
                    # Tính biên lợi nhuận
                    profit_margin = (abs(results['prediction'] - current_price) / current_price) * 100 if current_price != 0 else 0
                    # Hiển thị 3 cột với tiêu đề và giá trị
                    col1, col2, col3 = st.columns(3)

                    # Cột 1: Giá dự đoán
                    with col1:
                        st.metric(label="Predicted Price", value=f"{abs(results['prediction']):.2f} VND")

                    # Cột 2: Giá hiện tại
                    with col2:
                        st.metric(label="Current Price", value=f"{current_price:.2f} VND")

                    # Cột 3: Phần trăm biên lợi nhuận
                    with col3:
                        st.metric(label="Profit Margin (%)", value=f"{profit_margin:.2f}%")


                    # Trading signal with confidence
                    price_change = ((results['prediction'] - last_price) / last_price) * 100
                    
                    # Create a prediction distribution plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    predictions = list(results['individual_predictions'].values())
                    models = list(results['individual_predictions'].keys())
                    
                    # Horizontal bar chart showing predictions
                    y_pos = np.arange(len(models))
                    ax.barh(y_pos, predictions)
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(models)
                    ax.axvline(x=last_price, color='r', linestyle='--', label='Current Price')
                    ax.axvline(x=results['prediction'], color='g', linestyle='--', label='Ensemble Prediction')
                    ax.set_xlabel('Price (VND*1000)')
                    ax.set_title('Model Predictions Comparison')
                    ax.legend()
                    
                    st.pyplot(fig)
                    
                    # Trading signal box
                    signal_box = st.container()
                    if abs(price_change) > 10:  # For very large changes
                        if price_change > 0:
                            signal_box.success(f"💹 Strong BUY Signal (+{price_change:.1f}%)")
                        else:
                            signal_box.error(f"📉 Strong SELL Signal ({price_change:.1f}%)")
                    elif abs(price_change) > 3 and results['confidence_score'] > 0.8:
                        if price_change > 0:
                            signal_box.success(f"💹 BUY Signal (+{price_change:.1f}%)")
                        else:
                            signal_box.error(f"📉 SELL Signal ({price_change:.1f}%)")
                    elif abs(price_change) > 2 and results['confidence_score'] > 0.6:
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
                    prediction_range = results['upper_bound'] - results['lower_bound']
                    risk_level = "Low" if prediction_std < last_price * 0.02 else \
                                "Medium" if prediction_std < last_price * 0.05 else "High"
                    
                    risk_col1, risk_col2 = st.columns(2)
                    with risk_col1:
                        currency = "VND"
                        # if ".vn" in symbol.lower(): 
                        #     currency = "VND"
                        st.metric("Prediction Volatility", f"{prediction_std:.2f} {currency}")
                    with risk_col2:
                        st.metric("Risk Level", risk_level)
    
        ####-chart stock-############################################
    # Đảm bảo 'Date' là kiểu datetime
    df['Date'] = pd.to_datetime(df['time'])

    # Vẽ biểu đồ với Matplotlib
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Đoạn này vẽ đường giá đóng cửa (Close Price)
    ax1.plot(df['Date'], df['close'], color='blue', label='Close Price', linewidth=2)

    # Tạo trục y thứ hai cho Volume
    ax2 = ax1.twinx()
    ax2.bar(df['Date'], df['volume'], color='gray', alpha=0.3, label='Volume', width=0.8)

    # Cập nhật tiêu đề và nhãn cho các trục
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (VND)', color='blue')
    ax2.set_ylabel('Volume', color='gray')

    # Cập nhật tiêu đề biểu đồ
    ax1.set_title('Stock Price and Volume')

    # Thêm legend cho cả hai trục
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # Cải thiện hiển thị của trục x (ngày tháng)
    fig.autofmt_xdate(rotation=45)  # Xoay ngày tháng để tránh bị chồng lên

    # Hiển thị biểu đồ trong Streamlit
    st.subheader("Stock Price Chart")
    st.pyplot(fig)
    ####-END-####################################################
    # st.subheader("Stock Price Chart")
    # st.line_chart(df['close'])
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.write("Detailed error information:", str(e))
st.markdown("---")  

