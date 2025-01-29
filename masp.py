# Sử dụng các thư viện.
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
import yfinance as yf
from weights import WEIGHT_CONFIGURATIONS

# Nếu IDE cảnh báo `import ... cannot be resolved`, hãy bỏ qua.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping


class MultiAlgorithmStockPredictor:
    def __init__(self, symbol, training_years=5, weights=None):
        self.symbol = symbol
        self.training_years = training_years
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.weights = (
            weights if weights is not None else WEIGHT_CONFIGURATIONS["Default"]
        )

    def fetch_historical_data(self):
        # Same as original EnhancedStockPredictor
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * self.training_years)

        try:
            df = yf.download(self.symbol, start=start_date, end=end_date)
            if df.empty:
                st.warning(
                    f"Data for the last {self.training_years} years is unavailable. Fetching maximum available data instead."
                )
                df = yf.download(self.symbol, period="max")
            return df
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return yf.download(self.symbol, period="max")

    # Technical indicators calculation methods remain the same
    def calculate_technical_indicators(self, df):
        # Original technical indicators remain the same
        df["MA5"] = df["Close"].rolling(window=5).mean()
        df["MA20"] = df["Close"].rolling(window=20).mean()
        df["MA50"] = df["Close"].rolling(window=50).mean()
        df["MA200"] = df["Close"].rolling(window=200).mean()
        df["RSI"] = self.calculate_rsi(df["Close"])
        df["MACD"] = self.calculate_macd(df["Close"])
        df["ROC"] = df["Close"].pct_change(periods=10) * 100
        df["ATR"] = self.calculate_atr(df)
        df["BB_upper"], df["BB_lower"] = self.calculate_bollinger_bands(df["Close"])
        df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
        df["Volume_Rate"] = df["Volume"] / df["Volume"].rolling(window=20).mean()

        # Additional technical indicators
        df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
        df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
        df["MOM"] = df["Close"].diff(10)
        df["STOCH_K"] = self.calculate_stochastic(df)
        df["WILLR"] = self.calculate_williams_r(df)

        return df.dropna()

    @staticmethod
    def calculate_stochastic(df, period=14):
        low_min = df["Low"].rolling(window=period).min()
        high_max = df["High"].rolling(window=period).max()
        k = 100 * ((df["Close"] - low_min) / (high_max - low_min))
        return k

    @staticmethod
    def calculate_williams_r(df, period=14):
        high_max = df["High"].rolling(window=period).max()
        low_min = df["Low"].rolling(window=period).min()
        return -100 * ((high_max - df["Close"]) / (high_max - low_min))

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
        high_low = df["High"] - df["Low"]
        high_close = np.abs(df["High"] - df["Close"].shift())
        low_close = np.abs(df["Low"] - df["Close"].shift())
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
        feature_columns = [
            "Close",
            "MA5",
            "MA20",
            "MA50",
            "MA200",
            "RSI",
            "MACD",
            "ROC",
            "ATR",
            "BB_upper",
            "BB_lower",
            "Volume_Rate",
            "EMA12",
            "EMA26",
            "MOM",
            "STOCH_K",
            "WILLR",
        ]

        # Scale features
        scaled_data = self.scaler.fit_transform(df[feature_columns])

        # Prepare sequences for LSTM
        X_lstm, y = [], []
        for i in range(seq_length, len(scaled_data)):
            X_lstm.append(scaled_data[i - seq_length : i])
            y.append(scaled_data[i, 0])  # 0 index represents Close price

        # Prepare data for other models
        X_other = scaled_data[seq_length:]

        return np.array(X_lstm), X_other, np.array(y)

    def build_lstm_model(self, input_shape):
        model = Sequential(
            [
                Bidirectional(
                    LSTM(100, return_sequences=True), input_shape=input_shape
                ),
                Dropout(0.2),
                Bidirectional(LSTM(50, return_sequences=True)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25, activation="relu"),
                Dropout(0.1),
                Dense(10, activation="relu"),
                Dense(1),
            ]
        )
        model.compile(optimizer="adam", loss="huber", metrics=["mae"])
        return model

    def train_arima(self, df):
        model = ARIMA(df["Close"], order=(5, 1, 0))
        return model.fit()

    def predict_with_all_models(self, prediction_days=30, sequence_length=60):
        try:
            # Fetch and prepare data
            df = self.fetch_historical_data()

            # Check if we have enough data
            if (
                len(df) < sequence_length + 20
            ):  # Need extra days for technical indicators
                st.error(
                    f"Insufficient historical data. Need at least {sequence_length + 20} days of data."
                )
                return None

            # Calculate technical indicators
            df = self.calculate_technical_indicators(df)

            # Check for NaN values and handle them
            if df.isnull().any().any():
                df = df.fillna(method="ffill").fillna(method="bfill")

            # Verify we have enough valid data after cleaning
            if len(df.dropna()) < sequence_length:
                st.error("Insufficient valid data after calculating indicators.")
                return None

            # Prepare features
            feature_columns = [
                "Close",
                "MA5",
                "MA20",
                "MA50",
                "MA200",
                "RSI",
                "MACD",
                "ROC",
                "ATR",
                "BB_upper",
                "BB_lower",
                "Volume_Rate",
                "EMA12",
                "EMA26",
                "MOM",
                "STOCH_K",
                "WILLR",
            ]

            # Verify all required features exist
            missing_features = [col for col in feature_columns if col not in df.columns]
            if missing_features:
                st.error(f"Missing required features: {', '.join(missing_features)}")
                return None

            # Ensure we have valid data for all features
            df = df[feature_columns].dropna()
            if len(df) < sequence_length:
                st.error(
                    f"Insufficient valid data points after cleaning. Need at least {sequence_length} points."
                )
                st.write(f"Available data points: {len(df)}")
                return None

            try:
                # Scale features
                scaled_data = self.scaler.fit_transform(df[feature_columns])
            except ValueError as e:
                st.error(f"Scaling error: {str(e)}")
                st.write(
                    "This usually happens with newly listed stocks or stocks with insufficient trading history."
                )
                return None

            # Prepare sequences for LSTM
            X_lstm, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X_lstm.append(scaled_data[i - sequence_length : i])
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
            early_stopping = EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
            lstm_model.fit(
                X_lstm_train,
                y_train,
                epochs=50,
                batch_size=32,
                validation_data=(X_lstm_test, y_test),
                callbacks=[early_stopping],
                verbose=0,
            )
            lstm_pred = lstm_model.predict(X_lstm_test[-1:], verbose=0)[0][0]
            predictions["LSTM"] = lstm_pred

            # Train and predict with SVR
            svr_model = SVR(kernel="rbf", C=100, epsilon=0.1)
            svr_model.fit(X_other_train, y_train)
            svr_pred = svr_model.predict(X_other_test[-1:])
            predictions["SVR"] = svr_pred[0]

            # Train and predict with Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_other_train, y_train)
            rf_pred = rf_model.predict(X_other_test[-1:])
            predictions["Random Forest"] = rf_pred[0]

            # Train and predict with XGBoost
            xgb_model = XGBRegressor(objective="reg:squarederror", random_state=42)
            xgb_model.fit(X_other_train, y_train)
            xgb_pred = xgb_model.predict(X_other_test[-1:])
            predictions["XGBoost"] = xgb_pred[0]

            # Train and predict with KNN
            knn_model = KNeighborsRegressor(n_neighbors=5)
            knn_model.fit(X_other_train, y_train)
            knn_pred = knn_model.predict(X_other_test[-1:])
            predictions["KNN"] = knn_pred[0]

            # Train and predict with GBM
            gbm_model = GradientBoostingRegressor(random_state=42)
            gbm_model.fit(X_other_train, y_train)
            gbm_pred = gbm_model.predict(X_other_test[-1:])
            predictions["GBM"] = gbm_pred[0]

            # Train and predict with ARIMA
            try:
                close_prices = df["Close"].values
                arima_model = ARIMA(close_prices, order=(5, 1, 0))
                arima_fit = arima_model.fit()
                arima_pred = arima_fit.forecast(steps=1)[0]
                arima_scaled = (arima_pred - df["Close"].mean()) / df["Close"].std()
                predictions["ARIMA"] = arima_scaled
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
            valid_predictions = {
                model: float(pred)
                for model, pred in predictions.items()
                if np.isscalar(pred) and pred >= 0
            }
            if not valid_predictions:
                st.error(
                    "All model predictions are negative, unable to compute ensemble prediction."
                )
            else:
                total_weight = sum(weights[model] for model in valid_predictions.keys())
                adjusted_weights = {
                    model: weights[model] / total_weight
                    for model in valid_predictions.keys()
                }

                # Tính toán trung bình có trọng số chỉ với các model hợp lệ
                ensemble_pred = sum(
                    pred * adjusted_weights[model]
                    for model, pred in valid_predictions.items()
                )

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
                "prediction": final_prediction,
                "lower_bound": final_prediction - std_dev,
                "upper_bound": final_prediction + std_dev,
                "confidence_score": 1 / (1 + std_dev / final_prediction),
                "individual_predictions": {
                    model: pred
                    for model, pred in zip(predictions.keys(), individual_predictions)
                },
            }

        except Exception as e:
            st.error(f"Error in prediction: {str(e)}")
            return None
