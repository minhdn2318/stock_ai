# Define different weight configurations
WEIGHT_CONFIGURATIONS = {
    "Default": {
        "LSTM": 0.3,
        "XGBoost": 0.15,
        "Random Forest": 0.15,
        "ARIMA": 0.1,
        "SVR": 0.1,
        "GBM": 0.1,
        "KNN": 0.1,
    },
    "Trend-Focused": {
        "LSTM": 0.35,
        "XGBoost": 0.20,
        "Random Forest": 0.15,
        "ARIMA": 0.10,
        "SVR": 0.08,
        "GBM": 0.07,
        "KNN": 0.05,
    },
    "Statistical": {
        "LSTM": 0.20,
        "XGBoost": 0.15,
        "Random Forest": 0.15,
        "ARIMA": 0.20,
        "SVR": 0.15,
        "GBM": 0.10,
        "KNN": 0.05,
    },
    "Tree-Ensemble": {
        "LSTM": 0.25,
        "XGBoost": 0.25,
        "Random Forest": 0.20,
        "ARIMA": 0.10,
        "SVR": 0.08,
        "GBM": 0.07,
        "KNN": 0.05,
    },
    "Balanced": {
        "LSTM": 0.25,
        "XGBoost": 0.20,
        "Random Forest": 0.15,
        "ARIMA": 0.15,
        "SVR": 0.10,
        "GBM": 0.10,
        "KNN": 0.05,
    },
    "Volatility-Focused": {
        "LSTM": 0.30,
        "XGBoost": 0.25,
        "Random Forest": 0.20,
        "ARIMA": 0.05,
        "SVR": 0.10,
        "GBM": 0.07,
        "KNN": 0.03,
    },
}

WEIGHT_CONFIGURATIONS_BETA = {
    "Default": {
        "LSTM": 0.4,
        "XGBoost": 0.2,
        "Random Forest": 0.2,
        "ARIMA": 0.05,
        "SVR": 0.1,
        "GBM": 0.1,
        "KNN": 0.05,
    },
    "Trend-Focused": {
        "LSTM": 0.4,
        "XGBoost": 0.20,
        "Random Forest": 0.15,
        "ARIMA": 0.15,
        "SVR": 0.03,
        "GBM": 0.03,
        "KNN": 0.06,
    },
    "Statistical": {
        "LSTM": 0.20,
        "XGBoost": 0.15,
        "Random Forest": 0.15,
        "ARIMA": 0.20,
        "SVR": 0.15,
        "GBM": 0.10,
        "KNN": 0.05,
    },
    "Tree-Ensemble": {
        "LSTM": 0.25,
        "XGBoost": 0.25,
        "Random Forest": 0.25,
        "ARIMA": 0.05,
        "SVR": 0.08,
        "GBM": 0.07,
        "KNN": 0.05,
    },
    "Balanced": {
        "LSTM": 0.25,
        "XGBoost": 0.20,
        "Random Forest": 0.15,
        "ARIMA": 0.10,
        "SVR": 0.10,
        "GBM": 0.10,
        "KNN": 0.10,
    },
    "Volatility-Focused": {
        "LSTM": 0.40,
        "XGBoost": 0.25,
        "Random Forest": 0.15,
        "ARIMA": 0.05,
        "SVR": 0.05,
        "GBM": 0.07,
        "KNN": 0.03,
    },
}

WEIGHT_DESCRIPTIONS = {
    "Default": "Original configuration with balanced weights",
    "Trend-Focused": "Best for growth stocks, tech stocks, clear trend patterns",
    "Statistical": "Best for blue chip stocks, utilities, stable dividend stocks",
    "Tree-Ensemble": "Best for stocks with complex relationships to market factors",
    "Balanced": "Best for general purpose, unknown stock characteristics",
    "Volatility-Focused": "Best for small cap stocks, emerging market stocks, crypto-related stocks",
}