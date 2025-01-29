
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

def run_gbm(data):
    feature_columns = ['MA5', 'MA20', 'MA50', 'RSI', 'MACD', 'ATR', 'BB_upper', 'BB_lower']
    X = data[feature_columns]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    return model.predict(X_test.iloc[-1:].values)[0]
