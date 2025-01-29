
from statsmodels.tsa.arima.model import ARIMA

def run_arima(data):
    series = data['Close']
    model = ARIMA(series, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)
    return forecast.iloc[0]
