import streamlit as st
from datetime import datetime, timedelta
from vnstock import Vnstock
import yfinance as yf
import logging

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