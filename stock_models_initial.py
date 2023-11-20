
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
import pmdarima as pm
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Data Collection Module
def fetch_data(ticker, start_date, end_date):
    """
    Fetches historical stock data for a given ticker from Yahoo Finance.
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data.to_csv(f"{ticker}_data.csv")
    print(f'Data for {ticker} saved to {ticker}_data.csv')

# Data Preprocessing Module
def preprocess_data(file_path):
    """
    Cleans and preprocesses the stock data.
    """
    data = pd.read_csv(file_path)
    data.fillna(method='ffill', inplace=True)
    data.fillna(method='bfill', inplace=True)
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA10'] = data['Close'].rolling(window=10).mean()
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    return data

# Exploratory Data Analysis Module (EDA)
def eda(data):
    """
    Conducts exploratory data analysis on the stock data.
    """

    # Histogram of Closing Prices
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Close'], kde=True)
    plt.title('Histogram of Closing Prices')
    plt.xlabel('Closing Price')
    plt.ylabel('Frequency')
    plt.show()

    # Scatterplot of Closing Price vs Volume
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Close', y='Volume', data=data)
    plt.title('Closing Price vs Volume')
    plt.xlabel('Closing Price')
    plt.ylabel('Volume')
    plt.show()

    # Statistical Summary
    print('Statistical Summary:\n', data.describe())

    # Correlation Matrix
    plt.figure(figsize=(10, 6))
    # Select only numeric columns for correlation
    numeric_data = data.select_dtypes(include=['number'])
    sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

# ARIMA Model Module
def check_stationarity(data):
    """
    Performs the Augmented Dickey-Fuller test to check the stationarity of the time series.
    """
    result = adfuller(data.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    if result[1] > 0.05:
        print('Series is not stationary')
    else:
        print('Series is stationary')

def fit_arima_model(data, order=None):
    """
    Fits an ARIMA model to the time series data.
    """
    if order is None:
        # Using pmdarima's auto_arima to find the best order
        auto_model = pm.auto_arima(data, seasonal=False, stepwise=True, suppress_warnings=True, error_action='ignore', max_order=None, trace=True)
        order = auto_model.order
        model = ARIMA(data, order=order)
    else:
        # Directly using statsmodels ARIMA with provided order
        model = ARIMA(data, order=order)

    fitted_model = model.fit()
    print(f'Fitted ARIMA model with order {order}')
    return fitted_model

# LSTM Model Module
def prepare_lstm_data(data, time_step=1):
    """
    Prepares the data for LSTM model.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, Y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        a = scaled_data[i:(i + time_step), 0]
        X.append(a)
        Y.append(scaled_data[i + time_step, 0])
    return np.array(X), np.array(Y), scaler

def build_lstm_model(input_shape):
    """
    Builds and returns an LSTM model.
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Model Comparison and Visualization Module
def compare_models(actual, arima_predictions, lstm_predictions):
    """
    Compares the performance of ARIMA and LSTM models.
    """
    # Ensure all arrays are of the same length
    min_length = min(len(actual), len(arima_predictions), len(lstm_predictions))
    actual = actual[:min_length]
    arima_predictions = arima_predictions[:min_length]
    lstm_predictions = lstm_predictions[:min_length]

    # Calculate Mean Squared Error
    arima_mse = mean_squared_error(actual, arima_predictions)
    lstm_mse = mean_squared_error(actual, lstm_predictions)

    # Output results
    print(f'ARIMA Model MSE: {arima_mse}')
    print(f'LSTM Model MSE: {lstm_mse}')

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(actual, label='Actual')
    plt.plot(arima_predictions, label='ARIMA Predictions')
    plt.plot(lstm_predictions, label='LSTM Predictions')
    plt.title('Model Comparisons')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()

# Main Script
def main():
    """
    Main function to run the analysis.
    """
    ticker = 'AAPL'
    start_date = '2020-01-01'
    end_date = '2021-01-01'
    fetch_data(ticker, start_date, end_date)
    data = preprocess_data(f'{ticker}_data.csv')
    eda(data)
    check_stationarity(data['Close'])
    arima_model = fit_arima_model(data['Close'])
    arima_predictions = arima_model.predict(start=0, end=len(data))
    X, Y, scaler = prepare_lstm_data(data[['Close']].values)
    lstm_model = build_lstm_model((X.shape[1], 1))
    lstm_model.fit(X, Y, epochs=100, batch_size=32, verbose=0)
    lstm_predictions = lstm_model.predict(X)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)
    compare_models(data['Close'].values, arima_predictions, lstm_predictions)

if __name__ == "__main__":
    main()

