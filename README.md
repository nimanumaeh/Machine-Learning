# Stock Market Analysis and Prediction Using Machine Learning

This project aims to analyze and predict stock market trends using various machine learning models. It includes data collection, preprocessing, exploratory data analysis (EDA), and the implementation of ARIMA and LSTM models for time series forecasting. The project is designed to showcase skills in Python programming, data analysis, machine learning, and financial market understanding.

## Features

- **Data Collection**: Fetches historical stock data using Yahoo Finance API.
- **Data Preprocessing**: Cleans and prepares data for analysis, including handling missing values and calculating technical indicators like Moving Averages and RSI.
- **Exploratory Data Analysis (EDA)**: Performs statistical analysis and visualizations to understand stock price trends and volume.
- **ARIMA Model**: Implements the ARIMA model for time series forecasting, including stationarity checks and model fitting.
- **LSTM Model**: Utilizes Long Short-Term Memory (LSTM) networks for predicting stock prices, a type of recurrent neural network suitable for sequence prediction problems.
- **Model Comparison**: Compares the performance of ARIMA and LSTM models using Mean Squared Error (MSE) and visualizations.

## Usage

1. **Data Collection**: Specify the stock ticker, start, and end dates to fetch data.
2. **Data Preprocessing**: Run the preprocessing function on the collected data.
3. **Exploratory Data Analysis**: Analyze the preprocessed data using various EDA techniques.
4. **Model Implementation**: Fit both ARIMA and LSTM models on the stock data.
5. **Model Comparison**: Compare the models based on their predictions and error metrics.

## Libraries Used

- Libraries: yfinance, pandas, numpy, matplotlib, seaborn, statsmodels, pmdarima, keras, sklearn
