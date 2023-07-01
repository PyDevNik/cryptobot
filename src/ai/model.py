print("Loading...")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import ccxt

exchange = ccxt.binance()  
symbol = 'BTC/USDT' 
timeframe = '1d' 

print("Start!")

def fetch_historical_data(exchange: ccxt.binance, symbol: str, timeframe: str) -> pd.DataFrame:
    data = exchange.fetch_ohlcv(symbol, timeframe, limit = 4*60*60)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    return df

df = fetch_historical_data(exchange, symbol, timeframe)

def normalize_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(data.reshape(-1, 1))
    return normalized_data, scaler

prices_scaled, scaler = normalize_data(df['close'].values)

def split_data(data, train_size):
    train_size = int(len(data) * train_size)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

train_data, test_data = split_data(prices_scaled, train_size=0.8)

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 10
n_features = 1

X_train, y_train = create_sequences(train_data, sequence_length)
X_test, y_test = create_sequences(test_data, sequence_length)

def build_model(sequence_length, n_features):
    print("Build...")
    model = Sequential()
    model.add(LSTM(units=1000, return_sequences=True, input_shape=(sequence_length, n_features)))
    model.add(LSTM(units=1000))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    print("Model built!")
    return model
cd Desktop
model = build_model(sequence_length, n_features)
model.fit(X_train, y_train, epochs=10, batch_size=32)

y_pred = model.predict(X_test)

def inverse_normalize_data(data, scaler):
    return scaler.inverse_transform(data)

y_pred = inverse_normalize_data(y_pred, scaler)
y_test = inverse_normalize_data(y_test, scaler)

def plot_results(actual_prices, predicted_prices):
    plt.plot(actual_prices, label='Actual Price')
    plt.plot(predicted_prices, label='Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error (MSE):', mse)
plot_results(y_test, y_pred)
