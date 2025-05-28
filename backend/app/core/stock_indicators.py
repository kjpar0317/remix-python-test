import pandas as pd
import numpy as np
# import tensorflow as tf
import logging

from typing import List
from datetime import datetime
from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from scipy.signal import argrelextrema

from app.schemas.stock import TunningPoint

logger = logging.getLogger(__name__)

indicator_features = ['MA20', 'MA50', 'MA200', 'Golden Cross', 'RSI', 'stddev', 'Upper Band', 'Lower Band', 'Bollinger Breakout Upper', 'Bollinger Breakout Lower', 'MACD', 'Signal', 'Sniper Signal', 'Smart Sniper', 'Double Bottom', 'Double Top', 'Head and Shoulders', 'Inverse Head and Shoulders']

"""
    ta indicator 계산산
"""
def calc_price_with_ta(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 골든 크로스 (Golden Cross)
    df['MA20'] = SMAIndicator(close=df['Close'], window=20, fillna=True).sma_indicator()
    df['MA50'] = SMAIndicator(close=df['Close'], window=50, fillna=True).sma_indicator()
    df['MA200'] = SMAIndicator(close=df['Close'], window=200, fillna=True).sma_indicator()
    df['Golden Cross'] = ((df['MA50'] > df['MA200']) & (df['MA50'].shift(1) <= df['MA200'].shift(1))).astype(int)
    df["stddev"] = df["Close"].rolling(window=20).std(ddof=0) 

    # 2. RSI (Relative Strength Index)
    rsi_indicator = RSIIndicator(close=df["Close"], window=14, fillna=False)
    df['RSI'] = rsi_indicator.rsi()

    # 앞에 게산 안되어진 데이터 채워넣음
    df["RSI"]  = df["RSI"].bfill()
    
    # 3. 볼린저 밴드 (Bollinger Bands)
    indicator_bb = BollingerBands(close=df['Close'], window=14, window_dev=2, fillna=True)
    df['Avg Band'] = indicator_bb.bollinger_mavg()
    df['Upper Band'] = indicator_bb.bollinger_hband()
    df['Lower Band'] = indicator_bb.bollinger_lband()

    # 앞에 게산 안되어진 데이터 채워넣음
    df["Avg Band"]  = df["Avg Band"].bfill()
    df["Upper Band"] = df["Upper Band"].bfill()
    df["Lower Band"] = df["Lower Band"].bfill()

    df["Bollinger Breakout Upper"] = False
    df["Bollinger Breakout Lower"] = False

    if df['Close'].iloc[-1] > indicator_bb.bollinger_hband().iloc[-1]:
        df["Bollinger Breakout Upper"] = True
    elif df['Close'].iloc[-1] < indicator_bb.bollinger_lband().iloc[-1]:
        df["Bollinger Breakout Lower"] = True

    # 4. 스나이퍼 매매법 (Sniper Trading)
    df['Sniper Signal'] = (
        (df['Close'].pct_change() > 0.03) & 
        (df['Volume'] > df['Volume'].rolling(window=5).mean())
    ).astype(int)

    # 5. Smart Sniper
    df["Smart Sniper"] = (
        (df['Close'].pct_change() > 0.03) &
        (df['Volume'] > df['Volume'].rolling(window=5).mean()) &
        (df["RSI"] < 30)
    ).astype(int)

    df['EMA_short'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_long'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_short'] - df['EMA_long']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # 이중 바닥 / 이중 천장
    close = df['Close'].values[-20:]
    troughs = (np.diff(np.sign(np.diff(close))) > 0).nonzero()[0] + 1
    peaks = (np.diff(np.sign(np.diff(close))) < 0).nonzero()[0] + 1

    df["Double Bottom"] = False
    df["Double Top"]  = False

    if len(troughs) >= 2:
        if abs(close[troughs[-1]] - close[troughs[-2]]) / close[troughs[-1]] < 0.03:
            df["Double Bottom"] = True

    if len(peaks) >= 2:
        if abs(close[peaks[-1]] - close[peaks[-2]]) / close[peaks[-1]] < 0.03:
            df["Double Top"] = True

    df["Head and Shoulders"] = False
    df["Inverse Head and Shoulders"]  = False
    # 간단한 헤드앤숄더 탐지 (최근 7개 기준)
    c = close[-7:]
    if len(c) == 7:
        if c[0] < c[1] and c[1] < c[2] and c[2] > c[3] and c[3] > c[4] and c[4] < c[5] and abs(c[0] - c[5]) < 0.03 * c[2]:
            df["Head and Shoulders"] = True
        if c[0] > c[1] and c[1] > c[2] and c[2] < c[3] and c[3] < c[4] and c[4] > c[5] and abs(c[0] - c[5]) < 0.03 * c[2]:
            df["Inverse Head and Shoulders"] = True

    return df

# 마지막 행에서 예측에 필요한 피처 추출
def extract_features_from_last(df: pd.DataFrame) -> list[float]:
    last_row = df.iloc[-1]
    return [last_row[feature] for feature in indicator_features]

# Random Forest 예측
def predict_close_price_with_rf(df: pd.DataFrame, predict_days: int = 5) -> list[float]:
    df = calc_price_with_ta(df.copy())

    target_col = "Close"

    df["LSTM Close"] = df[target_col]
    df["CNN Close"] = df[target_col]

    x = df[indicator_features]
    y = df[target_col]

    # 모델 학습
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_val)

    rf_model.fit(x_train, y_train_log)

    # 학습은 딱 한 번만
    lstm_model, cnn_model, scaler = train_lstm_cnn_model(df)

    # 현재 날짜 가져오기
    today = datetime.now()
    # 미래 날짜 생성 (내일부터 시작)
    future_dates = pd.date_range(start=today, periods=predict_days, freq='D')

    for i in range(predict_days):
        df = calc_price_with_ta(df)
        features = extract_features_from_last(df)
        features_df = pd.DataFrame([features], columns=indicator_features)

        predicted_price_log = rf_model.predict(features_df)[0]
        predicted_close = np.expm1(predicted_price_log)

        scaler = MinMaxScaler()
        scaled_close = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
        x_last = scaled_close[-20:].reshape(1, 20, 1)
        lstm_pred = scaler.inverse_transform(lstm_model.predict(x_last))[0, 0]
        cnn_pred = scaler.inverse_transform(cnn_model.predict(x_last))[0, 0]
        
        # NaN 값을 0으로 바꿔서 safe_dict 생성
        # safe_row = {k: (0 if pd.isna(v) else v) for k, v in features_df.items()}

        df = pd.concat([df, pd.DataFrame([{
            'Date': future_dates[i].strftime('%Y-%m-%d'),
            **features_df.iloc[0].to_dict(),
            'Close': predicted_close,
            'LSTM Close': lstm_pred,
            'CNN Close': cnn_pred
        }])], ignore_index=True)

    df = df.fillna(0)

    return df

# 시퀀스 데이터 생성 함수 (ex. 20일씩 예측)
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def train_lstm_cnn_model(df: pd.DataFrame):
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
    X_seq, y_seq = create_sequences(scaled_close, window_size=20)

    # train/test 분리
    split = int(len(X_seq) * 0.8)
    X_train, y_train = X_seq[:split], y_seq[:split]

    # LSTM
    lstm_model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])

    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    # CNN
    cnn_model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        Conv1D(64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])

    cnn_model.compile(optimizer='adam', loss='mse')
    cnn_model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    return lstm_model, cnn_model, scaler

"""
    변곡점 후보 탐지
"""
def calc_tunning_point(df: pd.DataFrame) -> List[TunningPoint]:
    # N일 기준 국소 최저점/최고점 (변곡점 후보)
    n = 5  # n일 기준으로 국소적인 저점/고점 판단
    new_df = pd.DataFrame({
        'date': df['Date'],
        'min': None,
        'max': None,
        'close': df['Close']
    })
    
    new_df['min'] = df['Close'][argrelextrema(df['Close'].values, np.less_equal, order=n)[0]]
    new_df['max'] = df['Close'][argrelextrema(df['Close'].values, np.greater_equal, order=n)[0]]

    # 변곡점 리스트 만들기
    turning_points = []
    prev_type = None

    for i, row in new_df.iterrows():
        point_type = None
        if not np.isnan(row['min']):
            point_type = 'buy'
        elif not np.isnan(row['max']):
            point_type = 'sell'
        
        if point_type and point_type != prev_type:
            turning_points.append({
                'date': row['date'],
                'type': point_type,
                'price': row['close']
            })
            prev_type = point_type

    return turning_points