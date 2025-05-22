import os
import pandas as pd
import numpy as np
import tensorflow as tf

from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

indicator_features = ['MA20', 'MA50', 'MA200', 'Golden Cross', 'RSI', 'stddev', 'Upper Band', 'Lower Band', 'Bollinger Breakout Upper', 'Bollinger Breakout Lower', 'MACD', 'Signal', 'Sniper Signal', 'Smart Sniper', 'Double Bottom', 'Double Top', 'Head and Shoulders', 'Inverse Head and Shoulders']

"""
    ta indicator 계산산
"""
def calc_price_with_ta(df: pd.DataFrame) -> pd.DataFrame:
    # 1. 골든 크로스 (Golden Cross)
    # df = calculate_golden_cross(df)
    df['MA20'] = SMAIndicator(close=df['Close'], window=20, fillna=True).sma_indicator()
    df['MA50'] = SMAIndicator(close=df['Close'], window=50, fillna=True).sma_indicator()
    df['MA200'] = SMAIndicator(close=df['Close'], window=200, fillna=True).sma_indicator()
    df['Golden Cross'] = ((df['MA50'] > df['MA200']) & (df['MA50'].shift(1) <= df['MA200'].shift(1))).astype(int)
    df["stddev"] = df["Close"].rolling(window=20).std(ddof=0) 

    # 결측치 처리
    # df['MA200'].fillna(0, inplace=True) 

    # 2. RSI (Relative Strength Index)
    rsi_indicator = RSIIndicator(close=df["Close"], window=14, fillna=False)
    df['RSI'] = rsi_indicator.rsi()

    # 앞에 게산 안되어진 데이터 채워넣음
    df["RSI"]  = df["RSI"].bfill()
    
    # 3. 볼린저 밴드 (Bollinger Bands)
    indicator_bb = BollingerBands(close=df['Close'], window=14, window_dev=2)
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

"""
    기술적 지표 활용: 기술적 지표를 기반으로 예측을 수행할 수 있습니다. 예를 들어, 이동 평균 교차, RSI, MACD 등을 사용하여 매수/매도 신호를 생성하고 이를 기반으로 예측할 수 있습니다.
"""
def predict_close_price_with_rf(df: pd.DataFrame):
    # 학습 데이터 구성
    train_df = df.dropna()

    # 지연 피처
    df['Close_t-1'] = df['Close'].shift(1)
    df['Close_t-2'] = df['Close'].shift(2)
    # 결측치(NA/NaN 값)를 제거
    df.dropna()

    df['LSTM Close'] = df['Close']
    df['CNN Close'] = df['Close']

    x = train_df[indicator_features]
    y = train_df['Close']

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_val)

    model.fit(x_train, y_train_log)

    # 현재 날짜 가져오기
    today = datetime.now()
     # 미래 예측 준비
    future_days = 5
    # 미래 날짜 생성 (내일부터 시작)
    future_dates = pd.date_range(start=today, periods=future_days, freq='D')
    # 미래 데이터프레임 생성
    future_df = pd.DataFrame({'Date': future_dates.strftime('%Y-%m-%d')})

    # 미래 예측
    future_preds = []
    for i in range(future_days):
        df = calc_price_with_ta(df)

        # 지연 피처
        df['Close_t-1'] = df['Close'].shift(1)
        df['Close_t-2'] = df['Close'].shift(2)

        latest = df.iloc[-1]

        ma200_value = latest['MA200']

        if pd.isna(ma200_value):
            ma200_value = latest['MA50'] 

        new_row = {
            'MA20': latest['MA20'],
            'MA50': latest['MA50'],
            'MA200': ma200_value,
            'Golden Cross': latest['Golden Cross'],
            'RSI': latest['RSI'],
            'stddev': latest['stddev'],
            'Upper Band': latest['Upper Band'],
            'Lower Band': latest['Lower Band'],
            'Bollinger Breakout Upper': latest['Bollinger Breakout Upper'], 
            'Bollinger Breakout Lower': latest['Bollinger Breakout Lower'],
            'MACD': latest['MACD'],
            'Signal': latest['Signal'],
            'Sniper Signal': latest['Sniper Signal'],
            'Smart Sniper': latest['Smart Sniper'],
            'Double Bottom': latest['Double Bottom'], 
            'Double Top': latest['Double Top'], 
            'Head and Shoulders': latest['Head and Shoulders'], 
            'Inverse Head and Shoulders': latest['Inverse Head and Shoulders']
        }

        # 지연 피처
        df['Close_t-1'] = df['Close'].shift(1)
        df['Close_t-2'] = df['Close'].shift(2)
        # 결측치(NA/NaN 값)를 제거
        df.dropna()

        predicted_close_log = model.predict(pd.DataFrame([new_row]))[0]
        predicted_close = np.expm1(predicted_close_log)

        lstm_cnn_close = calc_price_with_lstm_cnn(df)

        print(lstm_cnn_close)

        # NaN 값을 0으로 바꿔서 safe_dict 생성
        safe_row = {k: (0 if pd.isna(v) else v) for k, v in new_row.items()}

        future_preds.append({
            'Date': future_dates[i].strftime('%Y-%m-%d'),
            **safe_row,
            # 'Golden Cross': 0,
            'Close': predicted_close,
            'LSTM Close': lstm_cnn_close['lstm'],
            'CNN Close': lstm_cnn_close['cnn']
        })

        # # 예측값을 기존 df에 추가하여 다음날 지표 계산에 반영
        df = pd.concat([df, pd.DataFrame([{
            'Date': future_dates[i].strftime('%Y-%m-%d'),
            **new_row,
            # 'Golden Cross': 0,
            'Close': predicted_close,
            'LSTM Close': lstm_cnn_close['lstm'],
            'CNN Close': lstm_cnn_close['cnn']
        }])], ignore_index=True)

    # # 미래 예측 결과
    future_df = pd.DataFrame(future_preds)

    # df와 future_df 결합
    df = pd.concat([df, future_df], ignore_index=True)

    print(future_df)

    # NaN 값을 0으로 대체
    df = df.fillna(0)

    # 결측치 처리
    # df.fillna(method='bfill', inplace=True)

    return df

# 시퀀스 데이터 생성 함수 (ex. 20일씩 예측)
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def calc_price_with_lstm_cnn(df: pd.DataFrame):
    if len(df) <= 20:
        return { "lstm": df['Close'][-1].item(), "cnn": df['Close'][-1].item() }
    
    # 1. 데이터 스케일링 및 시퀀스 생성
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(df["Close"].values.reshape(-1, 1))
    X_seq, y_seq = create_sequences(scaled_close, window_size=20)

    # 2. train/test 분리
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    # 3. LSTM 모델
    lstm_model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], 1)),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mse')
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    # 4. CNN 모델
    cnn_model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(50, activation='relu'),
        Dense(1)
    ])
    cnn_model.compile(optimizer='adam', loss='mse')
    cnn_model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

    # 5. 예측 및 역변환
    lstm_pred = scaler.inverse_transform(lstm_model.predict(X_test))
    cnn_pred = scaler.inverse_transform(cnn_model.predict(X_test))
    # 실제 주가 값
    # y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

    return { "lstm": lstm_pred[-1].item(), "cnn": cnn_pred[-1].item() }