import pandas as pd

from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

def calculate_golden_cross(df: pd.DataFrame):
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    df['golden_cross'] = (df['MA50'] > df['MA200']).astype(int)
    return df

def calculate_rsi(df: pd.DataFrame, period: int = 14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20):
    ma = df['Close'].rolling(window=window).mean()
    std = df['Close'].rolling(window=window).std()
    df['Upper Band'] = ma + 2 * std
    df['Lower Band'] = ma - 2 * std
    return df

def calculate_sniper(df: pd.DataFrame):
    # 간단화된 버전 (진짜 전략은 세부 규칙 필요)
    df['Sniper Signal'] = ((df['Close'].pct_change() > 0.03) & (df['Volume'] > df['Volume'].rolling(window=5).mean())).astype(int)
    return df

"""
    기술적 지표 활용: 기술적 지표를 기반으로 예측을 수행할 수 있습니다. 예를 들어, 이동 평균 교차, RSI, MACD 등을 사용하여 매수/매도 신호를 생성하고 이를 기반으로 예측할 수 있습니다.
"""
def predict_close_price_with_rf(df: pd.DataFrame):
    print(df)

    # 현재 날짜 가져오기
    today = datetime.now()

    # # 내일 날짜 계산
    # tomorrow = today + timedelta(days=1)

    # 미래 날짜 생성 (내일부터 시작)
    future_dates = pd.date_range(start=today, periods=5, freq='D')
    # 미래 데이터프레임 생성
    # future_df = pd.DataFrame({'Date': future_dates})
    future_df = pd.DataFrame({'Date': future_dates.strftime('%Y-%m-%d')})

    # 과거 데이터를 사용하여 미래의 MA50, MA200, RSI 추정
    future_df['MA20'] = df['Close'].iloc[-20:].mean()
    future_df['MA50'] = df['Close'].iloc[-50:].mean()
    future_df['MA200'] = df['Close'].iloc[-200:].mean() if len(df) >= 200 else df['Close'].mean()
    future_df['Golden Cross'] = (future_df['MA50'] > future_df['MA200']).astype(int)    
    future_df['RSI'] = 100 - (100 / (1 + df['Close'].diff().clip(lower=0).iloc[-14:].mean() / 
                                    -df['Close'].diff().clip(upper=0).iloc[-14:].mean()))
    future_df['stddev'] = df['Close'].iloc[-20:].std()
    future_df['Upper Band'] = future_df['MA20'] + (2 * future_df['stddev'])
    future_df['Lower Band'] = future_df['MA20'] - (2 * future_df['stddev'])
    future_df['Sniper Signal'] = (df['Close'].pct_change().iloc[-1] > 0.02).astype(int)

    # 모델 학습
    X = df.dropna()[['MA20', 'MA50', 'MA200', 'Golden Cross', 'RSI', 'stddev', 'Upper Band', 'Lower Band', 'Sniper Signal']]
    y = df.dropna()['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 미래 데이터에 대한 예측
    future_df['Close'] = model.predict(future_df[['MA20', 'MA50', 'MA200', 'Golden Cross', 'RSI', 'stddev', 'Upper Band', 'Lower Band', 'Sniper Signal']])

    print(future_df)

    # df와 future_df 결합
    df = pd.concat([df, future_df], ignore_index=True)

    print(df)

    # 결측치 처리
    # df.fillna(method='bfill', inplace=True)

    return df