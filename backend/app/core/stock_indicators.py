import pandas as pd

from ta.trend import SMAIndicator
from ta.momentum import RSIIndicator

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