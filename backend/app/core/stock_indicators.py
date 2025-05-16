import pandas as pd

import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

"""
    기술적 지표 활용: 기술적 지표를 기반으로 예측을 수행할 수 있습니다. 예를 들어, 이동 평균 교차, RSI, MACD 등을 사용하여 매수/매도 신호를 생성하고 이를 기반으로 예측할 수 있습니다.
"""
def predict_close_price_with_rf(df: pd.DataFrame):
    # 학습 데이터 구성
    features = ['MA20', 'MA50', 'MA200', 'RSI', 'stddev', 'Upper Band', 'Lower Band', 'Sniper Signal', 'Smart Sniper']
    train_df = df.dropna()

    # 지연 피처
    df['Close_t-1'] = df['Close'].shift(1)
    df['Close_t-2'] = df['Close'].shift(2)
    # 결측치(NA/NaN 값)를 제거
    df.dropna()

    X = train_df[features]
    y = train_df['Close']
    # X = df.drop(['Close'], axis=1)
    # X = train_df.drop(columns=["Close", "Date"]) 
    # y = train_df['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    # model = lgb.LGBMRegressor(
    #     n_estimators=500,
    #     learning_rate=0.05,
    #     max_depth=6,
    #     random_state=42,
    #     predict_disable_shape_check=True
    # )
    model.fit(X_train, y_train)

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
        # 현재 df에 지표 재계산 (calculate_features 함수는 지표를 계산해 df 컬럼으로 넣는 함수)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        df['MA200'] = df['Close'].rolling(window=200).mean()

        delta = df['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss

        df['RSI'] = 100 - (100 / (1 + rs))
        df['stddev'] = df['Close'].rolling(window=20).std()
        df['Upper Band'] = df['MA20'] + 2 * df['stddev']
        df['Lower Band'] = df['MA20'] - 2 * df['stddev']
        df['Sniper Signal'] = (df['Close'].pct_change() > 0.02).astype(int)
        df["Smart Sniper"] = (
            (df['Close'].pct_change() > 0.03) &
            (df['Volume'] > df['Volume'].rolling(window=5).mean()) &
            (df["RSI"] < 30)
        ).astype(int)  

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
            'RSI': latest['RSI'],
            'stddev': latest['stddev'],
            'Upper Band': latest['Upper Band'],
            'Lower Band': latest['Lower Band'],
            'Sniper Signal': latest['Sniper Signal'],
            'Smart Sniper': latest['Smart Sniper']
        }

        # 지연 피처
        df['Close_t-1'] = df['Close'].shift(1)
        df['Close_t-2'] = df['Close'].shift(2)
        # 결측치(NA/NaN 값)를 제거
        df.dropna()

        X = train_df[features]
        y = train_df['Close']
        # X = df.drop(['Close'], axis=1)
        # X = train_df.drop(columns=["Close", "Date"]) 
        # y = train_df['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        # model = lgb.LGBMRegressor(
        #     n_estimators=500,
        #     learning_rate=0.05,
        #     max_depth=6,
        #     random_state=42,
        #     predict_disable_shape_check=True
        # )
        model.fit(X_train, y_train)

        predicted_close = model.predict(pd.DataFrame([new_row]))[0]

        # NaN 값을 0으로 바꿔서 safe_dict 생성
        safe_row = {k: (0 if pd.isna(v) else v) for k, v in new_row.items()}

        future_preds.append({
            'Date': future_dates[i].strftime('%Y-%m-%d'),
            **safe_row,
            'Golden Cross': 0,
            'Close': predicted_close
        })

        # # 예측값을 기존 df에 추가하여 다음날 지표 계산에 반영
        df = pd.concat([df, pd.DataFrame([{
            'Date': future_dates[i].strftime('%Y-%m-%d'),
            **new_row,
            'Golden Cross': 0,
            'Close': predicted_close
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