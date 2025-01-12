import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import warnings
import plotly.graph_objects as go
from tabulate import tabulate

warnings.filterwarnings("ignore")


# Fetch stock data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    stock_data = stock.history(period="max")
    if stock_data.empty:
        return None, None
    stock_data['Date'] = stock_data.index
    stock_data.reset_index(drop=True, inplace=True)
    company_info = stock.info
    market_cap = company_info.get('marketCap', 'N/A')
    return stock_data, market_cap


# Train models and predict future prices
def train_models(stock_data):
    stock_data['Prediction'] = stock_data['Close'].shift(-1)  # Target is the next day's closing price
    stock_data.dropna(inplace=True)

    X = stock_data[['Open', 'High', 'Low', 'Volume']]
    y = stock_data['Prediction']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    stock_data['LR_Prediction'] = lr_model.predict(X)

    # Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    stock_data['RF_Prediction'] = rf_model.predict(X)

    # XGBoost Regressor
    xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    stock_data['XGB_Prediction'] = xgb_model.predict(X)

    # Predict next 2 days' prices using the last row of features
    next_day_features = X.iloc[-1:].values
    lr_next_days = lr_model.predict(np.repeat(next_day_features, 2, axis=0))
    rf_next_days = rf_model.predict(np.repeat(next_day_features, 2, axis=0))
    xgb_next_days = xgb_model.predict(np.repeat(next_day_features, 2, axis=0))

    return lr_model, rf_model, xgb_model, stock_data, [lr_next_days, rf_next_days, xgb_next_days]


# Plot interactive candlestick chart with predictions and future prices
def plot_interactive_candlestick(stock_data, time_frame, future_prices, title_suffix):
    stock_data = stock_data.tail(time_frame).copy()

    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=stock_data['Date'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name="Candlestick"
    ))

    # Add predicted price lines
    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['LR_Prediction'],
        mode='lines',
        name='Linear Regression Prediction',
        line=dict(color='orange', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['RF_Prediction'],
        mode='lines',
        name='Random Forest Prediction',
        line=dict(color='green', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['XGB_Prediction'],
        mode='lines',
        name='XGBoost Prediction',
        line=dict(color='red', width=2)
    ))

    # Add future predicted prices as markers
    future_dates = [stock_data['Date'].iloc[-1] + pd.Timedelta(days=i) for i in range(1, len(future_prices[0]) + 1)]
    for model_name, predictions, color in zip(
        ['Linear Regression', 'Random Forest', 'XGBoost'],
        future_prices,
        ['orange', 'green', 'red']
    ):
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='markers+text',
            name=f"{model_name} Future Prediction",
            text=[f"Day {i + 1}: ₹{price:.2f}" for i, price in enumerate(predictions)],
            textposition="top center",
            marker=dict(color=color, size=10, symbol='circle')
        ))

    # Add chart styling
    fig.update_layout(
        title=f"Candlestick Chart with Predictions ({title_suffix})",
        xaxis_title="Date",
        yaxis_title="Price (INR)",
        template="plotly_dark",
        xaxis_rangeslider_visible=False
    )

    fig.show()


# Main function
def main():
    print("📈 Welcome to the Interactive Stock Analysis Tool! 📈")
    ticker = input("Enter stock ticker symbol (e.g., RELIANCE.BO, TCS.BO): ").upper()
    stock_data, market_cap = fetch_stock_data(ticker)

    if stock_data is None:
        print("❌ No data found for the given ticker. Please try again.")
        return

    print(f"\n💰 Market Capitalization: ₹{market_cap / 1e7:.2f} Crores (if available)")

    print("\n📋 Last 7 Days OHLC Data:")
    last_7_days = stock_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(7)
    print(tabulate(last_7_days, headers='keys', tablefmt='fancy_grid', showindex=False))

    print("\n⚙️ Training models and predicting the next 2 days' prices...")
    lr_model, rf_model, xgb_model, prepared_data, future_prices = train_models(stock_data)

    print("\n📊 Next 2 Days Predicted Prices:")
    for i, day in enumerate(["Day 1", "Day 2"]):
        print(f"{day}:")
        print(f" - Linear Regression: ₹{future_prices[0][i]:.2f}")
        print(f" - Random Forest: ₹{future_prices[1][i]:.2f}")
        print(f" - XGBoost: ₹{future_prices[2][i]:.2f}\n")

    print("\n⏳ Short-term Predictions (7 Days):")
    plot_interactive_candlestick(prepared_data, time_frame=7, future_prices=future_prices, title_suffix="7 Days")

    print("\n⏳ Long-term Predictions (1 Month):")
    plot_interactive_candlestick(prepared_data, time_frame=30, future_prices=future_prices, title_suffix="1 Month")


if __name__ == "__main__":
    main()